# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
from pathlib import Path

from src import dist_utils, slurm, util
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task
from evaluate import _get_eval_data_iterator

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def save_distributed_torch_dataset(data, dataset_name, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / "tmp_dir_t"
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f"{opt.global_rank}.pt"
    torch.save(data, tmp_path)
    if opt.is_distributed:
        torch.distributed.barrier()
    if opt.is_main:
        final_path = dir_path / f"{dataset_name}.pt"
        logger.info(f"Saving embeddings at {final_path}")
        results_path = list(write_path.glob("*.pt"))
        results_path.sort()

        alldata = []
        for path in results_path:
            data = torch.load(path)
            alldata.append(data)
            path.unlink()
        alldata = torch.cat(alldata, 0)
        torch.save(alldata, final_path)
        write_path.rmdir()

@torch.no_grad()
def run_query_embedding_only(model, opt, data_path, step=None):
    model.eval()
    dataset_wpred = []
    q_embeddings = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)
    unwrapped_model.retriever.eval()

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        query_enc = model.retriever_tokenize(query)
        query_emb = unwrapped_model.retriever(query_enc["input_ids"].cuda(), query_enc["attention_mask"].cuda(), is_passages=False)
        
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue

        q_embeddings.append(query_emb.detach().cpu())

        for k in range(len(query_emb)):
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
            ex = {"query": query[k], "answers": gold}
            if batch_metadata is not None:
                ex["metadata"] = batch_metadata[k]
            if "id" in batch:
                ex["id"] = batch["id"][k]
            if "sub_label" in batch:
                ex["sub_label"] = batch["sub_label"][k]
            if "pattern" in batch:
                ex["pattern"] = batch["pattern"][k]
            dataset_wpred.append(ex)

    dataset_name, _ = os.path.splitext(os.path.basename(data_path))
    dataset_name = f"{dataset_name}-step-{step}-r-embedding"
    util.save_distributed_dataset(dataset_wpred, dataset_name, opt)
    
    save_distributed_torch_dataset(torch.cat(q_embeddings, dim=0), dataset_name, opt)
    logger.info(f"Retriever embeddings saved")

    return

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)

    logger.info("Start generating retriever embeddings")
    dist_utils.barrier()

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start generating for {data_path}")
        run_query_embedding_only(model, opt, data_path, step)
