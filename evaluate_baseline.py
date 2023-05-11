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

from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task

from evaluate import evaluate
from src.baselines import load_model

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()
    # force options to be compatible with the closed-book baseline models
    assert opt.closed_book, "The baseline must be closed-book"

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    model, _, _ = load_model(opt, eval_only=True)

    logger.info("Start Evaluation")
    dist_utils.barrier()

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")
        metrics = evaluate(model, None, opt, data_path, None)
        log_message = f"Dataset: {dataset_name}"
        for k, v in metrics.items():
            log_message += f" | {v:.3f} {k}"
        logger.info(log_message)
