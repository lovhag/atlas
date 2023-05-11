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
import logging
from evaluate import evaluate
from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model, save_atlas_model
from src.options import get_options
from src.tasks import get_task

from train import train
from src.baselines import load_model

os.environ["TOKENIZERS_PARALLELISM"] = "true"
GRAD_SCALE_UPPER_BOUND_MEAN: int = 1000
GRAD_SCALE_LOWER_BOUND_MEAN: float = 0.01
THRESHOLD_GRAD_STATS: int = 100

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()
    
    # force options to be compatible with the closed-book baseline models
    assert opt.closed_book, "The baseline must be closed-book"
    opt.train_retriever = False

    # same as train.py
    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    # use another model initializer for the baselines
    model, optimizer, scheduler = load_model(opt)

    # same as train.py but with removal of potential retriever training
    if opt.is_distributed:
        if opt.shard_grads:
            import fairscale.nn.data_parallel

            model.reader = fairscale.nn.data_parallel.ShardedDataParallel(
                model.reader, optimizer, auto_refresh_trainable=False
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=True,
            )
            model._set_static_graph()

    logger.info("Start training")
    dist_utils.barrier()
    # remove arguments that shouldn't be used for the baseline
    train(
        model,
        None,
        None,
        optimizer,
        scheduler,
        None,
        None,
        None,
        opt,
        checkpoint_path,
    )
