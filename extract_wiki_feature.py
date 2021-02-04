# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForQuestionAnswering,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = os.path.join(args.data_dir, f"{args.language}.txt")
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def extract_feature(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Extracting features {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    sample_size = 0
    summations = []
    ## first time calculate summations
    for batch in tqdm(eval_dataloader, desc="Calculate Mean"):
        with torch.no_grad():
            batch = batch.to(args.device)
            outputs = model(batch)
            logits, hidden_states = outputs
            tmp_hidden_states = []
            if args.target.lower()=="cls":
                for i, state in enumerate(hidden_states):
                    tmp_hidden_states.append(state[:,0,:])
            elif args.target=="words":
                for i, state in enumerate(hidden_states):
                    state = state[:,1:,:]
                    mask = (batch[:,1:]!=0).unsqueeze(2)
                    state = mask*state
                    state = torch.sum(state, dim=1)/torch.sum(mask, dim=1)
                    tmp_hidden_states.append(state)
            elif args.target=="all":
                for i, state in enumerate(hidden_states):
                    mask = (batch!=0).unsqueeze(2)
                    state = mask*state
                    state = torch.sum(state, dim=1)/torch.sum(mask, dim=1)
                    tmp_hidden_states.append(state)
            else:
                raise NotImplementedError()
            hidden_states = tmp_hidden_states
            assert hidden_states[0].dim() == 2
            if len(summations)==0:
                for state in hidden_states:
                    summations.append(torch.sum(state, dim=0))
                assert len(summations)==len(hidden_states)
            else:
                for i, state in enumerate(hidden_states):
                    summations[i] += torch.sum(state, dim=0)
            sample_size += len(hidden_states[0])
            if sample_size >= args.max_sample_size:
                break

    # assert sample_size==len(eval_dataset)
    mean = [s/sample_size for s in summations]
    
    ## second time calculate variance
    #  summations = []
    #  sample_size = 0
    #  for batch in tqdm(eval_dataloader, desc="Calculate Variance"):
    #      with torch.no_grad():
    #          batch = batch.to(args.device)
    #          outputs = model(batch)
    #          logits, hidden_states = outputs
    #          tmp_hidden_states = []
    #          if args.target.lower()=="cls":
    #              for i, state in enumerate(hidden_states):
    #                  tmp_hidden_states.append(state[:,0,:])
    #          elif args.target=="words":
    #              for i, state in enumerate(hidden_states):
    #                  state = state[:,1:,:]
    #                  mask = (batch[:,1:]!=0).unsqueeze(2)
    #                  state = mask*state
    #                  tmp_hidden_states.append(state)
    #          elif args.target=="all":
    #              for i, state in enumerate(hidden_states):
    #                  mask = (batch!=0).unsqueeze(2)
    #                  state = mask*state
    #                  tmp_hidden_states.append(state)
    #          else:
    #              raise NotImplementedError()
    #          hidden_states = tmp_hidden_states
    #          if args.target.lower()=="cls":
    #              if len(summations)==0:
    #                  for state, m in zip(hidden_states, mean):
    #                      summations.append(torch.sum((state-m.unsqueeze(0))**2, dim=0))
    #                  assert len(summations)==len(hidden_states)
    #              else:
    #                  for i, (state, m) in enumerate(zip(hidden_states, mean)):
    #                      summations[i] += torch.sum((state-m.unsqueeze(0))**2, dim=0)
    #              sample_size = len(eval_dataset)
    #          elif args.target=="words" or args.target=="all": 
    #              if len(summations)==0:
    #                  for state, m in zip(hidden_states, mean):
    #                      delta = ((state-m.unsqueeze(0))**2)*mask
    #                      summations.append(torch.sum(delta, dim=(0, 1)))
    #                  assert len(summations)==len(hidden_states)
    #              else:
    #                  for i, (state, m) in enumerate(zip(hidden_states, mean)):
    #                      delta = ((state-m.unsqueeze(0))**2)*mask
    #                      summations[i] += torch.sum(delta, dim=(0, 1)) 
    #              sample_size += torch.sum(mask)
    #          else:
    #              raise NotImplementedError
    #  variance = [s/(sample_size) for s in summations]
    #  statistic_variance = [s/(sample_size-1) for s in summations]
    #  assert len(mean) == len(variance) == len(statistic_variance) == 13
    #  assert mean[0].shape == variance[0].shape == statistic_variance[0].shape == (768,)
    output_file = os.path.join(eval_output_dir, f"{prefix}.pkl")
    with open(output_file, "wb") as fout:
        logger.info(f"***** Saving features {prefix}.pkl *****")
        mean = np.array([m.to('cpu').numpy() for m in mean])
        # variance = np.array([v.to('cpu').numpy() for v in variance])
        # statistic_variance = np.array([v.to('cpu').numpy() for v in statistic_variance])
        # pickle.dump({'mean': mean,
        #              'variance': variance,
        #              'statistic_variance': statistic_variance,
        #              'sample_size': sample_size},
        #             fout)
        pickle.dump({'mean': mean}, fout) 


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--language",
        default=None,
        type=str,
        required=True,
        help="Evaluation language. Also train language if `train_language` is set to None.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument(
        "--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--target", type=str, default="all", help="target token")
    parser.add_argument("--max_sample_size", type=int, default=float("inf"), help="max_sample_size")
    args = parser.parse_args()
    
    args.max_sample_size = int(512/args.block_size) * args.max_sample_size

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            output_hidden_states=True,
        )
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        logger.info("Extrating Features")
        model_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
        extract_feature(args, model, tokenizer, f"{args.language}_{model_name}")

if __name__ == "__main__":
    main()
