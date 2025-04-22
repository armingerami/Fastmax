# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import glob
import json
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

from litgpt.tokenizer import Tokenizer
from litgpt.data import DataModule
from litgpt.data.alpaca import download_if_missing
from litgpt.data.text_files import validate_tokenizer


@dataclass
class Wiki40b(DataModule):


    data_path: Path = Path("data/wiki40b")
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step."""
    seed: int = 42
    """The seed to use for shuffling the dataset."""
    num_workers: int = 8
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.data_path_train = self.data_path / "train"
        self.data_path_val = self.data_path / "val"

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: int = -1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        from litdata import optimize

        # download(self.data_path)
        if not Path(self.data_path_train).is_dir():
            train_files = sorted(glob.glob(str(self.data_path_train / "all_data" / "train*.json")))
            val_file = sorted(glob.glob(str(self.data_path_val / "all_data" / "val*.json")))
            assert len(train_files) > 0, f"No json files found in"
            assert len(train_files) > 1, f"Expected at least two json files in"
        # train/test split. let's use only shard 0 for test split, rest train
        num_workers = os.cpu_count() - 1

        if not Path(self.data_path_train).is_dir():
            validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=train_files,
                output_dir=str(self.data_path_train),
                num_workers=num_workers,
                chunk_bytes="200MB",
            )
        if not Path(self.data_path_val).is_dir():
            validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=val_file,
                output_dir=str(self.data_path_val),
                num_workers=1,  # there's only 1 file
                chunk_bytes="200MB",
            )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=str(self.data_path_train),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )
        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, StreamingDataLoader, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=str(self.data_path_val),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )
        val_dataloader = StreamingDataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader


def tokenize(filename: str, tokenizer: Tokenizer):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    global_rank = int(os.environ["DATA_OPTIMIZER_GLOBAL_RANK"])
    num_workers = int(os.environ["DATA_OPTIMIZER_NUM_WORKERS"])
    local_rank = global_rank % num_workers
    for example in tqdm(data, position=local_rank):
        text = example["text"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = tokenizer.encode(text, bos=True, eos=False)  # encode the text, use BOS
        yield tokens


