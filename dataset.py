import numpy as np
import torch
import json
import h5py
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor, PreTrainedTokenizerBase
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

@dataclass
class DataCollatorWithPaddingSkippingFeatures:
    """
    Data collator that will dynamically pad the inputs received, adapted to Lxmert. Will not touch visual features.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    features_to_skip: tuple = ()

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        separated_features = {key: [example[key] for example in features] for key in features[0].keys()}
        features_to_pad = {key: separated_features[key] for key in separated_features.keys() if key not in self.features_to_skip}
        batch = self.tokenizer.pad(
            features_to_pad,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # add features that should not be padded to batch, if they are there
        for key in self.features_to_skip:
            if key in separated_features:
                batch[key] = torch.stack(separated_features[key], 0)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

def get_text_image_pretraining_dataset(train_path: str, val_path: str, tokenizer, train_image_features_path: str, valid_image_features_path: str):
    train_ds = TextImageDataset(train_path, tokenizer, train_image_features_path)
    val_ds = TextImageDataset(val_path, tokenizer, valid_image_features_path)
    return train_ds, val_ds


class TextImageDataset(Dataset):

    def __init__(self, examples_path: str, tokenizer, image_features_path: str=None):
        """A dataset for pretraining using text and optionally precomputed image features

        Args:
            examples_path (str): Path to jsonl file with fields "text", "answer" and optionally "image_id"
            tokenizer (BertTokenizer): Tokenizer to tokenize text
            image_features_path (str, optional): Path to hdf5 file with "features" and "ids" datasets. Defaults to None.
        """
        super().__init__()

        self.tokenizer = tokenizer

        # Load examples set
        self.examples = [json.loads(line) for line in open(examples_path)]

        # Load image features
        if image_features_path is not None:
            buffer = h5py.File(image_features_path, mode="r")
            self.image_features = buffer["features"]
            self.image_id2idx = {id: idx for idx, id in enumerate(buffer["ids"].asstr())}
        else:
            self.image_features = self.image_id2idx = None
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        preprocessed_example = self.tokenizer(example["text"], truncation=True)
        if "image_id" in example and self.image_features is not None:
            preprocessed_example["img_feats"] = np.array(self.image_features[self.image_id2idx[example["image_id"]]], dtype=np.float16)[np.newaxis, ...]
        return preprocessed_example