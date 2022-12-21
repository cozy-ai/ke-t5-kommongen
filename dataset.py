from torch.utils.data import Dataset
import os

class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_source_length=32, max_target_length=32):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        self.inputs = self.encode_file(self.tokenizer, os.path.join(self.data_dir, self.type_path + ".source"),
                                       self.max_source_length)
        self.targets = self.encode_file(self.tokenizer, os.path.join(self.data_dir, self.type_path + ".target"),
                                        self.max_target_length)

    def encode_file(self, tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
        examples = []
        with open(data_path, "r") as f:
            for text in f.readlines():
                tokenized = tokenizer.batch_encode_plus(
                    [text], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
                    truncation=True
                )
                examples.append(tokenized)
        return examples