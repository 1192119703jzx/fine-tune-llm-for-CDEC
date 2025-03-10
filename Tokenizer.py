import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from Dataloader import EventPairDataset
import json
import os

special_tokens = ["<TRG>", "</TRG>", "<ARG>", "</ARG>", "<TIME>", "</TIME>", "<LOC>", "</LOC>"]
tokenizer_name = 'roberta-base'
tokenizer_used = RobertaTokenizer.from_pretrained(tokenizer_name)

tokenizer_used.add_special_tokens({'additional_special_tokens': special_tokens})
len_tokenizer = len(tokenizer_used)

class Tokenizer(Dataset):
    def __init__(self, file_path, test=False, srl=False, load_data=False):
        self.path = file_path
        self.tokenizer = tokenizer_used
        self.test = test
        self.srl = srl
        self.load_data = load_data
        self.data = self.load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_from_file(self, load_path):
        self.data = torch.load(load_path)

    def load(self):
        if self.load_data:
            with open(self.path, 'r') as f:
                base_dataset = json.load(f)
        else:
            base_dataset_object = EventPairDataset(self.path, srl=self.srl, test=self.test)
            base_dataset = base_dataset_object.data
            if self.srl:
                file_name = os.path.basename(self.path)
                save_path = file_name.split('.')[1][1:] + '_srl.json'
                with open(save_path, 'w') as f:
                    json.dump(base_dataset.data, f)

        encode_dataset = []
        for item in base_dataset:
            encoded = self.tokenizer(
                item['sentence1'],
                item['sentence2'],
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encoded['label'] = torch.tensor(item['label'], dtype=torch.long)
            '''
            id1_tensor = torch.tensor(item['id1'], dtype=torch.long)
            id2_tensor = torch.tensor(item['id2'], dtype=torch.long)
            encoded['input_ids'] = torch.cat((id1_tensor, id2_tensor), dim=0)
            '''
            encode_dataset.append(encoded)
        
        return encode_dataset