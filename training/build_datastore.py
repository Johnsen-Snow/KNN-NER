import argparse
from functools import partial
import json
import os
import numpy as np

from tqdm import tqdm

from utils import collate_to_max_length
import torch
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from transformers import BertConfig, BertForTokenClassification, RobertaConfig, RobertaForTokenClassification

from dataset import NERDataset
from utils import set_random_seed

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/share/zhangsen/ner/ner_data/en_data/en_conll03', type=str, help='train data path')
    parser.add_argument('--bert_path', default='/share/zhangsen/ner/models/bert-base-cased',type=str, help='bert config file')
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument('--file_name', default='word.bmes', type=str, help='use for truncated sets.')
    parser.add_argument("--checkpoint_path", default="/share/zhangsen/MyLearning/KNN-NER/con_bert_base/model_epoch_8.pt", type=str, help="use for evaluation.")
    parser.add_argument("--datastore_path", default="/share/zhangsen/MyLearning/KNN-NER/result/con_bert_base", type=str, help="use for saving datastore.")
    parser.add_argument("--en_roberta", action="store_true", help="whether load roberta for classification or not.")

    return parser

def get_dataloader(args) -> DataLoader:
    vocab_file = args.bert_path
    if not args.en_roberta:
        vocab_file = os.path.join(args.bert_path, "vocab.txt")
    dataset = NERDataset(directory=args.data_dir, prefix="train",
                         vocab_file=vocab_file,
                         max_length=args.max_length,
                         config_path=os.path.join(args.bert_path, "config"),
                         file_name=args.file_name, en_roberta=args.en_roberta)

    batch_size = args.batch_size
    data_sampler = SequentialSampler(dataset)

    # sampler option is mutually exclusive with shuffle
    dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                            num_workers=args.workers, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                            drop_last=False)

    return dataloader

def build(args, model):
    model.eval()

    dataloader = get_dataloader(args)

    with torch.no_grad():
        outputs = []
        for batch in tqdm(dataloader, ncols=80, desc='loading data...'):
            input_ids, gold_labels = batch
            input_ids = input_ids.to(args.device)
            gold_labels = gold_labels.to(args.device)
            sequence_mask = (input_ids != 0).long()
            features = model(input_ids=input_ids).hidden_states[-1]
            
            outputs.append({"features": features, "labels": gold_labels, "mask": sequence_mask})

        hidden_size = outputs[0]['features'].shape[2]
        token_sum = sum(int(x['mask'].sum(dim=-1).sum(dim=-1).cpu()) for x in outputs)

        data_store_key_in_memory = np.zeros((token_sum, hidden_size), dtype=np.float32)
        data_store_val_in_memory = np.zeros((token_sum,), dtype=np.int32)

        now_cnt = 0
        for x in tqdm(outputs, ncols=80, desc='buidling...'):
            features = x['features'].reshape(-1, hidden_size)
            mask = x['mask'].bool()
            labels = torch.masked_select(x['labels'], mask).cpu().numpy()
            mask = mask.reshape(features.shape[0], 1).expand(features.shape[0], features.shape[1])
            features = torch.masked_select(features, mask).view(-1, hidden_size).cpu()
            np_features = features.numpy().astype(np.float32)
            data_store_key_in_memory[now_cnt:now_cnt+np_features.shape[0]] = np_features
            data_store_val_in_memory[now_cnt:now_cnt+np_features.shape[0]] = labels
            now_cnt += np_features.shape[0]

        datastore_info = {
            "token_sum": token_sum,
            "hidden_size": hidden_size
        }
        json.dump(datastore_info, open(os.path.join(args.datastore_path, "datastore_info.json"), "w"),
                    sort_keys=True, indent=4, ensure_ascii=False)

        key_file = os.path.join(args.datastore_path, "keys.npy")
        keys = np.memmap(key_file, 
                     dtype=np.float32,
                     mode="w+",
                     shape=(token_sum, hidden_size))

        val_file = os.path.join(args.datastore_path, "vals.npy")
        vals = np.memmap(val_file, 
                     dtype=np.int32,
                     mode="w+",
                     shape=(token_sum,))
        
        keys[:] = data_store_key_in_memory[:]
        vals[:] = data_store_val_in_memory[:]

        print({"saved dir": args.datastore_path})

        

def main():
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    set_random_seed(args.seed)

    entity_labels = NERDataset.get_labels(os.path.join(args.data_dir, 'ner_labels.txt'))
    args.entity_labels = entity_labels
    num_labels = len(entity_labels)
    args.num_labels = num_labels

    if not args.en_roberta:
        bert_config = BertConfig.from_pretrained(args.bert_path, output_hidden_states=True, return_dict=True, num_labels=num_labels)
        model = BertForTokenClassification.from_pretrained(args.bert_path, config=bert_config)
    else:
        bert_config = RobertaConfig.from_pretrained(args.bert_path, output_hidden_states=True, return_dict=True, num_labels=num_labels)
        model = RobertaForTokenClassification.from_pretrained(args.bert_path, config=bert_config)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to(args.device)
    build(args, model)

if __name__ == '__main__':
    main()
