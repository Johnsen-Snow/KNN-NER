import argparse
from functools import partial
import json
import os
import numpy as np

from tqdm import tqdm

from utils import collate_to_max_length
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from transformers import BertConfig, BertForTokenClassification, RobertaConfig, RobertaForTokenClassification

from dataset import NERDataset
from utils import set_random_seed
from metrics import SpanF1ForNER

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/share/zhangsen/ner/ner_data/en_data/en_conll03', type=str, help='train data path')
    parser.add_argument('--bert_path', default='/share/zhangsen/ner/models/bert-base-cased',type=str, help='bert config file')
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument('--file_name', default='word.bmes', type=str, help='use for truncated sets.')
    parser.add_argument("--save_ner_prediction", action="store_true", help="only work for test.")
    parser.add_argument("--checkpoint_path", default="/share/zhangsen/MyLearning/KNN-NER/con_bert_base/model_epoch_8.pt", type=str, help="use for evaluation.")
    parser.add_argument("--datastore_path", default="/share/zhangsen/MyLearning/KNN-NER/result/con_bert_base", type=str, help="use for saving datastore.")
    parser.add_argument("--link_temperature", default=1.0, type=float, help="temperature used by edge linking.")
    parser.add_argument("--link_ratio", default=0.0, type=float, help="ratio of vocab probs predicted by edge linking.")
    parser.add_argument("--topk", default=64, type=int, help="use topk-scored neighbor tgt nodes for link prediction and probability compuation.")
    parser.add_argument("--en_roberta", action="store_true", help="whether load roberta for classification or not.")

    return parser

def get_dataloader(args) -> DataLoader:
    vocab_file = args.bert_path
    if not args.en_roberta:
        vocab_file = os.path.join(args.bert_path, "vocab.txt")

    dataset = NERDataset(directory=args.data_dir, prefix="test",
                                    vocab_file=vocab_file,
                                    max_length=args.max_length,
                                    config_path=os.path.join(args.bert_path, "config"),
                                    file_name=args.file_name, en_roberta=args.en_roberta)

    batch_size = args.batch_size
    data_sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                            num_workers=args.workers, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                            drop_last=False)

    info = json.load(open(os.path.join(args.datastore_path, "datastore_info.json")))
    key_file = os.path.join(args.datastore_path, "keys.npy")
    keys = np.memmap(key_file, 
                    dtype=np.float32,
                    mode="r",
                    shape=(info['token_sum'], info['hidden_size']))
    keys_in_memory = np.zeros((info['token_sum'], info['hidden_size']), dtype=np.float32)
    keys_in_memory[:] = keys[:]
    
    args.keys = torch.from_numpy(keys_in_memory).transpose(0, 1).cuda()
    
    val_file = os.path.join(args.datastore_path, "vals.npy")
    vals = np.memmap(val_file, 
                    dtype=np.int32,
                    mode="r",
                    shape=(info['token_sum'],))
    vals_in_memory = np.zeros((info['token_sum'],), dtype=np.int64)
    vals_in_memory[:] = vals[:]
    
    args.vals = torch.from_numpy(vals_in_memory).cuda()

    args.norm_1 = (args.keys ** 2).sum(dim=0, keepdim=True).sqrt()
    
    args.link_temperature = torch.tensor(args.link_temperature).cuda()

    args.link_ratio = torch.tensor(args.link_ratio).cuda()

    return dataloader

def postprocess_logits_to_labels(args, logits, hidden):
    """input logits should in the shape [batch_size, seq_len, num_labels]"""
    probabilities = F.softmax(logits, dim=2) # shape of [batch_size, seq_len, num_labels]

    batch_size = hidden.shape[0]
    sent_len = hidden.shape[1]
    hidden_size = hidden.shape[-1]
    token_num = args.keys.shape[1]
    
    # cosine similarity
    hidden = hidden.view(-1, hidden_size) # [bsz*sent_len, feature_size]
    sim = torch.mm(hidden, args.keys) # [bsz*sent_len, token_num]
    norm_2 = (hidden ** 2).sum(dim=1, keepdim=True).sqrt() # [bsz*sent_len, 1]
    scores = (sim / (args.norm_1 + 1e-10) / (norm_2 + 1e-10)).view(batch_size, sent_len, -1) # [bsz, sent_len, token_num]
    knn_labels = args.vals.view(1, 1, token_num).expand(batch_size, sent_len, token_num) # [bsz, sent_len, token_num]
    
    if (args.topk != -1 and scores.shape[-1] > args.topk):
        topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=args.topk)  # [bsz, sent_len, topk]
        scores = topk_scores
        knn_labels = knn_labels.gather(dim=-1, index=topk_idxs)  # [bsz, sent_len, topk]
    
    sim_probs = torch.softmax(scores / args.link_temperature, dim=-1) # [bsz, sent_len, token_num]
    
    knn_probabilities = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1, 1, args.num_labels])  # [bsz, sent_len, num_labels]
    knn_probabilities = knn_probabilities.scatter_add(dim=2, index=knn_labels, src=sim_probs) # [bsz, sent_len, num_labels]

    probabilities = args.link_ratio*knn_probabilities + (1-args.link_ratio)*probabilities

    argmax_labels = torch.argmax(probabilities, 2, keepdim=False) # [bsz, sent_len]
    return argmax_labels


def evaluate(args, data_iter, model):
    model.eval()
    ner_evaluation_metric = SpanF1ForNER(entity_labels=args.entity_labels, save_prediction=args.save_ner_prediction)
    with torch.no_grad():
        outputs = []
        for batch in tqdm(data_iter, ncols=80, desc='evaluating...'):
            input_ids, gold_labels = batch
            input_ids = input_ids.to(args.device)
            gold_labels = gold_labels.to(args.device)
            loss_mask = (input_ids != 0).long()
            bert_classifiaction_outputs = model(input_ids=input_ids)

            argmax_labels = postprocess_logits_to_labels(args, bert_classifiaction_outputs.logits, bert_classifiaction_outputs.hidden_states[-1])
            confusion_matrix = ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=loss_mask)

            outputs.append({"confusion_matrix": confusion_matrix})
        
        confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix

        if args.save_ner_prediction:
            precision, recall, f1, entity_tuple = ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative, prefix="test")
            gold_entity_lst, pred_entity_lst = entity_tuple
            # save_predictions_to_file(args, gold_entity_lst, pred_entity_lst, current_epoch, args.save_path)
        else:
            precision, recall, f1 = ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)
        print(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} , link_temperature: {args.link_temperature}, link_ratio: {args.link_ratio}")
        return {"test_f1": f1, "test_precision": precision, "test_recall": recall}

def inference(args, model):
    dataloader = get_dataloader(args)
    result = evaluate(args, dataloader, model)
    print(result)

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
    inference(args, model)

if __name__ == '__main__':
    main()
