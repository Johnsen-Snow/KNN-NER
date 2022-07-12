import argparse
import os
import torch
from transformers import BertConfig, BertForTokenClassification, RobertaConfig, RobertaForTokenClassification

from dataset import NERDataset
from ner_trainer import train, inference
from utils import init_logger, set_random_seed

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/share/zhangsen/ner/ner_data/en_data/en_conll03', type=str, help='train data path')
    parser.add_argument('--model_path', default='/share/zhangsen/MyLearning/KNN-NER/con_bert_large', type=str, help='use for evaluation.')
    parser.add_argument('--bert_path', default='/share/zhangsen/ner/models/bert-large-cased',type=str, help='bert config file')
    parser.add_argument('--log_file', default='/share/zhangsen/MyLearning/KNN-NER/logs/con_bert_large.log')
    parser.add_argument('--save_path', default='/share/zhangsen/MyLearning/KNN-NER/result/ontonotes5', type=str, help='train data path')
    parser.add_argument('--path_to_model_hparams_file', default='', type=str, help='use for evaluation')
    parser.add_argument("--save_ner_prediction", action="store_true", help="only work for test.")
    parser.add_argument('--file_name', default='word.bmes', type=str, help='use for truncated sets.')
    parser.add_argument('--seed', type=int, default=2333)
    
    parser.add_argument('--train_batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--workers', type=int, default=0, help='num workers for dataloader')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='Weight decay if we apply some.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_length', default=512, type=int, help='max length of dataset')
    parser.add_argument('--warmup_proportion', default=0.001, type=float, help='Proportion of training to perform linear learning rate warmup for.')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='torch.adam')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--model_val_per_epoch', default=1, type=int)
    
    parser.add_argument('--lower_case', default=False, type=bool, help='lowercase when load English data.')
    parser.add_argument('--language', default='en', type=str, help='the language of the dataset.')
    parser.add_argument('--en_roberta', action='store_true', help='whether load roberta for classification or not.')

    parser.add_argument('--do_train', default=True, type=bool)
    parser.add_argument('--do_inference', default=True, type=bool)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    init_logger(args.log_file)

    set_random_seed(args.seed)

    entity_labels = NERDataset.get_labels(os.path.join(args.data_dir, 'ner_labels.txt'))
    args.entity_labels = entity_labels
    num_labels = len(entity_labels)
    args.num_labels = num_labels

    if not args.en_roberta:
        bert_config = BertConfig.from_pretrained(args.bert_path, output_hidden_states=True, return_dict=True,
                                                 num_labels=num_labels, hidden_dropout_prob=args.hidden_dropout_prob)
        model = BertForTokenClassification.from_pretrained(args.bert_path, config=bert_config)
    else:
        bert_config = RobertaConfig.from_pretrained(args.bert_path, output_hidden_states=True, return_dict=True,
                                                 num_labels=num_labels, hidden_dropout_prob=args.hidden_dropout_prob)
        model = RobertaForTokenClassification.from_pretrained(args.bert_path, config=bert_config)

    if args.do_train:
        train(args, model)

    if args.do_inference:
        inference(args, model)

if __name__ == '__main__':
    main()
