from functools import partial
import glob
import os
import time

import torch
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

from dataset import NERDataset
from utils import collate_to_max_length, logger
from metrics import SpanF1ForNER

def load_dataset(args, prefix="test"):
    vocab_file = args.bert_path
    if not args.en_roberta:
        vocab_file = os.path.join(args.bert_path, "vocab.txt")
    dataset = NERDataset(directory=args.data_dir, prefix=prefix,
                         vocab_file=vocab_file,
                         max_length=args.max_length,
                         config_path=os.path.join(args.bert_path, "config"),
                         file_name=args.file_name, lower_case=args.lower_case,
                         language=args.language, en_roberta=args.en_roberta)

    return dataset

def get_dataloader(args, prefix="train", limit=None) -> DataLoader:
    """return {train/dev/test} dataloader"""
    dataset = load_dataset(args, prefix=prefix)

    if prefix == "train":
        batch_size = args.train_batch_size
        # small dataset like weibo ner, define data_generator will help experiment reproducibility.
        data_generator = torch.Generator()
        data_generator.manual_seed(args.seed)
        data_sampler = RandomSampler(dataset, generator=data_generator)
    else:
        batch_size = args.eval_batch_size
        data_sampler = SequentialSampler(dataset)

    # sampler option is mutually exclusive with shuffle
    dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                            num_workers=args.workers, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                            drop_last=False)

    return dataloader, dataset

def compute_loss(args, logits, labels, loss_mask=None):
    """
    Desc:
        compute cross entropy loss
    Args:
        logits: FloatTensor, shape of [batch_size, sequence_len, num_labels]
        labels: LongTensor, shape of [batch_size, sequence_len, num_labels]
        loss_mask: Optional[LongTensor], shape of [batch_size, sequence_len].
            1 for non-PAD tokens, 0 for PAD tokens.
    """
    loss_fct = CrossEntropyLoss()
    if loss_mask is not None:
        active_loss = loss_mask.view(-1) == 1
        active_logits = logits.view(-1, args.num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        loss = loss_fct(active_logits, active_labels)
    else:
        loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
    return loss

def postprocess_logits_to_labels(logits):
    """input logits should in the shape [batch_size, seq_len, num_labels]"""
    probabilities = F.softmax(logits, dim=2) # shape of [batch_size, seq_len, num_labels]
    argmax_labels = torch.argmax(probabilities, 2, keepdim=False) # shape of [batch_size, seq_len]
    return probabilities, argmax_labels

def save_predictions_to_file(args, gold_entity_lst, pred_entity_lst, current_epoch, save_path, prefix="test"):
    dataset = load_dataset(args, prefix=prefix)
    data_items = dataset.data_items

    save_file_path = os.path.join(save_path, f"test_predictions_{current_epoch}.txt")
    print(f"INFO -> write predictions to {save_file_path}")

    def _write(f, label_item):
        for item in label_item:
            label_name = item[0]
            offset_s, offset_e = str(item[1][0]), str(item[1][1])
            f.write(label_name + ': ' + f'({offset_s}, {offset_e})\n')

    with open(save_file_path, "w") as f:
        for gold_label_item, pred_label_item, data_item in zip(gold_entity_lst, pred_entity_lst, data_items):
            data_tokens = data_item[0]
            f.write("=!" * 20+"\n")
            f.write(" ".join(data_tokens)+"\n")
            f.write('gold_label_item: \n')
            _write(f, gold_label_item)
            f.write('pred_label_item: \n')
            _write(f, pred_label_item)

def evaluate(args, data_iter, model, current_epoch, global_step=0, inference=False):
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

            probabilities, argmax_labels = postprocess_logits_to_labels(bert_classifiaction_outputs.logits)
            confusion_matrix = ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=loss_mask)

            if not inference:
                loss = compute_loss(args, bert_classifiaction_outputs.logits, gold_labels, loss_mask=loss_mask)
                outputs.append({"val_loss": loss, "confusion_matrix": confusion_matrix})
            else:
                outputs.append({"confusion_matrix": confusion_matrix})
        
        confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix

        if not inference:
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            precision, recall, f1 = ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)
            logger.info(f"EVAL INFO -> current_epoch is: {current_epoch}, current_global_step is: {global_step}, loss is: {avg_loss}")
            logger.info(f"EVAL INFO -> valid_f1 is: {f1}, val_precision is: {precision}, val_recall is: {recall}")
            return {"val_loss": avg_loss, "val_f1": f1, "val_precision": precision, "val_recall": recall}
        else:
            if args.save_ner_prediction:
                precision, recall, f1, entity_tuple = ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative, prefix="test")
                gold_entity_lst, pred_entity_lst = entity_tuple
                save_predictions_to_file(args, gold_entity_lst, pred_entity_lst, current_epoch, args.save_path)
            else:
                precision, recall, f1 = ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)
            logger.info(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} ")
            return {"test_f1": f1, "test_precision": precision, "test_recall": recall}


def train(args, model):
    model.to(args.device)
    logger.info(model)

    tb_writer = SummaryWriter(args.model_path)

    train_dataloader, train_dataset = get_dataloader(args, prefix='train')

    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if args.optimizer == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), lr=args.lr, eps=args.adam_epsilon)
    elif args.optimizer == "torch.adamw":
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    elif args.optimizer == "torch.adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)
    else:
        raise ValueError("Please import the Optimizer first. ")
    
    t_total = len(train_dataloader) // args.accumulate_grad_batches * args.epochs
    warmup_steps = int(args.warmup_proportion * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps = t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.accumulate_grad_batches)
    logger.info("  Gradient Accumulation steps = %d", args.accumulate_grad_batches)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, max_f1 = 0, 0.0

    for epoch in range(args.epochs):
        losses = 0
        start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            model.train()
            input_ids, labels = batch
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            loss_mask = (input_ids != 0).long()
            bert_classifiaction_outputs = model(input_ids=input_ids, attention_mask=loss_mask)
            loss = compute_loss(args, bert_classifiaction_outputs.logits, labels, loss_mask=loss_mask)
            acc = (bert_classifiaction_outputs.logits.argmax(2) == labels).float().mean()
            optimizer.zero_grad()

            losses += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if step % 10 == 0:
                logger.info(f"Epoch{epoch}:  Batch[{step}/{len(train_dataloader)}]  Train loss :{loss.item():.3f}"
                            f"  Acc: {acc:.3f}  Cost time: {time.time() - start_time:6.0f}")
            tb_writer.add_scalar('lr1', optimizer.param_groups[0]['lr'], global_step)
            tb_writer.add_scalar('lr2', optimizer.param_groups[1]['lr'], global_step)
            tb_writer.add_scalar('loss', loss.item(), global_step)
            tb_writer.add_scalar('acc', acc, global_step)
            tb_writer.add_scalar('epoch', epoch, global_step)
        
        logger.info(f"Epoch: {epoch}, Train loss: {losses / len(train_dataloader):.3f}, Epoch time = {(time.time() - start_time):6.0f}")

        if (epoch + 1) % args.model_val_per_epoch == 0:
            val_iter, val_dataset = get_dataloader(args, prefix='dev')
            result = evaluate(args, val_iter, model, epoch, global_step)
            f1 = result['val_f1']
            if f1 > max_f1:
                max_f1 = f1
            checkpoint_path = os.path.join(args.model_path, 'model_epoch_%d.pt' % epoch)
            logger.info(f'val_f1 reached {f1} (best {max_f1}), saving checkpoint to {checkpoint_path}')
            if (not os.path.exists(checkpoint_path)):
                torch.save(model.state_dict(), checkpoint_path)

def inference(args, model):
    checkpoints = list(sorted(glob.glob(args.model_path + '/**/' + 'model_epoch_*.pt', recursive=True)))

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    results = {}
    for checkpoint in checkpoints:
        # Reload the model
        epoch = checkpoint.split('_')[-1].split('.')[0]
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)
        model.to(args.device)

        # Evaluate
        test_iter, test_dataset = get_dataloader(args, prefix='test')
        result = evaluate(args, test_iter, model, epoch, inference=True)

        result = dict((k + ('_{}'.format(epoch) if epoch else ''), v) for k, v in result.items())
        results.update(result)

    logger.info("Results: {}".format(results))

