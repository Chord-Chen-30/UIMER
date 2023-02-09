import sys
import pandas as pd
from difflib import Differ
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model import CrossEncoderForWNLI
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from utils import print_cmd, set_seed
from random import shuffle
from tqdm import tqdm
import random
import argparse

from Logger import MyLogger
from data_loader_nli import read_wnli_to_examples, find_rationale_wnli, convert_examples_to_features, augment_examples_and_conver_to_features
from data_loader_nli import read_esnli_to_examples, find_rationale_esnli
from data_loader_nli import modify_aug_flag_and_annotation_ids, gen_dataset, gen_dataset_aug


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(dataset_train, dataset_aug, dataset_dev, dataset_test, model, optimizer, scheduler, ce_loss, args, logger, seed):

    train_dataloader = DataLoader(dataset_train, batch_size=args.train_batch_size)
    train_aug_dataloader = DataLoader(dataset_aug, batch_size=args.train_batch_size)

    assert len(dataset_train) == len(dataset_aug), print("Different size of dataset_train and dataset_aug!")
    print("Train size: ", len(dataset_train))

    best_dev_acc = -1
    best_test_acc = -1
    early_stop = 0
    break_train_flag = False
    margin_loss, saliency_loss, loss_feng = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
    warming_up_flag = True if args.warmup_epoch > 0 else False

    for epoch in range(args.num_train_epochs):
        print("Epoch:", epoch)

        train_acc = 0.
        model.train()
        
        total_margin_loss, total_saliency_loss, total_loss_feng, total_ce_loss = 0, 0, 0, 0
        for step, (batch, batch_aug) in enumerate(tqdm(zip(train_dataloader, train_aug_dataloader), desc='Train')):
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            token_type_ids = batch[2].to(DEVICE)
            labels = batch[3].to(DEVICE)
            aug_flag = batch[4]
            annotation_ids = batch[5].to(DEVICE)

            _input = {'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids}

            logits = model(**_input)

            pred_correct = (logits.argmax(axis=1) == labels).float().sum()
            train_acc += pred_correct

            loss = ce_loss(logits, labels)
            total_ce_loss += loss.item()
            
            if args.weight != 0. and (not warming_up_flag):
                margin_loss = model.forward_aug(batch_aug, aug_flag)
                loss += margin_loss * args.weight
                total_margin_loss += margin_loss.item()
            
            if args.grad:
                saliency_loss = model.forward_saliency(logits, _input, labels,  annotation_ids)
                loss += saliency_loss * args.grad_weight
                total_saliency_loss += saliency_loss.item()

            if args.feng != 'none':
                loss_feng = model.forward_feng(loss, input_ids, annotation_ids)
                loss += loss_feng * args.weight
                total_loss_feng += loss_feng.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = epoch*len(train_dataloader) + step
            if global_step%args.eval_steps_interval == 0:
                dev_acc = evaluate(dataset_dev, mode='dev')
                test_acc = evaluate(dataset_test, mode='test')
                model.train()
                print("==================================")
                print("Dev Acc:", dev_acc.item())
                print("Test Acc:", test_acc.item())
                print("==================================")

                early_stop += 1
                print("Early_stop", early_stop)
                
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_test_acc = test_acc
                    early_stop = 0
                    # model.save_model(seed)
                
                if early_stop >= args.early_stop:
                    if args.warmup_epoch == 0:
                        break_train_flag = True
                        break
                    else:
                        if warming_up_flag:
                            warming_up_flag = False
                            early_stop = 0
                        else:
                            break_train_flag = True
                            break

        if (epoch+1) >= args.warmup_epoch:
            warming_up_flag = False
        
        scheduler.step()

        print("Train CE Loss:", total_ce_loss)
        print("Train Margin Loss:", total_margin_loss)
        print("Train Saliency Loss:", total_saliency_loss)
        print("Train Feng loss:", total_loss_feng)
        
        print("Train Acc:", train_acc.item()/len(dataset_train))
        
        if break_train_flag:
            break
        
    print("Best Dev:", best_dev_acc.item())
    print("Best Test:", best_test_acc.item())
    
    logger.result_info('best_dev_acc', best_dev_acc.item())
    logger.result_info('best_test_acc', best_test_acc.item())


@torch.no_grad()
def evaluate(dataset, mode='dev'):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=256)
    acc = 0.

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for batch in tqdm(dataloader, desc="Evaluate {}".format(mode)):
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        token_type_ids = batch[2].to(DEVICE)
        labels = batch[3].to(DEVICE)

        # for i in range(input_ids.size(0)):
        #     print(tokenizer.decode(input_ids[i]))
        #     print(labels[i])
        

        _input = {'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids}

        logits = model(**_input)

        # print("pred:")
        # print(logits.argmax(axis=1))

        # print("gold:")
        # print(labels)

        pred_correct = (logits.argmax(axis=1) == labels).float().sum()
        acc += pred_correct
    
    return acc/len(dataset)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default="30", type=str, help="Random seed")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Max epoch number")
    parser.add_argument("--early_stop", default=15, type=int, help="Early stop")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Eval Batch size")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--max_seq_len", default=120, type=int, help="Max sequence length")
    parser.add_argument("--dataset", default='wnli', type=str, help="NLI dataset to run. [wnli/esnli]")
    parser.add_argument("--model_dir", default='./ckpt/', type=str, help="Path to save checkpoints")
    parser.add_argument("--shot", default=-1, type=int, help="Number of training samples")
    parser.add_argument("--max_rationale_percentage", default=0.3, type=float, help="Max percentage of rationales we consider")
    parser.add_argument("--eval_steps_interval", default=50, type=int, help="Steps interval to do eval.")


    parser.add_argument("--margin", default=0.1, type=float, help="Margin in margin_loss")
    parser.add_argument("--weight", default=1., type=float, help="Weigth of margin loss")
    parser.add_argument("--grad_weight", default=1., type=float, help="Weigth of saliency loss")
    
    parser.add_argument("--lower_bound", default=1e-20, type=float, help="Lower bound used when calculating log-odds")

    parser.add_argument("--replace", default='random', type=str, help="Way to perform replacing, [none/random/frequency/bert]")
    parser.add_argument("--replace_repeat", default=4, type=int, help="Number of times to apply replacement on samples")
    parser.add_argument("--grad", default=0, type=int, help="Apply gradient-based method")
    
    parser.add_argument("--feng", default='base', type=str, help="[none/base/gate/order/gate+order]")
    parser.add_argument("--feng_weight", default=1.0, type=float, help="Weight of loss from Feng Yansong")
    
    parser.add_argument("--warmup_epoch", default=0, type=int, help="After [WARMUP_EPOCH] epochs, we add our loss to training objective")
    
    parser.add_argument("--verbose", default=0, type=int, help="Detailed print")
    parser.add_argument("--log_path", default='./log-temp/', type=str, help="Path of logs")
    
    # How Do Decisions Emerge..
    parser.add_argument("--num_train_extractor_epochs", default=10, type=int, help="num epochs to train extractor")
    parser.add_argument("--lr_extractor", default=1e-3, type=float, help="learning rate when training extractor")
    parser.add_argument("--gate", default='input', type=str, help="gate on [input/hidden]")
    parser.add_argument("--layer_pred", type=int, default=-1)
    parser.add_argument("--stop_train", type=int, default=1)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--weight_extractor", type=float, default=0.1)
    parser.add_argument("--loss_func_rationale", type=str, default='L2norm', help='L2norm/BCE/margin')
    parser.add_argument("--lr_decay", default=1, type=float, help="Used when running main_extractor.py")
    parser.add_argument("--load_model", type=int, default=1, help='Load a trained model')
    parser.add_argument("--total_round", type=int, default=1, help='Rounds of doing training')

    parser.add_argument("--gpu_name", type=str, default='-', help="Keep a record of GPU info")

    # Not used here
    parser.add_argument("--sub_task", type=str, default='', help="")

    args = parser.parse_args()

    # try:
    #     gpu_mem_G = torch.cuda.get_device_properties(0).total_memory /(1024.**3)
    #     if gpu_mem_G > 40:
    #         args.train_batch_size = 24
    #     elif gpu_mem_G >= 12:
    #         args.train_batch_size = 16
    #     elif gpu_mem_G < 12 and gpu_mem_G > 8:
    #         args.train_batch_size = 8
    #         print("Error, GPU memory -> train_batch_size=8, too small")
    #         exit(-1)
    #     else:
    #         print("Error, GPU memory too small")
    #         exit(-1)
    # except: # Except torch.cuda Error in different GPUs
    #     print("Batch size is not changed in args() function")

    try: 
        args.gpu_name = torch.cuda.get_device_name()
    except:
        print("Unable to obtain GPU name")
        pass


    mylogger = MyLogger(args.log_path, args, cmd=print_cmd(sys.argv))

    if args.dataset == 'wnli':
        all_examples = read_wnli_to_examples(path='./WNLI/WNLI/train.tsv')

        # Treat dev set as test
        all_examples += read_wnli_to_examples(path='./WNLI/WNLI/dev.tsv') 

        all_examples.sort(key=lambda x: ' '.join(x.sentence1))

        total_len = len(all_examples)

        all_examples_train = all_examples[:int(total_len*0.7)]
        all_examples_dev = all_examples[int(total_len*0.7):int(total_len*0.85)]
        all_examples_test = all_examples[int(total_len*0.85):]

        find_rationale_wnli(all_examples_train)
    
    elif args.dataset == 'esnli':
        all_examples_train = read_esnli_to_examples()
        all_examples_dev = read_esnli_to_examples(path=['./data/eSNLI/esnli_dev.csv'])
        all_examples_test = read_esnli_to_examples(path=['./data/eSNLI/esnli_test.csv'])

        if args.shot != -1:
            all_examples_train = all_examples_train[:args.shot]

        find_rationale_esnli(all_examples_train)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    features_train = convert_examples_to_features(args=args, examples=all_examples_train, max_seq_len=args.max_seq_len, tokenizer=tokenizer, )
    features_dev = convert_examples_to_features(args=args, examples=all_examples_dev, max_seq_len=args.max_seq_len, tokenizer=tokenizer, )
    features_test = convert_examples_to_features(args=args, examples=all_examples_test, max_seq_len=args.max_seq_len, tokenizer=tokenizer, )
    
    features_aug, aug_flags, annotation_ids = augment_examples_and_conver_to_features(all_examples_train, args.max_seq_len, tokenizer, replace=args.replace, args=args)
    modify_aug_flag_and_annotation_ids(features_train, aug_flags, annotation_ids)
    print('Features got!')

    dataset_train = gen_dataset(features_train, fold='train')
    dataset_dev = gen_dataset(features_dev, fold='dev')
    dataset_test = gen_dataset(features_test, fold='test')

    dataset_aug = gen_dataset_aug(features_aug)
    print("Dataset got!")

    for seed in args.seed.split(":"):
        set_seed(int(seed))
        print("Seed: ", seed)

        # model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-large')
        model = CrossEncoderForWNLI(args=args)
        model = model.to(DEVICE)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        ce_loss = CrossEntropyLoss(reduction='mean')
        t_total = len(dataset_train)/args.train_batch_size*args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=t_total)

        train(dataset_train, dataset_aug, dataset_dev, dataset_test, model, optimizer, scheduler, ce_loss, args, mylogger, seed)

    mylogger.cal_std_mean()
    mylogger.save()
    print("Success")