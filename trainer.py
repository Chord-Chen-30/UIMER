import os
import logging
from re import L
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from transformers import BertTokenizer
from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels, print_cmd
import sys
import copy
import time
import pdb

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, train_dataset_aug=None, cur_seed=None, logger=None, dev_dataset_aug=None, test_dataset_aug=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.train_dataset_aug = train_dataset_aug
        self.dev_dataset_aug = dev_dataset_aug
        self.test_dataset_aug = test_dataset_aug

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

        self.cur_seed = cur_seed
        self.logger = logger

        self.best_model_state_dict = None

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        best_dev_acc = -1

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results_dev = self.evaluate("dev")
                        results_test = self.evaluate("test")

                        _metric = 'intent_acc' if self.args.sub_task == 'intent' else 'slot_f1'
                        if results_dev[_metric] > best_dev_acc:
                            best_dev_acc = results_dev[_metric]
                            best_test_acc = results_test[_metric]


                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model_online()
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            self.logger.acc_info("dev_acc", self.cur_seed, results_dev[_metric])
            self.logger.acc_info("test_acc", self.cur_seed, results_test[_metric])

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        
        self.logger.result_info('best_dev_acc', best_dev_acc)
        self.logger.result_info('best_test_acc', best_test_acc)

        return global_step, tr_loss / global_step


    def train_aug(self):
        # train_sampler = RandomSampler(self.train_dataset)
        # train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size)
        train_aug_dataloader = DataLoader(self.train_dataset_aug, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        best_dev_acc, best_test_acc = -1, -1
        early_stop = 0
        nan_flag = 0
        warming_up_flag = True if self.args.warmup_epoch > 0 else False
        margin_loss, saliency_loss, feng_loss= torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        # tokenizer_debug = BertTokenizer.from_pretrained('./bert-base-uncased')
        print(self.args.warmup_epoch)
        print()
        for epoch in train_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            # epoch_iterator_aug = tqdm(train_aug_dataloader)

            loss_origin_log = 0
            loss_attr_log = 0

            for step, (batch, batch_aug) in enumerate(zip(train_dataloader, train_aug_dataloader)):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                batch_aug = tuple(t.to(self.device) for t in batch_aug)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                
                annotation_ids = batch[6]
                # print("Origin data forward train, step={}".format(step))
                # print(tokenizer_debug.convert_ids_to_tokens(inputs['input_ids'][0]))
                # print(inputs['attention_mask'])
                # print(inputs['intent_label_ids'])
                # print(inputs['slot_labels_ids'])
                # print(inputs['token_type_ids'])

                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.verbose:
                    print("Orgin Loss:", loss.item())
                loss_origin_log += loss

                if self.args.weight != 0 and (not warming_up_flag):
                    aug_flags = batch[5]
                    margin_loss = self.model.forward_aug(batch_aug, aug_flags)
                    loss += margin_loss * self.args.weight

                if self.args.grad:
                    intent_logits, _ = outputs[1]
                    slot_logits = outputs[-1]

                    logits = intent_logits if self.args.sub_task == 'intent' else slot_logits
                    saliency_loss = self.model.forward_saliency(logits, inputs, annotation_ids)
                    loss += saliency_loss * self.args.grad_weight

                if self.args.feng != 'none':
                    feng_loss = self.model.forward_feng(loss, inputs['input_ids'], annotation_ids)
                    loss += feng_loss * self.args.feng_weight

                if self.args.verbose:
                    print("Saliency Loss:", saliency_loss)
                    print("Margin Loss: ", margin_loss)
                    print("Feng Loss:", feng_loss)

                loss_attr_log += (margin_loss.item() + saliency_loss.item() + feng_loss.item())

                if torch.isnan(margin_loss):
                    print("Margin loss NAN Error")
                    exit(-1)
                    
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                nan_flag = 1 if torch.isnan(loss) else 0
                if nan_flag == 1:
                    print("return -1, -1")
                    print("NAN Error")
                    exit(-1)
                    return -1, -1

                loss.backward()

                tr_loss += loss.item()
                # torch.save(self.model.state_dict(), './model_state_{}'.format(step))
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                # if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                #     print("Dev inside train..")
                #     self.evaluate("dev")

                # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                #     self.save_model()

                # if 0 < self.args.max_steps < global_step:
                    # epoch_iterator.close()
                    # break
            
            if nan_flag == 1:
                print("nan Error!")
                print("return -1, -1")
                exit(-1)
                return -1, -1

            early_stop += 1
            results_dev = self.evaluate("dev")
            results_test = self.evaluate("test")

            _metric = 'intent_acc' if self.args.sub_task == 'intent' else 'slot_f1'

            print(_metric, results_dev[_metric])

            if results_dev[_metric] >= best_dev_acc:
                best_dev_acc = results_dev[_metric]
                best_test_acc = results_test[_metric]
                # self.save_model()
                print("Saving model online\n")
                self.save_model_online()
                early_stop = 0
                # self.save_model_case_study()
            
            self.logger.acc_info("dev_acc", self.cur_seed, results_dev[_metric])
            self.logger.acc_info("test_acc", self.cur_seed, results_test[_metric])
            self.logger.append_loss_origin(self.cur_seed, loss_origin_log.item() / (step+1))
            self.logger.append_loss_attr(self.cur_seed, loss_attr_log / (step+1))
            
            # if 0 < self.args.max_steps < global_step:
                # train_iterator.close()
                # break
            
            if (epoch+1) >= self.args.warmup_epoch:
                # Stop warming up
                warming_up_flag = False

            if early_stop > self.args.early_stop:
                if self.args.warmup_epoch == 0:
                    break
                else:
                    if warming_up_flag:
                        # We try to train the model after the origin task training is finished
                        warming_up_flag = False # Stop warming up, and train model with extra loss terms
                        early_stop = 0
                    else:
                        # The model is trained with task and rationale
                        break

        self.logger.result_info('best_dev_acc', best_dev_acc)
        self.logger.result_info('best_test_acc', best_test_acc)
        # self.logger.save_plot_loss(self.args.log_path)

        return global_step, tr_loss / global_step


    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
            # self.load_model()
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("\n***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")


    def save_model_online(self):
        # self.best_model_online = copy.deepcopy(self.model)
        self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
        print("Online model state_dict saved")

    def load_model_online(self):
        # if self.best_model_online is not None:
        #     self.model = copy.deepcopy(self.best_model_online)
        # else:
        #     # raise Exception("best_model_online is None")
        #     print("Warning: no model loaded, cz")
        #     pass

        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            print("Loaded model online!")
        else:
            print("Warning, no model loaded")
            pass

    def save_model_case_study(self):
        if not os.path.exists(self.args.case_study_dir):
            os.makedirs(self.args.case_study_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.case_study_dir)

    def load_model_case_study(self):
        try:
            self.model = self.model_class.from_pretrained(self.args.case_study_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

    @torch.no_grad()
    def case_study(self):

        test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.train_batch_size)
        test_aug_dataloader = DataLoader(self.test_dataset_aug, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(test_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(test_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)


        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        best_dev_acc, best_test_acc = -1, -1
        early_stop = 0
        nan_flag = 0
        # warming_up_flag = True if self.args.warmup_epoch > 0 else False
        margin_loss, saliency_loss, feng_loss= torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        lines_case_study = []

        for step, (batch, batch_aug) in enumerate(zip(test_dataloader, test_aug_dataloader)):
            # print(batch_aug[0])

            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
            batch_aug = tuple(t.to(self.device) for t in batch_aug)

            inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'intent_label_ids': batch[3],
                        'slot_labels_ids': batch[4]}
            if self.args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2]

            annotation_ids = batch[6]
            input_sentene_list = []
            for i in range(inputs['input_ids'].size(0)):
                sep_ind = inputs['input_ids'][i].tolist().index(102)
                input_sentene_list.append(self.model.tokenizer_debug.decode(inputs['input_ids'][i][1:sep_ind]))
                # print(self.model.tokenizer_debug.decode(inputs['input_ids'][i][1:sep_ind]))


            outputs = self.model.forward_case_study(**inputs)
            loss = outputs[0]

            pred = outputs[-2].tolist()
            gold = outputs[-1].tolist()

            if self.args.weight != 0:
                aug_flags = batch[5]
                m_1, m_2, margin_loss = self.model.forward_aug(batch_aug, aug_flags)
            
            m_index = 0

            for i in range(len(input_sentene_list)):
                if aug_flags[i] == 1:
                    print(pred[i], gold[i], input_sentene_list[i])
                    print(m_1[m_index].detach().item(), m_2[m_index].detach().item())
                    lines_case_study.append(str(pred[i]) + '  ' + str(gold[i]) + str(input_sentene_list[i]) + '\n')
                    lines_case_study.append(str(round(m_1[m_index].detach().item(),3)) + \
                                            '   ' + \
                                            str(round(m_2[m_index].detach().item(),3)) + '\n')
                    m_index += 1

                    
        with open('./case_study_our.txt', 'w') as f:
            f.writelines(lines_case_study)