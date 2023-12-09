import os
import logging
from re import L
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from model.modeling_extractor_bert import ExtractorBERT

from transformers import BertTokenizer
from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels, print_cmd
import sys
import copy
import time

import pdb

logger = logging.getLogger(__name__)


class TrainerExtractor(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, train_dataset_aug=None, cur_seed=None, logger=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.train_dataset_aug = train_dataset_aug

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES['extractorbert']
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)

        self.model = ExtractorBERT.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)

        # print(self.model.fix_model_tune_extractor)
        # exit()

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

        self.cur_seed = cur_seed
        self.logger = logger

        self.best_model_online = None
        self.best_model_state_dict = None


    def train_model(self):
        self.model.fix_extractor_tune_model()

        # train_sampler = RandomSampler(self.train_dataset)
        # train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        self.model.zero_grad()
        best_dev_acc = -1
        early_stop = 0

        train_iterator = trange(int(self.args.num_train_epochs), desc='Epoch')
        for epoch in train_iterator:
            self.model.train()
            origin_loss = 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}

                loss = self.model.forward_task(**inputs)
                origin_loss += loss.item()

                print("Orgin Loss:", loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                # print("batch end")
            early_stop += 1
            results_dev = self.evaluate('dev')
            results_test = self.evaluate('test')

            _metric = 'intent_acc' if self.args.sub_task == 'intent' else 'slot_f1'

            if results_dev[_metric] >= best_dev_acc:
                best_dev_acc = results_dev[_metric]
                best_test_acc = results_test[_metric]
                self.save_model()
                early_stop = 0
            
            self.logger.acc_info("dev_acc", self.cur_seed, results_dev[_metric])
            self.logger.acc_info("test_acc", self.cur_seed, results_test[_metric])
            self.logger.loss_info(self.cur_seed, "loss_origin", origin_loss)

            print("Epoch: {0}  Dev Acc: {1}".format(epoch, results_dev[_metric]))
            print("Epoch: {0}  Test Acc: {1}".format(epoch, results_test[_metric]))

            if early_stop > self.args.early_stop:
                break
            
        self.logger.result_info('best_dev_acc', best_dev_acc)
        self.logger.result_info('best_test_acc', best_test_acc)


    def train_extractor(self, _round):
        self.model.fix_model_tune_extractor()

        # train_sampler = RandomSampler(self.train_dataset)
        # train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr_extractor*(self.args.lr_decay**_round), eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        self.model.zero_grad()
        best_loss = float('inf')
        early_stop = 0

        train_iterator = trange(int(self.args.num_train_extractor_epochs), desc="Epoch")
        for epoch in train_iterator:
            dataloader_iterator = tqdm(train_dataloader)
            total_loss = 0
            self.model.train()
            for step, batch in enumerate(dataloader_iterator):
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                loss = self.model.forward_extractor(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                total_loss += loss.item()
            
            early_stop += 1

            # print("all batch trained extractor")
            # pdb.set_trace()

            if total_loss < best_loss:
                best_loss = total_loss
                early_stop = 0
            
            if early_stop > self.args.early_stop:
                break

            print("Epoch: {0}; Extractor Loss: {1}".format(epoch, total_loss))
        
            self.logger.loss_info(self.cur_seed, 'loss_extractor', total_loss)


    def train_model_with_rationale(self, _round, best_dev_over_rounds=-1, best_test_over_rounds=-1):
        self.model.fix_extractor_tune_model()

        # train_sampler = RandomSampler(self.train_dataset)
        # train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate*(self.args.lr_decay**_round), eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        self.model.zero_grad()
        best_dev_acc = best_dev_over_rounds
        best_test_acc = best_test_over_rounds
        early_stop = 0

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for epoch in train_iterator:
            dataloader_iterator = tqdm(train_dataloader)
            self.model.train()
            attr_loss, origin_loss = 0, 0
            for step, batch in enumerate(dataloader_iterator):
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                loss_task, loss_alignment = self.model.forward_rationale_supervision(batch)
                loss = loss_task + self.args.weight_extractor*loss_alignment
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                attr_loss += loss_alignment.item()
                origin_loss += loss_task.item()
                
            results_dev = self.evaluate('dev')
            results_test = self.evaluate('test')

            _metric = 'intent_acc' if self.args.sub_task == 'intent' else 'slot_f1'

            early_stop += 1
            if results_dev[_metric] >= best_dev_acc:
                best_dev_acc = results_dev[_metric]
                best_test_acc = results_test[_metric]
                self.save_model_online()
                # self.save_model_case_study()
                early_stop = 0
            
            self.logger.acc_info("dev_acc", self.cur_seed, results_dev[_metric])
            self.logger.acc_info("test_acc", self.cur_seed, results_test[_metric])
            self.logger.loss_info(self.cur_seed, "loss_attr", attr_loss)
            self.logger.loss_info(self.cur_seed, "loss_origin", origin_loss)
            
            if early_stop > self.args.early_stop:
                break
        
        if _round == 1:
            self.logger.result_info('best_dev_acc_extractor', best_dev_acc)
            self.logger.result_info('best_test_acc_extractor', best_test_acc)
        # else:
        #     self.logger.result_info('best_dev_acc_extractor_extra_round', best_dev_acc)
        #     self.logger.result_info('best_test_acc_extractor_extra_round', best_test_acc)

        print("Best Dev Acc Extractor: ", best_dev_acc)
        print("Best Test Acc Extractor: ", best_test_acc)
        return best_dev_acc, best_test_acc


    @torch.no_grad()
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
        logger.info("***** Running evaluation on %s dataset *****", mode)
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
            # time.sleep(1)
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

        print()
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        model_path = '{0}{1}{2}shot{3}{4}{5}.state_dict'.format(self.args.model_dir, self.args.sub_task, self.args.shot, self.args.gate, self.cur_seed, self.args.learning_rate)
        torch.save(self.model.state_dict(), model_path)
        logger.info("\nSaving model checkpoint to %s", model_path)

    def load_model(self):
        # Check whether model exists
        model_path = '{0}{1}{2}shot{3}{4}.state_dict'.format(self.args.model_dir, self.args.sub_task, self.args.shot, self.args.gate, self.cur_seed)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path), strict=False)
            print("Train phase 1 model loaded!")

        else:
            print("Error, No model {} Found".format(model_path))
            exit(-1)

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
    def case_study(self, _round, best_dev_over_rounds=-1, best_test_over_rounds=-1):
        # self.model.fix_extractor_tune_model()

        # train_sampler = RandomSampler(self.train_dataset)
        # train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.train_batch_size)
        t_total = len(test_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #      'weight_decay': self.args.weight_decay},
        #     {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate*(self.args.lr_decay**_round), eps=self.args.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        tokenizer_debug = BertTokenizer.from_pretrained('./bert-base-uncased')

        self.model.zero_grad()
        best_dev_acc = best_dev_over_rounds
        best_test_acc = best_test_over_rounds
        early_stop = 0

        test_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        dataloader_iterator = tqdm(test_dataloader)
        self.model.train()
        attr_loss, origin_loss = 0, 0
        lines = []

        for step, batch in enumerate(dataloader_iterator):
            batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

            loss_task, loss_alignment, gates, pred, gold = self.model.forward_rationale_supervision(batch)
            inputs_ids = batch[0]

            for i in range(inputs_ids.size(0)):
                sep_ind = inputs_ids[i].tolist().index(102)
                # print(tokenizer_debug.convert_ids_to_tokens(inputs_ids[i][1:sep_ind]))
                # print(gates[i][1:sep_ind])

                sentence = ' '.join(tokenizer_debug.convert_ids_to_tokens(inputs_ids[i][0:sep_ind])) + '\n'
                gate_line = ' '.join([str(g) for g in gates[i][0:sep_ind].tolist()]) + '\n'
                gold_gate_line = ' '.join([str(gg.item()) for gg in batch[-1][i][0:sep_ind]]) + '\n'
                pred_gold_line = '{0}  {1}'.format(pred[i].item(), gold[i].item()) + '\n'
                
                lines.append(sentence)
                lines.append(gate_line)
                lines.append(gold_gate_line)
                lines.append(pred_gold_line)
                lines.append('\n')
                

            attr_loss += loss_alignment.item()
            origin_loss += loss_task.item()
        
        with open('./case_study_ex_base_3shot_0122.txt', 'w') as f:
            f.writelines(lines)
        exit()

        results_dev = self.evaluate('dev')
        results_test = self.evaluate('test')

        _metric = 'intent_acc' if self.args.sub_task == 'intent' else 'slot_f1'

            
        print("Best Dev Acc Extractor: ", best_dev_acc)
        print("Best Test Acc Extractor: ", best_test_acc)
        return best_dev_acc, best_test_acc
