import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from .torchcrf_weigted import CRF, contain_nan
from .module import IntentClassifier, SlotClassifier

from transformers import BertTokenizer

import pdb


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.tokenizer_debug = BertTokenizer.from_pretrained('./bert-base-uncased')

        if self.args.grad:
            self.bert.embeddings.word_embeddings.weight.requires_grad_()
        
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(self.device)

    @staticmethod
    def load_tokenizer():
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained('./bert-base-uncased')


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        slot_logits_no_reduction = None # Used when running methods from Saliency Learning...

        if contain_nan(slot_logits):
            print("Input causes NAN Error: in forward()")
            for i in range(input_ids.size(0)):
                print(self.tokenizer_debug.convert_ids_to_tokens(input_ids[i]))
                # print(attention_mask[i])
                # print(token_type_ids[i])
            
        # if torch.isnan(pooled_output).any():
        #     print("pooled_output nan\n",pooled_output)
        #     print("outputs:", outputs)
        #     torch.save(self.bert.state_dict(), "./nan_model_state")
        #     print("NAN model saved!!")
        #     exit(-1)
        
        # if torch.isnan(sequence_output).any():
        #     print("sequence_output nan\n",sequence_output)
        #     exit(-1)
        

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += (1-self.args.slot_loss_coef) * intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                if torch.isnan(slot_logits).any():
                    print("slot_logits forward nan Error\n")
                    print(slot_logits)
                    exit(-1)
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='none')
                slot_loss = -1 * slot_loss  # negative log-likelihood
                
                if self.args.grad:
                    slot_logits_no_reduction = (-1*slot_loss).exp() # Used when running methods from Saliency Learning...

            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss.mean()

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs + (slot_logits_no_reduction, )

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits


    def forward_aug(self, batch_aug:list, aug_flags):
        # print("Forward aug")
        # print(aug_flags)
        # print(batch_aug[0].shape)
        # print(batch_aug[1].shape)
        # print(batch_aug[2].shape)
        # print(batch_aug[3].shape)
        # print(batch_aug[4].shape)
        
        if sum(aug_flags)==0:
            print("Batch without any annotated sample!")
            # exit(-1)
            return 0

        indices_with_annotation = self.find_1s_index(aug_flags).to(self.device)
        num_sample_with_rationale = len(indices_with_annotation)

        num_aug_each_sample = 1 if self.args.replace == 'mask' else self.args.replace_repeat
        bz = num_sample_with_rationale*num_aug_each_sample*2

        # print("input_ids shape:", batch_aug[0].shape)

        input_ids = torch.index_select(batch_aug[0], 0, indices_with_annotation).view(bz, -1)
        attention_mask = torch.index_select(batch_aug[1], 0, indices_with_annotation).view(bz, -1)
        token_type_ids = torch.index_select(batch_aug[2], 0, indices_with_annotation).view(bz, -1)
        intent_label_ids = torch.index_select(batch_aug[3], 0, indices_with_annotation).view(bz, -1)
        slot_label_ids = torch.index_select(batch_aug[4], 0, indices_with_annotation).view(bz, -1)
        weight = torch.index_select(batch_aug[5], 0, indices_with_annotation)

        # print("input_ids shape:", input_ids.shape)
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        # print(intent_label_ids.shape)
        # print(slot_label_ids.shape)
        # print(weight.shape)
        # exit()

        # print(input_ids.shape)
        # temp_tokenizer = self.load_tokenizer()
        # for i in range(input_ids.size(0)):
        #     words = temp_tokenizer.decode(input_ids[i])
        #     print(words)
        # exit()

        # tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        # for i in range(input_ids.size(0)):
            # print(tokenizer.decode(input_ids[i]))
            # print(input_ids[i])
            # print(attention_mask[i])
            # print(token_type_ids[i])
        # print()
        # exit()

        # print("Aug data forward...")

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if self.args.sub_task == 'intent':
            intent_logits = self.intent_classifier(pooled_output)
            intent_likelihood = torch.softmax(intent_logits, dim=1)

            intent_likelihood_on_gold = torch.gather(intent_likelihood, 1, intent_label_ids)
            intent_likelihood_on_gold = intent_likelihood_on_gold.view(num_sample_with_rationale, num_aug_each_sample*2)
            weighted_likelihood_on_gold = intent_likelihood_on_gold * weight

            likelihood_rationale_replaced = weighted_likelihood_on_gold[:, :num_aug_each_sample]
            likelihood_non_rationale_replaced = weighted_likelihood_on_gold[:, num_aug_each_sample:]

            m_1 = self.log_odds(likelihood_rationale_replaced.sum(dim=1, keepdim=True), lower_bound=self.args.lower_bound) # Refer to m in "...input marginalization"
            m_2 = self.log_odds(likelihood_non_rationale_replaced.sum(dim=1, keepdim=True), lower_bound=self.args.lower_bound)

            # print()
            # print("likelihood_rationale_replaced", m_1.size())
            # print(m_1)
            # print("likelihood_non_rationale_replaced", m_2.size())
            # print(m_2)
            # print()
            margin_loss = self.margin_loss(m_2, m_1, self.args.margin)

            # print(margin_loss.shape)
            if 'case' in self.args.case_study_dir:
                return m_1, m_2, margin_loss
            
            return margin_loss

        elif self.args.sub_task == 'slot':
            slot_logits = self.slot_classifier(sequence_output) # snips 74 label dimension
            # print("===========")
            # print(sequence_output.shape)
            # print(slot_logits.shape)
            # print(slot_logits)
            
            if slot_label_ids is not None:
                if self.args.use_crf:

                    if torch.isnan(slot_logits).any():
                        print("Slot logits nan Error:", slot_logits)
                        exit(-1)
                    # print("slot_logits \n", slot_logits)

                    log_likelihood = self.crf(slot_logits, slot_label_ids, mask=attention_mask.byte(), reduction='none')
                    log_likelihood = log_likelihood.view(-1, num_aug_each_sample*2)

                    # print("CRF log-likelihood:")
                    # print(log_likelihood)

                    weighted_likelihood = log_likelihood.exp() * weight

                    # print(weight)
                    # print(log_likelihood)
                    # print(log_likelihood.exp())
                    

                    likelihood_rationale_replaced = weighted_likelihood[:, :num_aug_each_sample]
                    likelihood_non_rationale_replaced = weighted_likelihood[:, num_aug_each_sample:]

                    # print(likelihood_rationale_replaced.sum(dim=1, keepdim=True))
                    # print(likelihood_non_rationale_replaced.sum(dim=1, keepdim=True))

                    m_1 = self.log_odds(likelihood_rationale_replaced.sum(dim=1, keepdim=True))
                    m_2 = self.log_odds(likelihood_non_rationale_replaced.sum(dim=1, keepdim=True))

                    # print("m_1:", m_1)
                    # print("m_2:", m_2)
                    # exit()

                    margin_loss = self.margin_loss(m_2, m_1, self.args.margin)
                    # print(margin_loss.shape)
                    # exit()
                    return margin_loss



    @staticmethod
    def find_1s_index(t):
        ret = []
        assert len(t.size())==1

        for i, _item in enumerate(t.tolist()):
            if _item == 1:
                ret.append(i)
        return torch.tensor(ret)

    @staticmethod
    def log_odds(x, lower_bound=1e-25):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ret = torch.zeros(x.size()).float().fill_(lower_bound).to(device)

        # if (x<=0).any() or (x>=1).any():
            # print(x)
            # print("Error log odds")
            # exit(-1)
            # x[x==0] += 1e-50
        
        # odds = x/(1-x)
        
        pos_valid = (x>lower_bound)
        ret[pos_valid] = (x/(1-x))[pos_valid]
        # print("log-odds:", ret.log2())
        return ret.log2()


    @staticmethod
    def margin_loss(x, y, epsilon):
        """
        Expect that X is EPSILON larger than Y
        :param x: Expected to be larger than y
        :param y: Expected to be smaller than x
        :param epsilon: margin
        :return: Marginal loss
        """
        # assert epsilon > 0
        # epsilon = torch.full(x.size(), epsilon).cuda()
        # return torch.mean(torch.max(torch.tensor(0.0).cuda(), y-x+epsilon))


        assert epsilon > 0
        mask = (~((x==0) + (y==0)).bool()).float().squeeze()
        mask = mask / mask.sum()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        epsilon = torch.full(x.size(), epsilon).to(device)
        ret = mask.dot(torch.max(torch.tensor(0.0).to(device), y-x+epsilon).squeeze())
        return ret


    def forward_saliency(self, logits, inputs, annotation_ids):
        # Idea from "Saliency Learning: Teaching the Model Where to Pay Attention"
        
        input_ids = inputs['input_ids']
        # pdb.set_trace()

        if self.args.sub_task == 'intent':
            index_select = inputs['intent_label_ids'].unsqueeze(0).T
            logits = torch.gather(logits, 1, index_select)

        # batch_size = input_ids.size(0)
        # saliency_loss = 0
        # for b in range(batch_size):

        #     assert annotation_ids.shape==input_ids.shape, print("Error, different size of annotation_ids and input_ids")
        #     embeddings_grad = torch.autograd.grad(logits[b], self.bert.embeddings.word_embeddings.weight, create_graph=True)[0]

        #     # print(input_ids.size()) # (bz, max_seq_len)
        #     pdb.set_trace()
                    
        #     embeddings_grad_used = embeddings_grad[input_ids[b]]
            
        #     # print(embeddings_grad_used.shape)
        #     # print(annotation_ids.shape)
        #     # print(embeddings_grad_used.shape)

        #     embeddings_grad_used = -embeddings_grad_used.sum(dim=1)
        #     saliency_loss_per_sample = (annotation_ids * embeddings_grad_used)

        #     if (saliency_loss_per_sample<=0).all():
        #         return torch.tensor(0.)
        #         print("Saliency loss non-positive !!!")

        #     saliency_loss += saliency_loss_per_sample[saliency_loss_per_sample>0].mean()

        # pdb.set_trace()
        
        
        embeddings_grad = torch.autograd.grad(logits.flatten(), self.bert.embeddings.word_embeddings.weight, 
                                                create_graph=True,
                                                grad_outputs=(torch.eye(torch.numel(logits)).to(self.device),),
                                                is_grads_batched=True
                                              )[0]

        input_ids_shape = list(input_ids.size())
        # input_ids_flatten = input_ids.view(-1)

        embeddings_grad_used = torch.zeros((embeddings_grad.size(0), self.args.max_seq_len, embeddings_grad.size(2))).to(self.device)
        
        for i in range(embeddings_grad_used.size(0)):
            embeddings_grad_used[i] = torch.index_select(embeddings_grad[i], 0, input_ids[i].flatten())

        embeddings_grad_used = embeddings_grad_used.view(input_ids_shape+[-1])

        embeddings_grad_used = -embeddings_grad_used.sum(dim=2)
        saliency_loss = (annotation_ids*embeddings_grad_used)
        
        # pdb.set_trace()

        if (saliency_loss<=0).all():
            print("Saliency loss = 0")
            return torch.tensor(0.)

        saliency_loss = saliency_loss[saliency_loss>0].mean()
        
        return saliency_loss


    def forward_feng(self, loss, input_ids, annotation_ids):
        def bern(p):
            """
            P is a tensor of size (bz, 1)
            each item is in (0,1)
            sample 1 with probability P
            return tensor of same shape of P, with [0, 1] entries
            """
            _sample = torch.rand_like(p)
            return (_sample<p).long()

        assert annotation_ids.shape==input_ids.shape, print("Error, different size of annotation_ids and input_ids")
        embeddings_grad = torch.autograd.grad(loss, self.bert.embeddings.word_embeddings.weight, create_graph=True)[0]
        
        input_ids_shape = list(input_ids.size())
        input_ids_flatten = input_ids.view(-1)
        
        embeddings_grad_used = embeddings_grad[input_ids_flatten]
        embeddings_grad_used = embeddings_grad_used.view(input_ids_shape+[-1]) # of size (bz, max_seq_len, hidden_dim)

        # g in paper "Exploring Distantly-Labeled Rationales in Neural Network Models"
        g = torch.norm(embeddings_grad_used, p=1, dim=2) # of size (bz, max_seq_len)
        # s in paper "Exploring Distantly-Labeled Rationales in Neural Network Models"
        s = g / g.sum(dim=1, keepdim=True) # of size (bz, max_seq_len)

        # print("s:", s)

        # Base Loss
        if self.args.feng == 'base':
            one_over_k = annotation_ids / (annotation_ids.sum(dim=1, keepdim=True)+1e-20) # +1e-20 in case of samples without rationales
            base_loss = (s[one_over_k!=0] - one_over_k[one_over_k!=0]).pow(2).sum()
            ret = base_loss

        # Order Loss
        elif self.args.feng == 'order':
            rationale_salience = s.clone()
            rationale_salience[annotation_ids==0] = 0

            non_rationale_salience = s.clone()
            non_rationale_salience[annotation_ids==1] = 0

            max_non_rationael_salience = non_rationale_salience.max(dim=1, keepdim=True).values # of size (bz, 1)
            order_loss = torch.pow(torch.min(
                                            (rationale_salience / max_non_rationael_salience) - 1, torch.zeros_like(rationale_salience)
                                            ), 2)

            order_loss = order_loss[annotation_ids==1].sum()
            ret = order_loss

        # Gate Loss
        elif self.args.feng == 'gate':
            base_loss =  ((s - 1).pow(2) * annotation_ids).sum(dim=1, keepdim=True) # of size (bz, 1)

            sum_of_rationales_salience_in_each_sample = s.clone()
            sum_of_rationales_salience_in_each_sample[annotation_ids==0] = 0
            p = sum_of_rationales_salience_in_each_sample.sum(dim=1, keepdim=True) # of size (bz, 1)

            gate_loss = bern(1-p) * base_loss
            gate_loss = gate_loss.mean()
            ret = gate_loss

        # Gate + Order Loss
        elif self.args.feng == 'gate+order':
            rationale_salience = s.clone()
            rationale_salience[annotation_ids==0] = 0

            non_rationale_salience = s.clone()
            non_rationale_salience[annotation_ids==1] = 0

            max_non_rationael_salience = non_rationale_salience.max(dim=1, keepdim=True).values # of size (bz, 1)
            order_loss = torch.pow(torch.min(
                                            (rationale_salience / max_non_rationael_salience) - 1, torch.zeros_like(rationale_salience)
                                            ), 2)

            sum_of_rationales_salience_in_each_sample = s.clone()
            sum_of_rationales_salience_in_each_sample[annotation_ids==0] = 0
            p = sum_of_rationales_salience_in_each_sample.sum(dim=1, keepdim=True) # of size (bz, 1)

            gate_order_loss = (order_loss*annotation_ids).sum(dim=1, keepdim=True) * bern(1-p)
            gate_order_loss = gate_order_loss.mean()
            ret = gate_order_loss
        
        else:
            ret = torch.tensor(0.)


        return ret

    def forward_case_study(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        slot_logits_no_reduction = None # Used when running methods from Saliency Learning...


        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
                
                pred = intent_logits.view(-1, self.num_intent_labels).argmax(dim=1)
                gold = intent_label_ids.view(-1)
                # print("pred:", pred)
                # print("gold:", gold)
                # exit()
            total_loss += (1-self.args.slot_loss_coef) * intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                if torch.isnan(slot_logits).any():
                    print("slot_logits forward nan Error\n")
                    print(slot_logits)
                    exit(-1)
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='none')
                slot_loss = -1 * slot_loss  # negative log-likelihood
                
                if self.args.grad:
                    slot_logits_no_reduction = (-1*slot_loss).exp() # Used when running methods from Saliency Learning...

            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss.mean()

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs + (slot_logits_no_reduction, ) + (pred,) + (gold,)

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
