from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
import os

import pdb

class CrossEncoderForWNLI(nn.Module):
    def __init__(self, args):
        super(CrossEncoderForWNLI, self).__init__()
        
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        if args.dataset == 'wnli':
            self.classifier = nn.Linear(768, 2)
        elif args.dataset == 'esnli':
            self.classifier = nn.Linear(768, 3)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.args.grad:
            self.bert.embeddings.word_embeddings.weight.requires_grad_()

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        
        # pdb.set_trace()

        if type(out) is tuple:
            logits = out[0][:, 0, :]
        else:
            logits = out.last_hidden_state[:, 0, :]        

        out = self.classifier(logits)
        return out
    
    def forward_aug(self, batch_aug, aug_flags):
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
            return torch.tensor(0.)

        indices_with_annotation = self.find_1s_index(aug_flags).to(self.device)
        num_sample_with_rationale = len(indices_with_annotation)

        num_aug_each_sample = 1 if self.args.replace == 'mask' else self.args.replace_repeat
        bz = num_sample_with_rationale*num_aug_each_sample*2

        # print("input_ids shape:", batch_aug[0].shape)

        input_ids = torch.index_select(batch_aug[0].to(self.device), 0, indices_with_annotation).view(bz, -1)
        attention_mask = torch.index_select(batch_aug[1].to(self.device), 0, indices_with_annotation).view(bz, -1)
        token_type_ids = torch.index_select(batch_aug[2].to(self.device), 0, indices_with_annotation).view(bz, -1)
        label = torch.index_select(batch_aug[3].to(self.device), 0, indices_with_annotation).view(bz, -1)
        weight = torch.index_select(batch_aug[4].to(self.device), 0, indices_with_annotation)

        # print("input_ids shape:", input_ids.shape)
        # exit()
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        # print(label)
        # print(slot_label_ids.shape)
        # print(weight)
        if (weight==0).any():
            print("Error, Weight == 0 ?!")
            exit(-1)
        # exit()

        # print(input_ids.shape)
        # temp_tokenizer = self.load_tokenizer()
        # for i in range(input_ids.size(0)):
        #     words = temp_tokenizer.decode(input_ids[i])
        #     print(words)
        # print(weight)
        # exit()

        # tokenizer = self.load_tokenizer()
        # for i in range(input_ids.size(0)):
        #     print(tokenizer.decode(input_ids[i]))
        #     print(input_ids[i])
        #     # print(attention_mask[i])
        #     # print(token_type_ids[i])
        # print()
        # exit()

        # print("Aug data forward...")

        intent_logits = self.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        intent_likelihood = torch.softmax(intent_logits, dim=1)

        intent_likelihood_on_gold = torch.gather(intent_likelihood, 1, label)
        intent_likelihood_on_gold = intent_likelihood_on_gold.view(num_sample_with_rationale, num_aug_each_sample*2)
        weighted_likelihood_on_gold = intent_likelihood_on_gold * weight

        likelihood_rationale_replaced = weighted_likelihood_on_gold[:, :num_aug_each_sample]
        likelihood_non_rationale_replaced = weighted_likelihood_on_gold[:, num_aug_each_sample:]

        # print(likelihood_rationale_replaced.sum(dim=1, keepdim=True))
        # print(likelihood_non_rationale_replaced.sum(dim=1, keepdim=True))

        m_1 = self.log_odds(likelihood_rationale_replaced.sum(dim=1, keepdim=True), lower_bound=self.args.lower_bound) # Refer to m in "...input marginalization"
        m_2 = self.log_odds(likelihood_non_rationale_replaced.sum(dim=1, keepdim=True), lower_bound=self.args.lower_bound)

        margin_loss = self.margin_loss(m_2, m_1, self.args.margin)

        # print(margin_loss.shape)
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
    def load_tokenizer():
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained('./bert-base-uncased')

    def log_odds(self, x, lower_bound=1e-25):
        ret = torch.zeros(x.size()).float().fill_(lower_bound).to(self.device)

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


    def margin_loss(self, x, y, epsilon):
        """
        Expect that X is EPSILON larger than Y
        :param x: Expected to be larger than y
        :param y: Expected to be smaller than x
        :param epsilon: margin
        :return: Marginal loss
        """

        assert epsilon > 0

        epsilon = torch.full(x.size(), epsilon).to(self.device)

        # print("Loss before margin: ", (y-x+epsilon))

        ret = torch.max(torch.tensor(0.0).to(self.device), y-x+epsilon).squeeze()
        return ret.mean()


    def forward_saliency(self, logits, _input, labels, annotation_ids):
        # Idea from "Saliency Learning: Teaching the Model Where to Pay Attention"
        
        input_ids = _input['input_ids']

        labels = labels.unsqueeze(0).T
        logits = torch.gather(logits, 1, labels)

        embeddings_grad = torch.autograd.grad(logits.flatten(), self.bert.embeddings.word_embeddings.weight, 
                                                create_graph=True,
                                                grad_outputs=(torch.eye(torch.numel(logits)).to(self.device),),
                                                is_grads_batched=True
                                              )[0]

        input_ids_shape = list(input_ids.size())

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


        # assert annotation_ids.shape==input_ids.shape, print("Error, different size of annotation_ids and input_ids")
        # embeddings_grad = torch.autograd.grad(loss, self.bert.embeddings.word_embeddings.weight, create_graph=True)[0]

        # # print(input_ids.size()) # (bz, max_seq_len)

        # input_ids_shape = list(input_ids.size())
        # input_ids_flatten = input_ids.view(-1)
                
        # embeddings_grad_used = embeddings_grad[input_ids_flatten]

        # # self._print(embeddings_grad_used)

        # embeddings_grad_used = embeddings_grad_used.view(input_ids_shape+[-1])
        
        # # print(embeddings_grad_used.shape)
        # # print(annotation_ids.shape)
        # # print(embeddings_grad_used.shape)

        # embeddings_grad_used = -embeddings_grad_used.sum(dim=2)
        # saliency_loss = (annotation_ids * embeddings_grad_used)
        # saliency_loss = saliency_loss[saliency_loss>0].mean()

        # return saliency_loss

        # # def _print(tensor):
        # #     # print(tensor.shape)
        # #     for i in range(tensor.size(0)):
        # #         for j in range(tensor.size(1)):

        # #             if tensor[i][j] != 0:
        # #                 print(tensor[i][j])

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


    def save_model(self, cur_seed):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        bert_state_dict_path = "{0}{1}{2}bert.state_dict".format(self.args.model_dir, cur_seed, self.args.dataset)
        classifier_state_dict_path = "{0}{1}{2}classifier.state_dict".format(self.args.model_dir, cur_seed, self.args.dataset)

        torch.save(self.bert.state_dict(), bert_state_dict_path)
        torch.save(self.classifier.state_dict(), classifier_state_dict_path)
        print("Model saved")
        
    def load_model(self, cur_seed):
        bert_state_dict_path = "{0}{1}{2}bert.state_dict".format(self.args.model_dir, cur_seed, self.args.dataset)
        classifier_state_dict_path = "{0}{1}{2}classifier.state_dict".format(self.args.model_dir, cur_seed, self.args.dataset)

        print(bert_state_dict_path)
        self.bert.load_state_dict(torch.load(bert_state_dict_path))
        self.classifier.load_state_dict(torch.load(classifier_state_dict_path))

        print("Model loaded")