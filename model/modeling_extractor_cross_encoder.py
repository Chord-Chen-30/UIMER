import curses
from model import CrossEncoderForWNLI
from .module import *
from .getter_setter import bert_getter, bert_setter
from .distributions import *

import numpy as np
import torch
import os
import copy

class CrossEncoderExtractorForWNLI(CrossEncoderForWNLI):
    def __init__(self, args):
        super(CrossEncoderExtractorForWNLI, self).__init__(args)

        self.alpha = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones(()))
                for _ in range(self.bert.config.num_hidden_layers + 2)
            ]
        )

        gate = DiffMaskGateInput if self.args.gate == "input" else DiffMaskGateHidden
        
        self.gate = gate(
            hidden_size=self.bert.config.hidden_size,
            hidden_attention=self.bert.config.hidden_size // 4,
            num_hidden_layers=self.bert.config.num_hidden_layers + 2,
            max_position_embeddings=1,
            gate_bias=True,
            placeholder=True,
            init_vector=self.bert.embeddings.word_embeddings.weight[
                self.tokenizer.mask_token_id
            ]
            if self.args.layer_pred == 0 or self.args.gate == "input"
            else None,
        )

        self.register_buffer(
            "running_acc", torch.ones((self.bert.config.num_hidden_layers + 2,))
        )
        self.register_buffer(
            "running_l0", torch.ones((self.bert.config.num_hidden_layers + 2,))
        )
        self.register_buffer(
            "running_steps", torch.zeros((self.bert.config.num_hidden_layers + 2,))
        )

        self.best_model_state_dict = None
    
    def fix_model_tune_extractor(self):
        for p in self.bert.parameters():
            p.requires_grad_(False)
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        
        for p in self.gate.parameters():
            p.requires_grad_(True)
        for p in self.alpha:
            p.requires_grad_(True)

    def fix_extractor_tune_model(self):
        for p in self.bert.parameters():
            p.requires_grad_(True)
        for p in self.classifier.parameters():
            p.requires_grad_(True)
        
        for p in self.gate.parameters():
            p.requires_grad_(False)
        for p in self.alpha:
            p.requires_grad_(False)

    def _forward_explainer(self, batch, layer_pred=None, attribution=False):
        inputs_dict = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        self.bert.eval()

        # (logits_orig,), hidden_states = bert_getter(self, inputs_dict)
        logits_orig, hidden_states = bert_getter(self, inputs_dict)

        # logits_orig.size() (64, 5)
        # hidden_states (tuple, length=14), hidden_states[0].size()= (64, 84, 768)
        # 84 is max_seq_len

        if layer_pred is None:
            if self.args.stop_train:
                stop_train = (
                    lambda i: self.running_acc[i] > 0.75
                    and self.running_l0[i] < 0.1
                    and self.running_steps[i] > 100
                )
                p = np.array(
                    [0.1 if stop_train(i) else 1 for i in range(len(hidden_states))]
                )
                layer_pred = torch.tensor(
                    np.random.choice(range(len(hidden_states)), (), p=p / p.sum()),
                    device=batch[0].device,
                )
            else:
                layer_pred = torch.randint(len(hidden_states), ()).item()

        # print("forward explainer 1")
        # pdb.set_trace()

        if "hidden" in self.args.gate:
            layer_drop = layer_pred
        else:
            layer_drop = 0

        (
            new_hidden_state,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
        ) = self.gate(
            hidden_states=hidden_states,
            mask=batch[1],
            layer_pred=None if attribution else layer_pred,
        )

        if attribution:
            return expected_L0_full
        else:
            new_hidden_states = (
                [None] * layer_drop
                + [new_hidden_state]
                + [None] * (len(hidden_states) - layer_drop - 1)
            )

            # (logits,), _ = bert_setter(
            #     self, inputs_dict, new_hidden_states, self.forward
            # )

            logits, hidden_states = bert_setter(
                self, inputs_dict, new_hidden_states, self.forward
            )

        # print("forward explainer 2")
        # pdb.set_trace()

        return (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        )

    def forward_extractor(self, batch):
        input_ids = batch[0]
        attention_mask = batch[1]
        token_type_ids = batch[2]

        # Code From diffmask
        (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        ) = self._forward_explainer(batch)

        loss_c = (
            torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=logits_orig),
                torch.distributions.Categorical(logits=logits),
            )
            - self.args.eps
        )

        loss_g = (expected_L0 * attention_mask).sum(-1) / attention_mask.sum(-1)

        loss = self.alpha[layer_pred] * loss_c + loss_g

        acc, _, _, f1 = accuracy_precision_recall_f1(
            logits.argmax(-1), logits_orig.argmax(-1), average=True
        )

        l0 = (expected_L0.exp() * attention_mask).sum(-1) / attention_mask.sum(-1)

        outputs_dict = {
            "loss_c": loss_c.mean(-1),
            "loss_g": loss_g.mean(-1),
            "alpha": self.alpha[layer_pred].mean(-1),
            "acc": acc,
            "f1": f1,
            "l0": l0.mean(-1),
            "layer_pred": layer_pred,
            "r_acc": self.running_acc[layer_pred],
            "r_l0": self.running_l0[layer_pred],
            "r_steps": self.running_steps[layer_pred],
        }

        outputs_dict = {
            "loss": loss.mean(-1),
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        if self.training:
            self.running_acc[layer_pred] = (
                self.running_acc[layer_pred] * 0.9 + acc * 0.1
            )
            self.running_l0[layer_pred] = (
                self.running_l0[layer_pred] * 0.9 + l0.mean(-1) * 0.1
            )
            self.running_steps[layer_pred] += 1

        # print("Forward Extractor return")
        # pdb.set_trace()

        return outputs_dict['loss']

    def forward_rationale_supervision(self, batch, ce_loss):
        # Calculate first loss: task loss
        #           second loss: rationale alignment loss
        labels = batch[3]

        (
            logits,
            logits_orig,
            gates,
            expected_L0,
            gates_full,
            expected_L0_full,
            layer_drop,
            layer_pred,
        ) = self._forward_explainer(batch)

        # pdb.set_trace()

        inputs = {'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2]}

        logits = self.forward(**inputs)
        loss_task = ce_loss(logits, labels)

        if self.args.loss_func_rationale == 'L2norm':
            loss_alignment = torch.norm((1-gates)-batch[-1], p=2)

        elif self.args.loss_func_rationale == 'BCE':
            bce_loss = torch.nn.BCELoss(reduction='none') # none because of padding element
            sigmoid = torch.nn.Sigmoid()
            loss_alignment = bce_loss(sigmoid(1-gates), batch[-1].float())
            loss_alignment = (loss_alignment*batch[1]).mean()
            
        elif self.args.loss_func_rationale == 'margin':
            annotations = batch[-1]
            saliency =  gates

            rationale_saliency = saliency.clone()
            rationale_saliency[annotations==0] = 0

            non_rationale_salicy = saliency.clone()
            non_rationale_salicy[annotations==1] = 0

            max_non_rationale_saliency = non_rationale_salicy.max(dim=1, keepdim=True).values
            loss_alignment = torch.pow(torch.min(
                                                (rationale_saliency / max_non_rationale_saliency)-1, 
                                                torch.zeros_like(rationale_saliency)), 
                                    2)
            loss_alignment = loss_alignment[annotations==1].sum()

        return logits, loss_task + self.args.weight_extractor*loss_alignment

    
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

        # Here due to the issue of different versions of transformer packages. Setting strict=True might lead to some "Unexpected key Error"  
        self.bert.load_state_dict(torch.load(bert_state_dict_path), strict=False)
        self.classifier.load_state_dict(torch.load(classifier_state_dict_path), strict=False)

        print("Model loaded")

    def save_model_online(self):
        self.best_model_state_dict = copy.deepcopy(self.state_dict())
        print("Online model state dict saved")

    def load_model_online(self):
        if self.best_model_state_dict is not None:
            self.load_state_dict(self.best_model_state_dict)
            print("Loaded model online!")
        else:
            print("Warning, no model loaded")
            pass