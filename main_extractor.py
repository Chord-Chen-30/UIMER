import argparse
import sys

from trainer_extrctor import TrainerExtractor
from utils import init_logger, load_tokenizer, print_cmd, read_prediction_text, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples
from Logger import MyLogger
import torch

import pdb

def main(args):
    init_logger()
    mylogger = MyLogger(args.log_path, args, cmd=print_cmd(sys.argv))
    for seed in args.seed.split(":"):
        print("Seed:", seed)
        set_seed(int(seed))
        tokenizer = load_tokenizer(args)

        train_dataset, train_dataset_aug = load_and_cache_examples(args, tokenizer, mode='train')
        
        dev_dataset, _ = load_and_cache_examples(args, tokenizer, mode="dev")
        test_dataset, _ = load_and_cache_examples(args, tokenizer, mode="test")

        trainer = TrainerExtractor(args, train_dataset, dev_dataset, test_dataset, train_dataset_aug, seed, mylogger)

        # Step 1. Train the model by origin task
        if not args.load_model:
            trainer.train_model()
            print("Model Training (1) Finished!")
            break

        else:
            trainer.load_model()
            results_dev = trainer.evaluate('dev')
            results_test = trainer.evaluate('test')
            _metric = 'intent_acc' if trainer.args.sub_task == 'intent' else 'slot_f1'
            trainer.logger.result_info('best_dev_acc', results_dev[_metric])
            trainer.logger.result_info('best_test_acc', results_test[_metric])

        # continue
        if not args.load_model:
            best_dev_over_rounds = -1
            best_test_over_rounds = -1
        else:
            best_dev_over_rounds = results_dev[_metric]
            best_test_over_rounds = results_test[_metric]

        round_early_stop = 0
        best_dev_saved = best_dev_over_rounds
        best_test_saved = best_test_over_rounds
        for _round in range(1, args.total_round+1):
            print("Round: ", _round)
            round_early_stop += 1
            print("Early stop:", round_early_stop)

            if _round >= 2:
                trainer.load_model_online()
            
            # Step 2. Fix model param, train extractor
            trainer.train_extractor(_round)
            print("Extractor Training Finished!")

            # # Step 3. Fix extractor, add in another loss
            best_dev_over_rounds, best_test_over_rounds = trainer.train_model_with_rationale(_round, best_dev_over_rounds, best_test_over_rounds)
            print("Model Training (2) Finished!")

            if best_dev_over_rounds >= best_dev_saved:
                round_early_stop = 0
                best_dev_saved = best_dev_over_rounds
                best_test_saved = best_test_over_rounds
            
            if round_early_stop == (args.early_stop*2):
                break

        mylogger.result_info('best_dev_acc_extractor_extra_round', best_dev_saved)
        mylogger.result_info('best_test_acc_extractor_extra_round', best_test_saved)
        
    trainer.logger.save_plot_loss(args.log_path)
    trainer.logger.cal_std_mean()
    trainer.logger.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default='./ckpt/', type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=str, default="0:30", help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=70.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

    # Mask
    parser.add_argument("--sub_task", default='intent', type=str, help="intent/slot")
    parser.add_argument("--replace", default='random', type=str, help="Replace strategy [none/random/frequency/bert]")
    parser.add_argument("--replace_repeat", default=4, type=int, help="Times to apply replacement")
    parser.add_argument("--lower_bound", default=1e-25, type=float, help="Lower bound when calculating log-odds")
    parser.add_argument("--max_rationale_percentage", default=0.3, type=float, help="Maximum percentage of rationales we apply replacement")
    parser.add_argument("--warmup_epoch", default=0, type=int, help="Add extra loss to objective after [WARMUP_EPOCH] epochs")

    parser.add_argument("--weight", default=0.0001, type=float, help="weight of attribution loss")
    parser.add_argument("--margin", default=0.1, type=float, help="Margin loss - margin")
    parser.add_argument("--shot", default=10, type=int, help="few shot setting: SHOT for each class")

    parser.add_argument("--grad", default=0, type=int, help="use gradient-based method")
    parser.add_argument("--grad_weight", default=1., type=float, help="weight of saliency loss")

    parser.add_argument("--feng", default='none', type=str, help="[none/base/order/gate/gate+order]")
    parser.add_argument("--feng_weight", default=1., type=float, help="weight feng loss")

    parser.add_argument("--log_path", default="./log-temp/", type=str, help="Log dir")
    parser.add_argument("--verbose", default=0, type=int, help="Detailed print()")
    parser.add_argument("--early_stop", default=15, type=int, help="Early stop")


    # How Do Decisions Emerge..
    parser.add_argument("--num_train_extractor_epochs", default=10, type=int, help="num epochs to train extractor")
    parser.add_argument("--lr_extractor", default=1e-3, type=float, help="learning rate when training extractor")
    parser.add_argument("--lr_decay", default=1., type=float, help="Lr decay when running extra round")
    parser.add_argument("--gate", default='input', type=str, help="gate on [input/hidden]")
    parser.add_argument("--layer_pred", type=int, default=-1)
    parser.add_argument("--stop_train", type=int, default=1)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--weight_extractor", type=float, default=0.1)
    parser.add_argument("--loss_func_rationale", type=str, default='L2norm', help='loss function when running forward_rationale_supervision, [L2norm/BCE/margin]')
    parser.add_argument("--load_model", type=int, default=1, help='Load model of training phase 1 (before training extractor)')
    parser.add_argument("--total_round", type=int, default=1, help='Total rounds of training extractor and model')
    parser.add_argument("--re_init_extractor", type=int, default=0, help='Whether re-init extractor each round')

    parser.add_argument("--gpu_name", type=str, default='-', help="Keep a record of GPU info")
    

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]

    args.slot_loss_coef = 1.0 if args.sub_task == 'slot' else 0.0

    try:
        args.gpu_name = torch.cuda.get_device_name()
    except:
        print("Unable to obtain gpu name")
        pass

    main(args)

    print("Success")


"""
python main.py --seed 44 --learning_rate 5e-05 --train_batch_size 24 --task snips --model_type bert --model_dir snips_model --do_train   --do_eval   --use_crf   --logging_step 1 --log_path ./log-temp/ --sub_task slot --shot 5 --replace bert --replace_repeat 4 --weight 0.1 --margin 0.1 --verbose 0 --early_stop 10


"""