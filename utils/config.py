import argparse
import yaml
from easydict import EasyDict
import torch


class Config:
    def __init__(self, dictionary):
        self.update(dictionary)
    
    def update(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=None, type=str, help="The config file"
    )
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
    )
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument("--text_db", type=str, help='the file path of text databse.')
    parser.add_argument(
        "--num_workers", default=0, help="the number of workers to load data"
    )
    parser.add_argument(
        "--pin_memory", default=False, help="pin memory to load data"
    )
    parser.add_argument(
        "--num_aug", default=0, help="number of augment images"
    )
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=5e-05
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--epochs", help="number of epochs", default=20, type=int
    )
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=64, type=int
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=1024, type=int
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )
    parser.add_argument("--save_path", help="save path", type=str)
    parser.add_argument(
        "--save_every_n",
        default=1,
        type=int,
        help="saves the model every n epochs; "
        "this is useful for validation/grid search",
    )
    parser.add_argument(
        "--save_model",
        help="indicate if you want to save the model state dict()",
        action="store_true",
    )
    parser.add_argument("--seed", help="seed value", default=0, type=int)

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,
        type=int
    )
    parser.add_argument('--prompt_template', type=str, default="a photo of X X")

    #  for evaluate.py
    parser.add_argument(
        "--open_world",
        help="evaluate on open world setup",
        action="store_true",
    )
    parser.add_argument(
        "--bias",
        help="eval bias",
        type=float,
        default=1e3,
    )
    parser.add_argument(
        "--topk",
        help="eval topk",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        help="batch size of the text encoder",
        default=16,
        type=int,
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help="optional threshold"
    )
    parser.add_argument(
        '--threshold_trials',
        type=int,
        default=50,
        help="how many threshold values to try"
    )

    args = parser.parse_args()
    cfg = vars(args)  # the default config from arguments

    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg_new = EasyDict(yaml.safe_load(f))
        cfg.update(cfg_new)

    return Config(cfg)
    
    