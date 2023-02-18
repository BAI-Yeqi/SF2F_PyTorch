import os
import os.path as osp
import argparse
import yaml
from utils import update_values
from utils import int_tuple, float_tuple, str_tuple, bool_flag

COCO_DIR = os.path.expanduser('data/coco')

parser = argparse.ArgumentParser()

# Optimization hyperparameters
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
# by default, it is disabled
parser.add_argument('--decay_lr_epochs', default=10000, type=float)
parser.add_argument('--beta1', type=float, default=0.9,
                    help='momentum term of adam')
parser.add_argument('--eval_epochs', default=1, type=int)
parser.add_argument('--eval_mode_after', default=10000, type=int)
parser.add_argument('--disable_l1_loss_after', default=10000000, type=int)
parser.add_argument('--path_opts', type=str,
                    default='options/vg_baseline_small.yaml', help="Options.")

parser.add_argument('--workers', default=8, type=int)

# Output options
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--visualize_every', type=int,
                    default=100, help="visualize to visdom.")
parser.add_argument('--timing', action="store_true")
parser.add_argument('--output_dir', type=str)
parser.add_argument('--log_suffix', type=str)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--checkpoint', nargs='+')
parser.add_argument('--resume', type=str)
parser.add_argument('--evaluate', action='store_true',
    help="Set to evaluate the model.")
parser.add_argument('--evaluate_train', action='store_true',
    help="Set to evaluate the training set.")

# For inference in run_model.py
parser.add_argument('--output_demo_dir', default='output/results')
parser.add_argument('--samples_path', default='samples')

# For evaluation
parser.add_argument('--recall_method', default='cos_sim')
parser.add_argument('--facenet_return_pooling', default=False)
# parser.add_argument('--face_gen_mode', default='average_facenet_embedding')
parser.add_argument('--face_gen_mode', nargs='+')

# For test.py
parser.add_argument('--get_faces_from_different_segments', default=False)

# Attention Fuser Mode
parser.add_argument('--train_fuser_only', default=False)
# Train the fuser and decoder
parser.add_argument('--train_fuser_decoder', default=False)
parser.add_argument('--pretrained_path', default=None)
parser.add_argument('--freeze_discriminators', default=False)

# For Inference
parser.add_argument('--input_wav_dir', type=str, default='data/example_audio')
parser.add_argument('--fuser_infer', default=False)

args = parser.parse_args()

options = {
    "data": {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "data_opts": {},
    },
    "optim": {
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "eval_epochs": args.eval_epochs,
    },
    "logs": {
        "output_dir": args.output_dir,
    },
}

with open(args.path_opts, "r") as f:
    # options_yaml = yaml.load(f)
    options_yaml = yaml.full_load(f)
with open(options_yaml["data"]["data_opts_path"], "r") as f:
    # data_opts = yaml.load(f)
    data_opts = yaml.full_load(f)
    options_yaml["data"]["data_opts"] = data_opts

options = update_values(options, options_yaml)
if args.log_suffix:
    options["logs"]["name"] += "-" + args.log_suffix
