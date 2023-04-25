import os
import pprint
from src.edlsm_learner import edlsmLearner
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--directory",
                    type=str,
                    default="./data_scene_flow/training",
                    help="Directory to the dataset")
parser.add_argument("--train_val_split_dir",
                    type=str,
                    default="./dataset",
                    help="Directory to the dataset")
parser.add_argument("--train_dataset_name",
                    type=str,
                    default="tr_160_18_100.txt",
                    help="Training set")
parser.add_argument("--val_dataset_name",
                    type=str,
                    default="val_40_18_100.txt",
                    help="Validation set")
parser.add_argument("--checkpoint_dir",
                    type=str,
                    default="./checkpoints",
                    help="Directory name to save the checkpoints")
parser.add_argument("--logs_path",
                    type=str,
                    default="logs",
                    help="Tensorboard log path")
parser.add_argument("--continue_train",
                    action='store_true',
                    help="Resume training")
parser.add_argument("--gpu", action='store_true', help="Use GPU if available")
parser.add_argument("--init_checkpoint_file",
                    type=str,
                    default="edlsm_24000.ckpt",
                    help="checkpoint file")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="The size of of a sample batch")
parser.add_argument("--psz", type=int, default=18, help="Left patch size")
parser.add_argument("--half_range",
                    type=int,
                    default=100,
                    help="Right patch half range")
parser.add_argument("--image_width",
                    type=int,
                    default=1224,
                    help="Image width, must same as prepare_dataset")
parser.add_argument("--image_height",
                    type=int,
                    default=370,
                    help="Image height, must same as prepare_dataset")
parser.add_argument("--start_step",
                    type=int,
                    default=0,
                    help="Starting training step")
parser.add_argument("--max_steps",
                    type=int,
                    default=50000,
                    help="Maximum number of training iterations")
parser.add_argument("--l_rate", type=float, default=0.01, help="learning rate")
parser.add_argument("--l2", type=float, default=0.0005, help="Weight Decay")
parser.add_argument("--reduce_l_rate_itr",
                    type=int,
                    default=8000,
                    help="Reduce learning rate after this many iterations")
parser.add_argument("--pxl_wghts",
                    type=float,
                    default=[[1.0, 4.0, 10.0, 4.0, 1.0]],
                    help="Weights for three pixel error")
parser.add_argument("--summary_freq",
                    type=int,
                    default=200,
                    help="Logging every summary_freq iterations")
parser.add_argument("--valid_freq",
                    type=int,
                    default=500,
                    help="Logging every valid_freq iterations")
parser.add_argument(
    "--save_latest_freq",
    type=int,
    default=200,
    help="Save the latest model every save_latest_freq iterations")
args = parser.parse_args()


def main():
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    edlsm = edlsmLearner()
    edlsm.train(args)


if __name__ == '__main__':
    main()
