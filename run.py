#!pip install --upgrade -q git+https://github.com/MisterMap/undeepvo.git@develop
# example: python3 train.py -epoch 10 -max_depth 100 -split 100 10 10 -frames_range 20 260 2
# salloc -p gpu_big --mem 100Gb --gpus 4
# python3 run.py -epoch 20 -max_depth 100 -split 3600 235 235 -frames_range 0 4070 1 -mlflow_tags_name Jhores_slurm

import argparse

import pykitti.odometry

from undeepvo.criterion import UnsupervisedCriterion, SupervisedCriterion
from undeepvo.models import UnDeepVO, DepthNet
from undeepvo.problems import UnsupervisedDatasetManager, UnsupervisedDepthProblem, SupervisedDatasetManager, \
    SupervisedDepthProblem
from undeepvo.utils import OptimizerManager, TrainingProcessHandler

from undeepvo.data.supervised import GroundTruthDataset

parser = argparse.ArgumentParser(description='Run parameters')
parser.add_argument('-method',
                    default='unsupervised',
                    type=str,
                    help='Unsupervised or supervised method')

parser.add_argument('-main_dir',
                    default='dataset',
                    type=str,
                    dest='main_dir',
                    help='Path to the directory containing data')

parser.add_argument('-split',
                    default=(100, 10, 10),
                    type=int,
                    dest='split',
                    nargs='+',
                    help='train/test/valid sequence split')

parser.add_argument('-frames_range',
                    default=(0, 120, 1),
                    type=int,
                    dest='frames_range',
                    nargs='+',
                    help='frames_range from sequence')

parser.add_argument('-mlflow_tags_name',
                    default="Jhores",
                    type=str,
                    dest='mlflow_tags_name',
                    help='tag name for mlflow experiment')

parser.add_argument('-epoch',
                    default=10,
                    type=int,
                    dest='epoch',
                    help='epochs to train')

parser.add_argument('-max_depth',
                    default=100,
                    type=int,
                    dest='max_depth',
                    help='max_depth value')

parser.add_argument('-lambda_disparity',
                    default=0.01,
                    type=float,
                    dest='lambda_disparity',
                    help='lambda_disparity loss value')

parser.add_argument('-lambda_s',
                    default=0.85,
                    type=float,
                    dest='lambda_s',
                    help='lambda_s loss value')

parser.add_argument('-lambda_rotation',
                    default=0.1,
                    type=float,
                    dest='lambda_rotation',
                    help='lambda_rotation loss value')

parser.add_argument('-lambda_position',
                    default=0.01,
                    type=float,
                    dest='lambda_position',
                    help='lambda_position loss value')

parser.add_argument('-lr',
                    default=0.0001,
                    type=float,
                    dest='lr',
                    help='learning rate')

parser.add_argument('-batch',
                    default=5,
                    type=int,
                    dest='batch',
                    help='batch size')

parser.add_argument('-supervised_lambda',
                    default=0.1,
                    type=float,
                    help='lambda os supervised method')

args = parser.parse_args()

MAIN_DIR = args.main_dir
lengths = args.split
if args.method == "unsupervised":
    dataset = pykitti.odometry(MAIN_DIR, '08', frames=range(*args.frames_range))
    dataset_manager = UnsupervisedDatasetManager(dataset, lenghts=lengths)

    model = UnDeepVO(args.max_depth).cuda()

    criterion = UnsupervisedCriterion(dataset_manager.get_cameras_calibration("cuda:0"),
                                      args.lambda_position, args.lambda_rotation, args.lambda_s, args.lambda_disparity)
    handler = TrainingProcessHandler(enable_mlflow=True, mlflow_tags={"name": args.mlflow_tags_name},
                                     mlflow_parameters={"image_step": args.frames_range[2], "max_depth": args.max_depth,
                                                        "epoch": args.epoch, "lambda_position": args.lambda_position,
                                                        "lambda_rotation": args.lambda_rotation,
                                                        "lambda_s": args.lambda_s,
                                                        "lambda_disparity": args.lambda_disparity})
    optimizer_manager = OptimizerManager(lr=args.lr)
    problem = UnsupervisedDepthProblem(model, criterion, optimizer_manager, dataset_manager, handler,
                                       batch_size=args.batch, name="undeepvo")

elif args.method == "supervised":
    dataset = GroundTruthDataset(length=lengths)
    dataset_manager = SupervisedDatasetManager(dataset, lenghts=lengths)

    model = DepthNet(args.max_depth).cuda()

    criterion = SupervisedCriterion(args.supervised_lambda)

    handler = TrainingProcessHandler(enable_mlflow=True, mlflow_tags={"name": args.mlflow_tags_name},
                                     mlflow_parameters={"image_step": args.frames_range[2], "max_depth": args.max_depth,
                                                        "epoch": args.epoch,
                                                        "supervised_lambda": args.supervised_lambda})
    optimizer_manager = OptimizerManager(lr=args.lr)
    problem = SupervisedDepthProblem(model, criterion, optimizer_manager, dataset_manager, handler,
                                     batch_size=args.batch, name="supervised depthnet")
else:
    exit("Unknown method")

problem.train(args.epoch)
