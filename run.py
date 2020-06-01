#!pip install --upgrade -q git+https://github.com/MisterMap/undeepvo.git@develop
#example: python3 train.py -epoch 10 -max_depth 100 -split 100 10 10 -frames_range 20 260 2
# salloc -p gpu_big --mem 100Gb --gpus 4
#python3 run.py -epoch 20 -max_depth 100 -split 3600 235 235 -frames_range 0 4070 1 -mlflow_tags_name Jhores_slurm

import argparse

import pykitti.odometry

from undeepvo.criterion import UnsupervisedCriterion
from undeepvo.models import UnDeepVO
from undeepvo.problems import UnsupervisedDatasetManager, UnsupervisedDepthProblem
from undeepvo.utils import OptimizerManager, TrainingProcessHandler

parser = argparse.ArgumentParser(description='Run parameters')
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

args = parser.parse_args()

MAIN_DIR = args.main_dir
lengths = args.split

dataset = pykitti.odometry(MAIN_DIR, '08', frames=range(*args.frames_range))
dataset_manager = UnsupervisedDatasetManager(dataset, lenghts=lengths)

model = UnDeepVO(args.max_depth).cuda()

criterion = UnsupervisedCriterion(dataset_manager.get_cameras_calibration("cuda:0"),
                                  0.1, 1, 0.85)
handler = TrainingProcessHandler(mlflow_tags={"name": args.mlflow_tags_name},
                                 mlflow_parameters={"image_step": args.frames_range[2], "max_depth": args.max_depth,
                                                    "epoch": args.epoch})
optimizer_manger = OptimizerManager()
problem = UnsupervisedDepthProblem(model, criterion, optimizer_manger, dataset_manager, handler,
                                   batch_size=5, name="undeepvo")
problem.train(args.epoch)
