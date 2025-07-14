import torch
import os
from telepath.fmriTransformer.model import FMRITransformerModel, Config
from telepath.fmriTransformer.train import train_one_epoch, evaluate
from telepath.fmriTransformer.data_load_align import prepare_and_save_aligned_data
from torch.utils.data import DataLoader
import wandb

print("running test.py")