
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
import logging
import math
import argparse
from typing import List, Tuple, Callable, Optional
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from collections import namedtuple
from itertools import product
from functools import partial

import numpy as np
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
torch.manual_seed(42)

train_sets, validation_set, test_set = get_datasets()
(train_loader_labeled, train_loader_unlabeled), validation_loader, test_loader = create_loaders(train_sets['labeled'], train_sets['unlabeled'], validation_set, test_set)