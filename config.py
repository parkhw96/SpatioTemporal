from cmath import inf
from numpy import Inf
import pandas as pd

# Data Setup
SEQ_SAMPLING = 200
SAMPLE_UNIT = 4
OVERLAP_GAP = 20
MISSING_VALUE = Inf

WIDTH = 320
HEIGHT = 240

RADIUS = 7
IOU_THRESHOLD = 0.1

# Neural Network Training
BATCH_SIZE = 4
FEATURE_DIM = 44
RNN_DIM = 256
INPUT_DIM = 1
HIDDEN_DIM = 4
KERNEL_SIZE = (5, 5)
NUM_LAYERS = 2
OUTPUT_DIM = 484
LR = 1e-4
W_DECAY = 0
N_EPOCHS = 2000
