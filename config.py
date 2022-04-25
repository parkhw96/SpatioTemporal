from cmath import inf
from numpy import Inf
import pandas as pd

# Data Setup
SEQ_SAMPLING = 100
SAMPLE_UNIT = 4
OVERLAP_GAP = 20
MISSING_VALUE = 10000

# ConvLSTM hypterparameter
WIDTH = 108
HEIGHT = 72

# ConvLSTM, LSTM hyperparameter
DISTANCE_THRESHOLD = 2

# ConvLSTM hyperparameter
LOCAL_MAXIMA_THRESHOLD = 0.05
RADIUS = 3

# Neural Network Training
BATCH_SIZE = 8
FEATURE_DIM = 44
RNN_DIM = 640
INPUT_DIM = 1
HIDDEN_DIM = 256
KERNEL_SIZE = (5, 5)
NUM_LAYERS = 3
OUTPUT_DIM = 484
LR = 1e-4
W_DECAY = 0.1
N_EPOCHS = 3000
