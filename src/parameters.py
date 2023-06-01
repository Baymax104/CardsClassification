"""
this file contains all parameters
"""
import torch.cuda

# Path
# ROOT_DIR = '../dataset'  # for local
# LOG_DIR = '../log_images'  # for local
ROOT_DIR = '/root/autodl-tmp/dataset'  # for autodl
LOG_DIR = '/root/tf-logs'  # for autodl
TARGET_FILENAME = 'cards.csv'
SAVE_DIR = '../model'

# NN
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
RANDOM_SEED = 120
EPOCH = 120
LEARN_RATE = 1e-4
WEIGHT_DECAY = 8.4e-3

# other
CLASSES = 53
