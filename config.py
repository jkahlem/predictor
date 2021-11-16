import argparse
import os
from typing import Tuple

import torch

# The port to use
def get_port() -> int:
    global port
    return port

# the directory where the script is located
def get_script_dir() -> str:
    return os.path.dirname(os.path.realpath(__file__))

# the directory where the labels should be stored
def get_labels_path() -> str:
    return os.path.join(get_script_dir(), 'resources', 'data', 'labels.csv')

# the model type and model name to use
def get_model_config() -> Tuple[str, str]:
    return 'bert', 'bert-base-uncased'

# if true, setups the predictor for integration testing (= "mocks" the model functionality)
def is_test_mode() -> bool:
    return False

# returns true if CUDA is available
def is_cuda_available() -> bool:
    global use_cuda
    if use_cuda and torch.cuda.is_available():
        return True
    return False

# Loads the configuration of the predictor by command line
def load_config():
    global port, use_cuda

    parser = argparse.ArgumentParser(description="Runs a server for making prediction of return types")
    parser.add_argument('--port', dest='port', default=10000, help="The port to listen to for messages", type=int)
    parser.add_argument('--no-cuda', dest='cuda', default=True, help="Does not use CUDA if set. Does nothing if CUDA is not available.", action='store_false')
    args = parser.parse_args()

    port = args.port
    use_cuda = args.cuda