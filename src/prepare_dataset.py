
import os
import json
import torch
import argparse
from preprocessing.preprocess import download, preprocess
import numpy as np
from tqdm import tqdm
import ConfigSpace as CS
import matplotlib.pyplot as plt
import preprocessing.preprocess as p
from preprocessing.process import DataCollatorCTCWithPadding
from preprocessing.preprocess import preprocess, tokenisation


parser = argparse.ArgumentParser(description="Arguments for training script")
parser.add_argument("token", type=str, help="Token to download dataset")
args = parser.parse_args()
token = str(args.token)
data = download(token=token, dataset_name="mozilla-foundation/common_voice_9_0")
training_set, validation_set, test_set = preprocess(dataset=data, num_workers=12)
os.mkdir("training_set")
os.mkdir("validation_set")
os.mkdir("test_set")
training_set.save_to_disk("training_set")
validation_set.save_to_disk("validation_set")
test_set.save_to_disk("test_set")