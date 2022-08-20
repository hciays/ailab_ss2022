import os
import json
import argparse
from datasets import load_from_disk
from preprocessing.preprocess import download, preprocess, tokenisation

parser = argparse.ArgumentParser(description="Arguments for training script")
parser.add_argument("--token", type=str, help="Token to download dataset")
parser.add_argument(
    "--workers", type=int, help="Number of workers to use for preprocessing"
)
args = parser.parse_args()
token = str(args.token)
workers = int(args.workers)
data = download(token=token, dataset_name="mozilla-foundation/common_voice_9_0")
training_set, validation_set, test_set = preprocess(dataset=data, num_workers=workers)
os.mkdir("training_set")
os.mkdir("validation_set")
os.mkdir("test_set")
training_set.save_to_disk("training_set")
validation_set.save_to_disk("validation_set")
test_set.save_to_disk("test_set")
vocab_dict = tokenisation(
    load_from_disk("training_set"),
    load_from_disk("validation_set"),
    load_from_disk("test_set"),
    num_of_proc=workers,
    batch_size=128,
)
with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)
