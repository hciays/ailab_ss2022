import argparse
from ast import arg
import os
import torch
from jiwer import wer
from torchsummary import summary
from datetime import datetime
import preprocessing.preprocess as p
from preprocessing.process import DataCollatorCTCWithPadding
from datasets import load_from_disk
from transformers import Wav2Vec2Processor, Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = ""


TOTAL_TEST_SIZE = 16033
model, processor = None, None


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        batch["speech"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


parser = argparse.ArgumentParser(description="Arguments for training script")
parser.add_argument(
    "--data",
    type=int,
    default=20,
    help="An integer to specify the percentage of data to use for test.",
)
parser.add_argument(
    "--model",
    type=str,
    default="transformer",
    help="String to specify if we use transformer or conformer.",
)
parser.add_argument(
    "--model_dir",
    type=str,
    default="transformer_model",
    help="String to specify the directory where the model is saved.",
)
parser.add_argument(
    "--print_examples",
    type=bool,
    help="Boolean to specify if we print some examples or not.",
    default=True,
    action=argparse.BooleanOptionalAction,
)

args = parser.parse_args()
if os.path.exists(args.model_dir) is False:
    raise ValueError("The provided model directory does not exist")

if args.model.lower() == "transformer":
    from transformers import Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    summary(model=model)
elif args.model.lower() == "conformer":
    from transformers import Wav2Vec2ConformerForCTC

    model = Wav2Vec2ConformerForCTC.from_pretrained(args.model_dir)
    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    summary(model=model)
else:
    print(args.model)
    raise ValueError("The model should be a string. Either transformer or conformer")
if args.data > 100 or args.data < 0:
    raise ValueError("Enter a valid percentage between 1 and 100")
Sub_sample_test = int((args.data * TOTAL_TEST_SIZE) / 100)
print(
    "The {0} model in directory {1} will be tested on {2} % of the test set ......................... \n".format(
        args.model, args.model_dir, args.data
    )
)

print("Processing test data.........................\n")
now = datetime.now()
print("Beginning time : ", now)
test = load_from_disk("test_set")
subset_test = test.shuffle(seed=p.SEED).select(range(Sub_sample_test))
prepared_test = subset_test.map(prepare_dataset, num_proc=1)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
now = datetime.now()
print("Ending time : ", now)

print("Testing .........................\n")
now = datetime.now()
print("Beginning time : ", now)
tester = Trainer(
    model=model,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)
pred, pred_, metrics = tester.predict(test_dataset=prepared_test)
now = datetime.now()
print("Ending time : ", now)

predicted_ids = torch.argmax(torch.tensor(pred), dim=-1)
hypothesis = processor.batch_decode(predicted_ids)
for idx in range(len(hypothesis)):
    hypothesis[idx] = str(hypothesis[idx]).replace("[PAD]", "")
ground_truth = prepared_test["sentence"]
error = round(wer(ground_truth, hypothesis) * 100, 2)
print("Word Error Rate: {0} %".format(error))
if args.print_examples:
    for i in range(len(hypothesis)):
        print(
            "Sentence {0}:  Ground truth: {1}. Prediction: {2} \n".format(
                (i + 1), ground_truth[i], hypothesis[i]
            )
        )
