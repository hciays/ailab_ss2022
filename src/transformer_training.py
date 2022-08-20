import argparse
import os
import numpy as np
import preprocessing.preprocess as p
from preprocessing.process import DataCollatorCTCWithPadding
from datasets import load_from_disk, load_metric
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TOTAL_TRAINING_SIZE = 439325
TOTAL_VALIDATION_SIZE = 16033

wer_metric = load_metric("wer")
if os.path.exists("transformer_results") is True:
    print("Model logging directory already exists.")
else:
    print("Creating logging directory.........................")
    os.mkdir("transformer_results")
    print("Directory transformer_results created.........................")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


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
    "--epochs", type=int, default=20, help="An integer to specify the number of epochs."
)
parser.add_argument(
    "--data", type=int, default=20, help="Integer to specify the percentage of training data to use."
)
parser.add_argument(
    "--resume_training",
    type=bool,
    help="Boolean to specify if we resume training or restart all over.",
    default=False,
    action=argparse.BooleanOptionalAction
)
args = parser.parse_args()

if args.epochs < 0:
    raise ValueError("The number of epochs must be positive")
if args.data > 100 or args.data < 0:
    raise ValueError("Enter a valid percentage between 1 and 100")
Sub_sample_training = int((args.data * TOTAL_TRAINING_SIZE) / 100)
Sub_sample_validation = int((args.data * TOTAL_VALIDATION_SIZE) / 100)
print(
    "The model will be trained for {0} epochs on {1} % of the training set ......................... \n".format(
        args.epochs, args.data
    )
)

tokenizer = Wav2Vec2CTCTokenizer(
    "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

print("Processing training and validation data.........................\n")
train = load_from_disk("training_set")
subset_train = train.shuffle(seed=p.SEED).select(range(Sub_sample_training))
val = load_from_disk("validation_set")
subset_val = val.shuffle(seed=p.SEED).select(range(Sub_sample_validation))


prepared_training = subset_train.map(prepare_dataset, num_proc=16)
prepared_evaluation = subset_val.map(prepare_dataset, num_proc=1)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
print("Defining the model .........................\n")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    attention_dropout=0.1,
    hidden_dropout=0.1,
    mask_time_prob=0.05,
)
model.freeze_feature_extractor()
training_arg = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    seed=p.SEED,
    fp16=True,
    weight_decay=0.005,
    num_train_epochs=args.epochs,
    warmup_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=1000,
    save_steps=1000,
    per_device_train_batch_size=16,
    save_total_limit=3,
    dataloader_num_workers=8,
)
trainer = Trainer(
    model=model,
    args=training_arg,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=prepared_training,
    eval_dataset=prepared_evaluation,
    tokenizer=processor.feature_extractor,
)
print("Training.........................")

if args.resume_training:
    print("Resuming training from last checkpoint.........................")
    trainer.train(resume_from_checkpoint=True)
else:
    print("Starting training.........................")
    trainer.train()
r = trainer.evaluate()
print(r)
