import os
import re
import torchaudio
from datasets import load_dataset, Audio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SEED = 2022
SPECIAL = '[\!\?\.\,\-\;\:\"\“\%\‘\”\�\']'
RESAMPLER = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)


def download(token: str, dataset_name: str):
    """
    Download the dataset
    return: dataset as dictionary
    """
    data = load_dataset(dataset_name, use_auth_token=token)
    return data


def remove_and_resample(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = RESAMPLER(speech_array).squeeze().numpy()
    batch["sentence"] = re.sub(SPECIAL, '', batch["sentence"]).lower()
    return batch


def extract_all_chars(batch):
    """
    Extract all unique characters in a batch of sentences
    """
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def preprocess(dataset, num_workers):
    """
    Does the dataset preprocessing
    return the preprocessed dataset splits
    """

    # Removing unecessary columns.
    dataset = dataset.remove_columns(
        ["client_id", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"])

    # Resamplig the audio data.
    dataset = dataset.map(remove_and_resample, num_proc=num_workers)

    # Changing the sampling frequency.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    # Splitting the data
    training_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]

    return training_set, validation_set, test_set


def tokenisation(train, val, test, num_of_proc, batch_size):
    """
    Creates the vocabulary for the the model.
    Returns vocabulary as a dictionnary.
    """

    vocabs_train = train.map(extract_all_chars,
                             batched=True,
                             batch_size=batch_size,
                             keep_in_memory=True,
                             num_proc=num_of_proc,
                             remove_columns=train.column_names)
    vocabs_eval = val.map(extract_all_chars,
                          batched=True,
                          batch_size=batch_size,
                          keep_in_memory=True,
                          num_proc=num_of_proc,
                          remove_columns=train.column_names)
    vocabs_test = test.map(extract_all_chars,
                           batched=True,
                           batch_size=batch_size,
                           keep_in_memory=True,
                           num_proc=num_of_proc,
                           remove_columns=train.column_names)

    vocab = set()

    for vocab_data in {vocabs_train, vocabs_eval, vocabs_test}:
        for char_list in vocab_data['vocab']:
            vocab.update(char_list)
    vocab_dict = {v: k for k, v in enumerate(vocab)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    print(f'Vocab size:{len(vocab_dict)}')

    return vocab_dict
