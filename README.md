# Transformer based ASR System for German Language 

This repository contains code to build a German ASR system based on transformers and the [Common voice 9.0 dataset](https://commonvoice.mozilla.org/en/datasets). We use the pre-trained [Wav2Vec Transformer](https://huggingface.co/docs/transformers/model_doc/wav2vec2) as well as the pre-trained [Wave2Vec Conformer](https://huggingface.co/docs/transformers/model_doc/wav2vec2-conformer) , both from Facebook.

## Hardware Requirements

1. At least 100Gb Disk space


## Setup
Recommended Python version 3.9 

1. Create a new conda-environment and activate it.
   * ``conda create -n ailab python=3.9``
   * ``conda activate ailab``
   

2. Install all requirements. (Please manually install cuda if you plan to use a GPU)
   * ``pip install -r  requirements.txt``


## Preprocessing
The preprocessing does the following:

1. Downloads the dataset via [HuggingFace](https://huggingface.co/datasets/common_voice). You need to create an account and get a token to be able to download this particular dataset.
2. Removes unnecessary columns from the dataset.
3. Resamples the audio files from an initial frequency of 48 000 Hz to 16 000Hz.
4. Removes special characters.
5. Takes care of padding the sentences.
6. Saves the data set in directories ``training_set, validation_set, test_set``.
7. Creates the model's tokenizer and saves it.

The script requires following arguments: Token ( string from huggingface), num_workers(int). It can be launched via : ``python prepare_dataset.py --token Token --workers num_workers``


## Training

There are 2 training scripts: One for the Transformer model and one for the Conformer model. The scripts need as parameters the following:
- number of epochs: An integer
- Percentage of data to use: an Integer in [0, 100].
- If to resume the training or not.


To train the transformer for Example, One could use the following command:
``python transformer_training.py --epochs 10 --data 50 --no-resume_training``