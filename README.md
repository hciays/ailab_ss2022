# Transformer based ASR System for German Language 

This repository contains code to build a German ASR system based on transformers and the [Common voice 9.0 dataset](https://commonvoice.mozilla.org/en/datasets). We use the pre-trained [Wav2Vec Transformer](https://huggingface.co/docs/transformers/model_doc/wav2vec2) as well as the pre-trained [Wave2Vec Conformer](https://huggingface.co/docs/transformers/model_doc/wav2vec2-conformer) , both from Facebook.


## Setup
Recommended Python version 3.9 

1. Create a new conda-environment and activate it.
   * ``conda create -n ailab python=3.9``
   * ``conda activate ailab``
   

2. Install all requirements.
   * ``pip install -r  requirements.txt``