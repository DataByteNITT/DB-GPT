# Char-RNN using a Transformer
This repository contains PyTorch code for training a character-level language model using the Transformer architecture. The trained model can generate text character-by-character in a similar style to the training data.

## Requirements
This code was developed using Python 3.7 and PyTorch 1.7.1. The following packages are required:

torch
tqdm
## Usage
The code can be run from the command line as follows:

python demo.py
This will train a model on the input text file specified in train.py, which is set to input.txt by default. The trained model parameters will be saved to a file named model.pt and  generate text using the trained model

## Code Structure
The code is organized as follows:

## Acknowledgements
This code is based on the karpathy/char-rnn repository, which implements a character-level language model using a recurrent neural network (RNN). The Transformer architecture used in this repository was introduced in the paper "Attention Is All You Need" by Vaswani et al.
