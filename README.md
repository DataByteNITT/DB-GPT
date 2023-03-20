# Char-RNN using a Transformer
This repository contains PyTorch code for training a character-level language model using the Transformer architecture. The trained model can generate text character-by-character in a similar style to the training data. The code begins by importing the required PyTorch libraries, and setting hyperparameters such as batch size, maximum context length for predictions, learning rate, etc. The code then sets the seed for reproducibility.

Next, the code reads in a text file and creates a vocabulary by identifying all unique characters in the text. It also creates mappings from characters to integers and vice versa. The text is split into train and validation sets.

The get_batch function generates a small batch of data of inputs x and targets y. It takes either the train or validation split as input, selects a random sequence of indices, and selects the corresponding x and y values from the dataset. These x and y tensors are returned.

The estimate_loss function computes the loss on a batch of data for both the train and validation set. This function is used to estimate the performance of the model while training.

The Transformer model is implemented in several sub-modules. The Head class implements one head of self-attention. The MultiHeadAttention class creates multiple heads of self-attention in parallel. The FeedFoward class implements a simple linear layer followed by a non-linearity. The Block class implements the main Transformer block, which consists of communication (multi-head self-attention) followed by computation (feed-forward network). The TransformerModel class combines these sub-modules into the complete Transformer model.

Finally, the code trains the model for a given number of iterations, prints the training loss every eval_interval steps, and computes the validation loss every eval_iters steps. The trained model is then saved to disk.

## Requirements
This code was developed using Python 3.7 and PyTorch 1.7.1. The following packages are required:

torch

## Usage
The code can be run from the command line as follows:

python demo.py
This will train a model on the input text file specified in train.py, which is set to input.txt by default. The trained model parameters will be saved to a file named model.pt and  generate text using the trained model

## Code Structure
The code is organized as follows:

## Acknowledgements
This code is based on the karpathy/char-rnn repository, which implements a character-level language model using a recurrent neural network (RNN). The Transformer architecture used in this repository was introduced in the paper "Attention Is All You Need" by Vaswani et al.
