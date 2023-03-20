# Text Generation using Tranformer : 
This repository contains PyTorch code for training a character-level language model using the Transformer architecture. The trained model can generate text character-by-character in a similar style to the training data.

1) The Transformer model is implemented in several sub-modules.
    * The Head class implements one head of self-attention. 
    * The MultiHeadAttention scaled dot product attention allows a network to attend over a sequence. sub-queries, sub-keys, and sub-values, which we pass through the scaled dot product attention independently. Afterward, we concatenate the heads and combine them with a final weight matrix.
    * The FeedFoward class implements a simple linear layer followed by a non-linearity. ![image](https://user-images.githubusercontent.com/83644731/226430054-3a0dbaba-0a0a-4b5f-9c7f-b75ad87e0f9c.png)

    * The Block class implements the main Transformer block, which consists of communication (multi-head self-attention) followed by computation (feed-forward network). 
    * The TransformerModel class got an encoder-decoder structure where the encoder takes an input text and generates an attention-based representation.
    

2) The estimate_loss function computes the loss on a batch of data for both the train and validation set. This function is used to estimate the performance of the model while training.

3) Finally, the code trains the model for a given number of iterations, prints the training loss every eval_interval steps, and computes the validation loss every eval_iters steps. The trained model is then saved to disk.
![image](https://user-images.githubusercontent.com/83644731/226430251-799d6b02-ff92-47a1-8453-773b54900fc3.png)

## Requirements:
This code was developed using Python 3.7 and PyTorch 1.7.1. The following packages are required:

torch


# Usage:
The code can be run from the command line as follows:
python demo.py This will train a model on the input text file specified in train.py, which is set to input.txt by default. The trained model parameters will be saved to a file named model.pt and generate text using the trained model
#Acknowledgements
This code is based on the karpathy/char-rnn repository, which implements a character-level language model using a recurrent neural network (RNN). The Transformer architecture used in this repository was introduced in the paper "Attention Is All You Need" by Vaswani et al.
