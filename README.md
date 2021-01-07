# Emotion Detector
This project is code from a submission to the following [Kaggle Project](https://www.kaggle.com/c/p3-cs4740-2020fa/leaderboard).  In this project, we were given a dataset of tweets which are tagged with a corresponding emotion.  For example, the tweet "feel like I am still looking at a blank canvas blank pieces of paper" is tagged with the sadness emotion.  Our task was to build a model to predict the emotion of a given tweet.  Our final submission to Kaggle, which acheives a test accuracy of roughly 90%, currently sits at 12th out of over 100 submissions.  

Our model first tokenizes the given tweet and obtains the GloVe embedding for each token.  The Glove embeddings are pretrained from the 840B Common Crawl and loaded in using [torchtext](https://torchtext.readthedocs.io/en/latest/vocab.html).  We then use PyTorch to build a Recurrent Neural Network (RNN) that sequentially processes the inputted token embeddings.  We use the final token's hidden state as a representation of the whole tweet, and we pass this last hidden state through a fully connected linear layer to output the probablity that the tweet belongs to each of the emotion classes.

# Usage
First, install the necessary python packages using pip: `pip install -r requirements.txt`.  To configure the hyperparameters for the model, edit the `params.yml` file accordingly.  The hyperparameters are the following:

| Hyperparameter | Description |
| --- | --- |
| h | hidden layer dimension |
| num_layers | number of hidden layers in RNN |
| non_linearity | type of non-linear activation used (either "tanh" or "relu") |
| lstm | whether or not to use LSTMs (True/False) |
| learning_rate | the learning rate for the optimizer |
| num_epochs | number of epochs to train for |
| batch_size | batch size for data loader |

Lastly, run the model with `python main.py`.  This assumes that the hyperparameters are stored in a YAML file called `params.yml` in the current directory and the data files are stored in the `datafiles/` subdirectory.
