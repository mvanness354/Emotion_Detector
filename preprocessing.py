import torch
import torch.nn as nn
from torch.nn import init

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from tqdm import tqdm, trange


emotion_to_idx = {
    "anger": 0,
    "fear": 1,
    "joy": 2,
    "love": 3,
    "sadness": 4,
    "surprise": 5,
}
idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}
UNK = "<UNK>"

def max_sequence_length(train, test, val):
  length = 0
  for dataset in [train,val,test]:
    for document in dataset:
      if len(document)>length:
        length = len(document)
  return length 

def all_zeros(x):
  return torch.all(x.eq(torch.zeros(x.shape)))

def fetch_data(train_data_path, val_data_path, test_data_path):
    """fetch_data retrieves the data from a json/csv and outputs the validation
    and training data

    :param train_data_path:
    :type train_data_path: str
    :return: Training, validation pair where the training is a list of document, label pairs
    :rtype: Tuple[
        List[Tuple[List[str], int]],
        List[Tuple[List[str], int]],
        List[List[str]]
    ]
    """
    with open(train_data_path) as training_f:
        training = training_f.read().split("\n")[1:-1]
    with open(val_data_path) as valid_f:
        validation = valid_f.read().split("\n")[1:-1]
    with open(test_data_path) as testing_f:
        testing = testing_f.read().split("\n")[1:-1]
    
    # If needed you can shrink the training and validation data to speed up somethings but this isn't always safe to do by setting k < 10000
    # k = #fill in
    # training = random.shuffle(training)
    # validation = random.shuffle(validation)
    # training, validation = training[:k], validation[:(k // 10)]

    tra = []
    val = []
    test = []
    for elt in training:
        if elt == '':
            continue
        txt, emotion = elt.split(",")
        tra.append((txt.split(" "), emotion_to_idx[emotion]))
    for elt in validation:
        if elt == '':
            continue
        txt, emotion = elt.split(",")
        val.append((txt.split(" "), emotion_to_idx[emotion]))
    for elt in testing:
        if elt == '':
            continue
        txt = elt
        test.append(txt.split(" "))

    return tra, val, test




def rnn_preprocessor(data, max_seq_len, embed_model, test=False):

    vectorized_data = []

    # In test there are no labels
    if test:
      for mention in data:
        vectors=[]
        for word in mention:
          vector = embed_model[word] # Get the word embedding

          
          if not all_zeros(vector):  
            vectors.append(vector)

        # Pad with all 0 embeddings
        while (len(vectors) < max_seq_len):
          vectors.append(embed_model[""])

        # Add embedings and original doc length
        vectorized_data.append((vectors,len(mention)))

    else:
      for mention, label in data:
        vectors=[]
        for word in mention:
          vector = embed_model[word] # Will be zeros if unknown

          # Only add embedding if it is a known word
          if not all_zeros(vector):
            vectors.append(vector)

        # Pad with all 0 embeddings
        while (len(vectors)<max_seq_len):
          vectors.append(embed_model[""])

        # Add embeddings, original doc length, and true label
        vectorized_data.append((vectors,len(mention), label))

    return vectorized_data

class EmotionDatasetRNN(Dataset):
    def __init__(self, data):
        self.X = [X for X, _, _ in data]
        self.y = torch.LongTensor([y for _, _, y in data])
        self.data=data
        self.l = [l for _, l, _ in data]
        self.len = len(data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        x = torch.stack(self.X[index])
        l= torch.tensor(int(self.l[index]))
        y= self.y[index]
        return x,l,y

def get_data_loaders_rnn(train, val, batch_size=16):
    # First we create the dataset given our train and validation lists
    dataset = EmotionDatasetRNN(train + val)

    # Then, we create a list of indices for all samples in the dataset
    train_indices = [i for i in range(len(train))]
    val_indices = [i for i in range(len(train), len(train) + len(val))]

    # Now we define samplers and loaders for train and val
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader