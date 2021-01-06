
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