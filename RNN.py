import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"

class RNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, non_linearity="relu", lstm=False):
    super(RNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.lstm = lstm
    self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, nonlinearity=non_linearity, batch_first=True)
    if self.lstm:
      self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
     
  
    self.linear = nn.Linear(hidden_dim, output_dim)
    self.softmax = nn.LogSoftmax(dim=1)
    self.loss = nn.NLLLoss()

  def compute_Loss(self, predicted_vector, gold_label):
    return self.loss(predicted_vector, gold_label)

  def forward(self, inputs, lengths):

    # Pack inputs, which have been padded with 0s so they are all
    # the same length
    inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

    output, hidden = self.rnn(inputs)

    # LSTM output is (hn, cn)
    if self.lstm:
      hidden = hidden[0]

    # Undo packing operation
    output,_ = pad_packed_sequence(output, batch_first=True)

    # Use the hidden layer of the last word in sequence as output to linear
    out = self.linear(hidden[-1])
    
    predicted_vector = self.softmax(out)
    return predicted_vector

  def load_model(self, save_path):
    self.load_state_dict(torch.load(save_path))

  def save_model(self, save_path):
    torch.save(self.state_dict(), save_path)