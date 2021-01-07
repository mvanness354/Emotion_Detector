from preprocessing import rnn_preprocessor, fetch_data, max_sequence_length, get_data_loaders_rnn
from RNN import get_device, RNN
import torchtext
import torch
import torch.optim as optim

from tqdm import tqdm, trange

import yaml


emotion_to_idx = {
    "anger": 0,
    "fear": 1,
    "joy": 2,
    "love": 3,
    "sadness": 4,
    "surprise": 5,
}


def train_epoch_rnn(model, train_loader, optimizer):
	model.train()
	total = 0
	loss = 0
	total_loss = 0
	correct = 0
	for (input_batch, input_len, expected_out) in tqdm(train_loader, leave=False, desc="Training Batches"):
		output = model(input_batch.to(get_device()), input_len)
		total += output.size()[0]
		_, predicted = torch.max(output, 1)
		correct += (expected_out == predicted.to("cpu")).cpu().numpy().sum()
		loss = model.compute_Loss(output, expected_out.to(get_device()))
		total_loss += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	# Print accuracy
	# print(f"Train ave loss: {total_loss / len(train_loader)}")
	# print(f"Train accuracy: {correct / total}")


def evaluation_rnn(model, val_loader, verbose=False):
	model.eval()
	loss = 0
	correct = 0
	total = 0
	for (input_batch, input_len, expected_out) in tqdm(val_loader, leave=False, desc="Validation Batches"):
		output = model(input_batch.to(get_device()), input_len)
		total += output.size()[0]
		_, predicted = torch.max(output, 1)
		correct += (expected_out.to("cpu") == predicted.to("cpu")).cpu().numpy().sum()
		loss += model.compute_Loss(output, expected_out.to(get_device()))
	loss /= len(val_loader)
	# Print validation metrics

	if verbose:
		print(f"\nEval ave loss: {loss}")
		print(f"Eval accuracy: {correct / total}")
 
	return loss, correct / total

def train_and_evaluate_rnn(number_of_epochs, model, train_loader, val_loader, learning_rate=0.001, momentum=0.9, verbose=False):
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	losses = []
	accuracies = []

	for epoch in trange(number_of_epochs, desc="Epochs"):
		train_epoch_rnn(model, train_loader, optimizer)
		loss, correct =  evaluation_rnn(model, val_loader, verbose)
		losses.append(loss)
		accuracies.append(correct)
	
		# Early Stopping Criteria: if loss increases twice, stop training
		if len(losses) >= 3 and losses[-3] < losses[-2] < losses[-1]:
			break

	return losses[-1], accuracies[-1]


train_path = "datafiles/train.txt"
val_path = "datafiles/val.txt"
test_path = "datafiles/test.txt"

train, val, test = fetch_data(train_path, val_path, test_path)
max_length = max_sequence_length([tmp[0] for tmp in train], test, [tmp[0] for tmp in val])
embed_model = torchtext.vocab.GloVe()
train_vectorized_rnn = rnn_preprocessor(train, max_length, embed_model)
val_vectorized_rnn = rnn_preprocessor(val, max_length, embed_model)
test_vectorized_rnn = rnn_preprocessor(test, max_length, embed_model, test=True)


# Read in params from yml
with open("params.yml") as f:
	params = yaml.full_load(f)



h = params["h"]
input_dim = len(train_vectorized_rnn[0][0][0])
n_layers = params["num_layers"]
output_dim = len(emotion_to_idx)
n_epochs = params["num_epochs"]
non_linearity = params["non_linearity"]
lstm = params["lstm"]
batch_size = params["batch_size"]
learning_rate=params["learning_rate"]

rnn_train_loader, rnn_val_loader =  get_data_loaders_rnn(train_vectorized_rnn, val_vectorized_rnn, batch_size=batch_size)

model = RNN(input_dim, h, output_dim, n_layers, non_linearity=non_linearity, lstm=lstm).to(get_device())
train_and_evaluate_rnn(n_epochs, model, rnn_train_loader, rnn_val_loader, learning_rate=learning_rate, verbose=True)