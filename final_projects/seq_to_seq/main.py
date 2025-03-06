from dataset import generate_sequence_dataset
from model import Seq2SeqWithAttention
from train import train
from evaluate import evaluate
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, Y = generate_sequence_dataset(1000, 10)
train_size = int(0.8 * len(X))
X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size:], Y[train_size:]

model = Seq2SeqWithAttention(input_size=10, output_size=10, hidden_size=64).to(device)

# Train the model
train(model, X_train, Y_train, num_epochs=10)

# Evaluate the model
evaluate(model, X_val, Y_val)
