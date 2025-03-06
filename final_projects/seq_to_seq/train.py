import torch
import torch.optim as optim
from model import Seq2SeqWithAttention
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

def train(model, X_train, Y_train, num_epochs=10, batch_size=64, teacher_forcing_ratio=0.5):
    model.train()  
    total_samples = len(X_train)
    
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(Y_train, dtype=torch.long))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)

            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)

        accuracy = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}, Accuracy: {accuracy:.2f}%")
