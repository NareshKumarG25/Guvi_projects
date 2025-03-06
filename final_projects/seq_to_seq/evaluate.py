import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def evaluate(model, X_val, Y_val):
    model.eval()  
    total_samples = len(X_val)

    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(Y_val, dtype=torch.long))
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():  
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output, tgt)
            val_loss += loss.item()

            pred = output.argmax(1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)

    accuracy = correct / total * 100
    avg_loss = val_loss / len(val_loader)
    
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy