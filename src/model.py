# fully connected neural network:
import torch
import matplotlib
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


matplotlib.use('Agg')
class ElectronicsCoolingModel(nn.Module):
    def __init__(self):
        super(ElectronicsCoolingModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 2 input features
        self.relu = nn.ReLU() # activation function
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)  # 1 output feature

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # to prevent overfitting (getting tuned to th training data and persoems poor to unseen (new) data)
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

def train_model(model, criterion, optimizer, train_loader, epochs=55):
    loss_history = []
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, targets)
            loss.backward() # backward pass (gradient calculation)
            optimizer.step() # minimizes the loss.
        loss_history.append(loss.item())
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return loss_history

def main():

    dataset = pd.read_csv('../data/electronics_cooling_simulation_data.csv')
    inputs = dataset[['Power Load (W)', 'Ambient Temp (C)']].values
    targets = dataset[['Circuit Temp (C)']].values

    inputs = torch.Tensor(inputs)
    targets = torch.Tensor(targets)

    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True)

    model = ElectronicsCoolingModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Adam optimization is a stochastic gradient descent method

    loss_history = train_model(model, criterion, optimizer, train_loader)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History Over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('../plots/loss_history.png')
    print("Loss plot saved.")

    torch.save(model.state_dict(), '../models/electronics_cooling_model.pth')
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
