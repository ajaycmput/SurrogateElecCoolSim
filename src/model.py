# fully connected neural network:
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

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

def train_model(model, criterion, optimizer, train_loader, epochs=50):
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward() # computes the gradients of the loss function
            optimizer.step() # updates the model's weights and biases and minimizes the loss.
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def main():
    # Load your dataset
    dataset = pd.read_csv('../data/electronics_cooling_simulation_data.csv')
    inputs = dataset[['Power Load (W)', 'Ambient Temp (C)']].values
    targets = dataset[['Circuit Temp (C)']].values

    # Convert to PyTorch tensors
    inputs = torch.Tensor(inputs)
    targets = torch.Tensor(targets)

    # Create a dataset and loader
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Model, criterion, and optimizer
    model = ElectronicsCoolingModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimization is a stochastic gradient descent method

    # Train the model
    train_model(model, criterion, optimizer, train_loader)

    # Save the model
    torch.save(model.state_dict(), '../models/electronics_cooling_model.pth')
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
