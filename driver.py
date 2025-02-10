import kagglehub
import pandas as pd
import torch
import numpy as np
from src.viz import build_dashboard
from src.models import FeedFowardNet, SimpleNet
from config import config
from src.DataFactory import DataFactory

data_factory = DataFactory(config)

# build_dashboard(data_factory.df)

train_loader, val_loader, test_loader = data_factory.get_loaders()

net = SimpleNet(n_features=8, n_outputs=1)
criterion = torch.nn.MSELoss()
optimizer = config['training_space']['optimizer'](net.parameters(), **config['training_space']['optimizer_space'])

# Training
for epoch in range(config['training_space']['epochs']):
    mean_train_loss = 0
    mean_val_loss = 0

    # Training
    for samples, targets in train_loader:

        predictions = net(samples)
        train_loss = criterion(predictions, targets)
        train_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        mean_train_loss += train_loss.item()

    mean_train_loss /= len(train_loader)

    # Validation
    for samples, targets in val_loader:

        predictions = net(samples)
        val_loss = criterion(predictions, targets)
        val_loss.backward()

        mean_val_loss += val_loss.item()

    mean_val_loss /= len(val_loader)

    print(f'Train loss: {round(mean_train_loss, 6)} Val loss:{round(mean_val_loss, 6)}')





