#!/usr/bin/env python3

import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, network, optimizer, loss):
        self._network = network
        self._optimizer = optimizer
        self._loss = loss
        self._cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if self._cuda else "cpu")

    def train(self, input_loader, n_epochs=5):
        self._network.train()

        for epoch in range(n_epochs):
            print('Epoch {}'.format(epoch+1))

            epoch_losses = []

            for X, y in tqdm(input_loader):
                X, y = X.to(self._device), y.to(self._device)

                predictions = self._network(X)
                loss = self._loss(predictions, y)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                epoch_losses.append(loss.item())

            print('[{}/{}] Loss: {:.8f}'.format(epoch+1, n_epochs, np.mean(epoch_losses)))

    def validate(self):
        pass
