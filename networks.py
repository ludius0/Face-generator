import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CelebADataset(Dataset):
    
    def __init__(self, file, device="cpu", img_object="img_align_celeba"):
      self.file_object = h5py.File(file, 'r')
      self.dataset = self.file_object[img_object]
      self.img_shape = (218, 178, 3)
      self.device = device
    
    def __len__(self):
      return len(self.dataset)
    
    def __getitem__(self, index):
      if (index >= len(self.dataset)):
        raise IndexError()
      img = np.array(self.dataset[str(index)+'.jpg'])

      if self.device == "cpu":
        return torch.FloatTensor(img) / 255.0
      else:
        return torch.cuda.FloatTensor(img) / 255.0
    
    def plot_image(self, index):
      plt.imshow(np.array(self.dataset[str(index)+'.jpg']), interpolation='nearest')
      plt.show()

class View(nn.Module):
    def __init__(self, shape):
      super().__init__()
      """
      Reshape 3-dimensional Tensor to 1-Dimensional Tensor (Vector)
      """
      self.shape = shape, # Tuple

    def forward(self, x):
      return x.view(*self.shape)

class Discriminator(nn.Module):
    def __init__(self):
      super().__init__()
      """
      Create structure of NN
      """
      
      # Neural Network layers
      self.model = nn.Sequential(
          View(218*178*3),
          nn.Linear(3*218*178, 100),
          nn.LeakyReLU(),
          nn.LayerNorm(100),
          nn.Linear(100, 1),
          nn.Sigmoid()
          )

      # Binary cross entropy loss
      self.loss_function = nn.BCELoss()

      # Adam optimiser
      self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

      # Timestap of loss for ploting progress
      self.counter = 0
      self.progress = []
    
    
    def forward(self, inputs):
      """
      Pass through NN and get its answer
      """
      return self.model(inputs)
    
    
    def train(self, inputs, targets):
      """
      Train NN; Take tensor of image with label identificator of image;
      Pass through NN; get loss/cost function and backpropagate NN
      to tweak weights (layers)
      """
      outputs = self.forward(inputs)
      loss = self.loss_function(outputs, targets)

      # For timestamp and plotting
      self.counter += 1
      if (self.counter % 1  == 0):
          self.progress.append(loss.item())
      if (self.counter % 1000 == 0):
          print("counter = ", self.counter)

      # Backpropagation -> zero gradients, perform a backward pass, update weights
      self.optimiser.zero_grad()
      loss.backward()
      self.optimiser.step()
    
    
    def plot_progress(self):
      """
      Plot loss of NN for every image it was trained
      """
      df = pd.DataFrame(self.progress, columns=['loss'])
      df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True,
                yticks=(0, 0.25, 0.5, 1.0, 5.0), title="Discriminator Loss")

class Generator(nn.Module):
    
    def __init__(self):
      super().__init__()
      """
      Create structure of NN
      """
      self.model = nn.Sequential(
          nn.Linear(100, 3*10*10),
          nn.LeakyReLU(),
          nn.LayerNorm(3*10*10),
          nn.Linear(3*10*10, 3*218*178),
          nn.Sigmoid(),
          View((218,178,3))
          )

      # No loss function; will use one from discriminator to calculate error

      self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

      # counter and accumulator for progress
      self.counter = 0
      self.progress = []

    def forward(self, inputs):        
      """
      Pass through NN and get its answer
      """
      return self.model(inputs)
    
    
    def train(self, D, inputs, targets):
      """
      Train NN; Take tensor of image with label identificator of image;
      Pass through NN; get loss/cost function and backpropagate NN
      to tweak weights (layers)
      """
      g_output = self.forward(inputs)
      d_output = D.forward(g_output)
      
      loss = D.loss_function(d_output, targets)

      self.counter += 1
      if (self.counter % 1 == 0):
          self.progress.append(loss.item())

      # Backpropagation
      self.optimiser.zero_grad()
      loss.backward()
      self.optimiser.step()
    
    def plot_progress(self):
      """
      Plot loss of NN for every image it was trained
      """
      df = pd.DataFrame(self.progress, columns=['loss'])
      df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True,
              yticks=(0, 0.25, 0.5, 1.0, 5.0), title="Generator Loss")