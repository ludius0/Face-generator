import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import networks

# functions to generate random data
def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data

def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data

# Functions to plot
def plot_loss():
    D.plot_progress()
    G.plot_progress()

def plot_results(G):
    f, ax = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(100))
            img = output.detach().cpu().numpy()
            ax[i, j].imshow(img, interpolation="none", cmap="Blues")
    plt.show()

# CUDA
if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  print("using cuda:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Networks
Dataset = networks.CelebADataset("Celeba/200_img_align_celeba.h5py", device="cpu")
D = networks.Discriminator().to(device)
G = networks.Generator().to(device)

# Train
epochs = 1
for e in range(epochs):
    print(f"Training in {e+1} / {epochs} epochs...")
    for index, img_tensor in tqdm(zip(range(len(Dataset)), Dataset), total=len(Dataset)):
        # Train Discriminator -> Real data
        D.train(img_tensor, torch.FloatTensor([1.0]))
        # Train Discriminator -> Fake data
        D.train(G.forward(generate_random_seed(100)), torch.FloatTensor([0.0]))
        # Train Generator
        G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))

plot_loss()
plot_results(G)