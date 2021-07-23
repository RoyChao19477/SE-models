# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import h5py
import sys

# check arguments
if len(sys.argv) < 2:
    print("Usage: python v2_py39_torch19/v2_train_DNN.py data.h5")
    sys.exit(1)

# check gpu()
use_gpu = torch.cuda.is_available()

# Parameter
data = "20160508"
FRAMESIZE = 512
FRAMEWIDTH = 2
FBIN = FRAMESIZE//2+1
input_dim = FBIN*(FRAMEWIDTH*2+1)

BATCHSIZE = 200
EPOCH = 30

# load data.h5
data_path = "src/dataset/" + sys.argv[1]
print ("data loading")
"""
X_train = HDF5Matrix(data_path, "trnoisy")
y_train = HDF5Matrix(data_path, "trclean")
"""
h5_file = h5py.File(data_path)
X_train = torch.tensor(h5_file["trnoisy"])
y_train = torch.tensor(h5_file["trclean"])

print(X_train)
print(y_train)

dataset = Data.TensorDataset(X_train, y_train)     # TensorDataset(inputs, targets)
loader = Data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False)

print("model building...")

model = torch.nn.Sequential(
        torch.nn.Linear(1285, 2048),
        torch.nn.ELU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ELU(),
        torch.nn.Linear(2048, 257)
        )

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0002)
loss_fn = torch.nn.MSELoss()

for t in range(EPOCH):
    for batch_ndx, batch in enumerate(loader):
        # Forward
        if use_gpu:
            x = batch[0].cuda()
            y = batch[1].cuda()
        else:
            x = batch[0]
            y = batch[1]


        optimizer.zero_grad()
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()


# linear_layer = model[0]
# print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

torch.save(model, "src/model/DDAE.pt")

"""
class DDAE(nn.Module):
    def __init__(self):
        super(DDAE, self).__init__()
        self.layer_1 = nn.Linear(1285, 2048)
        self.

    model = torch.nn.Sequential(
            torch.nn.Linear(1285, 2048)
            )

    def forward(self, 
"""
