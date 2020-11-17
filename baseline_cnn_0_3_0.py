import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import pandas as pd
import torchvision
from torchsummary import summary
from torchvision import transforms, utils
import os


class loader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        img = Image.open(img_name)
        prev_coords = torch.FloatTensor(self.frame.iloc[idx, 1:5].to_list())
        out_coord = torch.FloatTensor(self.frame.iloc[idx, 5:].to_list())

        if self.transform:
            img = self.transform(img)

        return (img, prev_coords), out_coord


class Nnet(nn.Module):
    def __init__(self):
        super(Nnet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 21, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 20, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 18, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(18),
            nn.ReLU(inplace=True),
            nn.Conv2d(18, 15, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace=True),
            nn.Conv2d(15, 10, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 5, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1200, 200),
            nn.ReLU(inplace=True)
        )
        self.reset = nn.Sequential(
            nn.Linear(204, 200),
            nn.ReLU(inplace=True)
        )

        self.rnn = nn.RNN(200, 200, 2, nonlinearity='relu')

        self.fc2 = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
            # nn.Softmax(dim=1)
        )

    def num_flat_features(self, inputs):

        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

    def forward(self, input, bb_coords=None, prev_out=None):
        x = self.main(input.view(-1, 3, 720, 1280))
        x = self.fc1(x.view(-1, self.num_flat_features(x)))

        if not bb_coords is None:
            x = torch.hstack((x, bb_coords.view(-1, 4)))
            x = self.reset(x)

        x, h = self.rnn(x.view(-1, 1, 200), prev_out)

        return self.fc2(x.view(-1, 200)), h
