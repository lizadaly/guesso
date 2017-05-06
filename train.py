# Train a neural network on fine art images from CC0-licensed datasets on the
# image and its best-known date of origin.
#
# Stopped at:
# INFO:root:last guess/target: 1763/1781 (18) / avg loss 0.60
# INFO:root:Validation loss: 4.79
# INFO:root:Starting epoch 30

import os
import logging

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import pandas as pd
import numpy as np
from PIL import Image

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Expects a CSV data file like:

# id,year,path
# AK-BR-JAN-1,1200,/data/rijksmuseum-images/1/AK-BR-JAN-1.jpeg
# AK-MAK-1154,1800,/data/rijksmuseum-images/1/AK-MAK-1154.jpeg
# AK-MAK-116,1200,/data/rijksmuseum-images/1/AK-MAK-116.jpeg
# 751139,1655,/data/met-images/751/751139.jpg
# 751140,1655,/data/met-images/751/751140.jpg
# 751141,1655,/data/met-images/751/751141.jpg

# You'll have to download the images yourself from the provided CSV files in the data/ directory
YEAR_DATA = 'FIXME.csv'

BATCH_SIZE = 32
NUM_WORKERS = 5
NUM_EPOCHS = 10000
NUM_CLASSES = 1  # We're only evaluating one feature: the year of origin
MODEL_NAME = "guesso-resnet-152.pth"


class ImageSet(data.Dataset):
    def __init__(self, imagedata, transform=None, train=True):
        self.imagedata = imagedata
        self.train = train  # training set or test set
        self.transform = transform

    def __getitem__(self, index):
        row = self.imagedata.iloc[index]
        # Diving the year values (e.g. 1463) by 100 gives us numbers that the network
        # can better deal with, and more sensible loss values
        year = float(row['year']) / 100.0
        path = row['path']
        try:
            # Not all of the metadata is good, so some images may not load
            img = Image.open(path).convert('RGB')
        except OSError as e:
            os.remove(path)
            log.warn(e)
            return None, None
        if self.transform is not None:
            img = self.transform(img)
        return img, year

    def __len__(self):
        return len(self.imagedata)


dataset = pd.read_csv(YEAR_DATA)
dataset = dataset.loc[dataset.year >= 1200]
dataset = dataset.loc[dataset.year <= 1930]
dataset = dataset.set_index('id')

# Shuffle entire set and drop any duplicates (which could result in training
# data leaking into the test set)
dataset = dataset.sample(frac=1)  # Shuffles entire set as a side effect
dataset = dataset.drop_duplicates()

# Split into training and validation
msk = np.random.rand(len(dataset)) < 0.9
train = dataset[msk]
test = dataset[~msk]

assert len(train) + len(test) == len(dataset)

train.to_csv("train-after-split.csv", index=False)
test.to_csv("test-after-split.csv", index=False)

log.info("%d train records, %d test", len(train), len(test))

train_loader = torch.utils.data.DataLoader(
    ImageSet(train, transforms.Compose([
        transforms.Scale(300),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    ImageSet(test, transforms.Compose([
        transforms.Scale(300),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)

start_epoch = 0

log.info("Loading model...")

model = models.resnet50(pretrained=True)

# Remove the classification layer from the pretrained network and replace with 1 feature (the year)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.cuda()

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_avg = 0
count = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    log.info("Starting epoch %d", epoch)
    model.train()
    for i, (inp, target) in enumerate(train_loader):
        # The pytorch loader framework will think our year values are int64, but we want
        # floats for MSELoss
        target = torch.Tensor([float(f) for f in target])
        target_var = Variable(target, requires_grad=False).cuda()
        input_var = Variable(inp, requires_grad=False).cuda()
        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg += loss.data[0]
        count += 1

    # Convert the guess values back to integer years for debugging
    last_guess = int(output[0][0].data[0] * 100)
    last_target = int(target[0] * 100)
    log.info("last guess/target: %d/%d (%d) / avg loss %.2f", last_guess, last_target, abs(last_guess - last_target), loss_avg / count)
    torch.save(model.state_dict(), MODEL_NAME)

    model.eval()

    # Run a validation step; loss under 5.0 is pretty good
    val_loss = 0
    for i, (inp, target) in enumerate(test_loader):
        input_var = torch.autograd.Variable(inp, volatile=True).cuda()
        target = torch.Tensor([float(f) for f in target])
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        val_loss += loss.data[0]
    log.info("Validation loss: %.2f", val_loss / BATCH_SIZE)
