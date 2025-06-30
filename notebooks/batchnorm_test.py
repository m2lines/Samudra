#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os

import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim

device = "cuda"


# In[14]:


class CNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            hidden_channels, output_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # BatchNorm2d
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# In[15]:


input_channels = 3
hidden_channels = 16
output_channels = 3
frame_height = 64
frame_width = 64
batch_size = 1  # Batch size of 1 to simulate autoregressive input
num_epochs = 2
num_future_frames = 5  # Number of future frames to predict


# In[16]:


# Instantiate the model, loss function, and optimizer
model = CNN(input_channels, hidden_channels, output_channels)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())


# In[17]:


# Dummy data
initial_frame = torch.randn(batch_size, input_channels, frame_height, frame_width).to(
    device
)
target_frames = torch.randn(
    batch_size, num_future_frames, output_channels, frame_height, frame_width
).to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    x = initial_frame
    loss = 0
    predictions = []
    for t in range(num_future_frames):
        print("Time step", t)
        # Print BN Stats
        for name, param in model.bn1.named_parameters():
            print(name, param.shape, param)
        print()
        for name, param in model.bn1.named_buffers():
            print(name, param.shape, param)

        print()

        x = model(x)  # Predict next frame
        predictions.append(x.unsqueeze(1))  # Collect predictions
        target = target_frames[:, t, :, :, :]
        loss += criterion(x, target)  # Accumulate loss
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


# ### Where is the Error coming from? - torch.nn.parallel.DistributedDataParallel
#
# https://github.com/pytorch/pytorch/issues/66504#issuecomment-1007806934

# In[18]:


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "45678"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
world_size = 1
rank = 0


# In[23]:


torch.distributed.destroy_process_group()


# In[24]:


torch.distributed.init_process_group(
    backend="nccl", init_method="env://", rank=rank, world_size=world_size
)
model = CNN(input_channels, hidden_channels, output_channels)
model = model.to(device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[rank],
    broadcast_buffers=False,  # this is important
)


# In[25]:


model.train()

# Dummy data
initial_frame = torch.randn(batch_size, input_channels, frame_height, frame_width).to(
    device
)
target_frames = torch.randn(
    batch_size, num_future_frames, output_channels, frame_height, frame_width
).to(device)

optimizer.zero_grad()
x = initial_frame
loss = 0
predictions = []

for t in range(num_future_frames):
    x = model(
        x
    )  # Predict next frame # Calls inplace _copy on buffer right before forward
    predictions.append(x.unsqueeze(1))  # Collect predictions
    target = target_frames[:, t, :, :, :]
    loss += criterion(x, target)
out = loss.mean()
out.backward()  # may throw error!
print("Passed!")


# ### my hacky fix for synchronized buffers

# In[26]:


class CNN_with_loop(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(CNN_with_loop, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            hidden_channels, output_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # BatchNorm2d
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

    def forward(self, x, num_future_frames, target_frames, criterion):  # Hack
        predictions = []
        loss = 0.0
        for t in range(num_future_frames):
            x = self.forward_once(x)  # Predict next frame
            predictions.append(x.unsqueeze(1))
            target = target_frames[:, t, :, :, :]
            loss += criterion(x, target)
        return loss, predictions


# In[27]:


torch.distributed.destroy_process_group()


# In[28]:


torch.distributed.init_process_group(
    backend="nccl", init_method="env://", rank=rank, world_size=world_size
)

model = CNN_with_loop(input_channels, hidden_channels, output_channels)
model = model.to(device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[rank],
    # broadcast_buffers=False # not important anymore
)


# In[29]:


model.train()

# Dummy data
initial_frame = torch.randn(batch_size, input_channels, frame_height, frame_width).to(
    device
)
target_frames = torch.randn(
    batch_size, num_future_frames, output_channels, frame_height, frame_width
).to(device)

optimizer.zero_grad()
x = initial_frame
loss = 0
predictions = []

for t in range(num_future_frames):
    # Calls inplace _copy on buffer right before forward, now no recursive inplace _copy on buffer
    loss, predictions = model(x, num_future_frames, target_frames, criterion)

out = loss.mean()
out.backward()  # may throw error!
print("Passed!")


# Relevant links -
# * https://github.com/pytorch/pytorch/issues/66504#issuecomment-1007806934
# * https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/comm.cpp#L39
# * https://github.com/pytorch/pytorch/blob/main/torch/distributed/utils.py#L334

# In[ ]:


# In[ ]:


#
