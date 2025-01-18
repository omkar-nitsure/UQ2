import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gpu_idx = 3
device = torch.device(f"cuda:{gpu_idx}")

n_epochs = 250
max_lr = 6e-4
min_lr = max_lr * 0.1

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):

        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        
        return output

model = LeNet()
model.to(device)

n_shards = 10
x = []
y = []

for i in range(n_shards):
    x.append(np.load(f'../cifar10/x_train/train_{i}.npz')['x_train'])
    y.append(np.load(f'../cifar10/x_train/train_{i}.npz')['y_train'])

x = np.concatenate(x)
y = np.concatenate(y)

x = torch.tensor(x).float()
y = torch.tensor(y).float()

def train(model, dataloader, optimizer, scheduler, criterion, epochs=10):
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
            
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")
        
    model.eval()
        
    return model
            

x = x.unsqueeze(1)      

train_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_lr)

model = train(model, train_loader, optimizer, scheduler, F.cross_entropy, n_epochs)

torch.save(model.state_dict(), "lenet_cifar.pt")