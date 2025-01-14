import copy
import numpy as np
from tqdm import tqdm
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


gpu_idx = 6
device = torch.device(f"cuda:{gpu_idx}")

n_shards = 10
x_train = []
y_train = []

for i in range(n_shards):
    x_train.append(np.load(f'cifar10/x_train/train_{i}.npz')['x_train'])
    y_train.append(np.load(f'cifar10/x_train/train_{i}.npz')['y_train'])
    
x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()

x_train = x_train.unsqueeze(1)
x_train = x_train.to(device)
y_train = y_train.to(device)

x_test = np.load('cifar10/test.npz')['x_test']
x_test = torch.tensor(x_test).float()
x_test = x_test.unsqueeze(1)
x_test = x_test.to(device)

x_test = x_test[:1000]


adversaries = 5
optim_steps = 15
adv_iterations = 10
lr_adv = 0.001
gamma = 0
c_0 = 1e-0
eta = 3
ce = nn.CrossEntropyLoss()


g_cpu = torch.Generator(device=device)
n_classes = 10
weight_decay = 1e-3
batch_size = 32

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
model.load_state_dict(torch.load('lenet_cifar.pt', map_location=device, weights_only=True))
model.to(device)

with torch.no_grad():
    train_net_pred = model(x_train)
    train_net_ce = ce(train_net_pred, y_train)

p = np.zeros((len(x_test), 10))

# Adversarial Attack
for a, l in product(range(adversaries), range(n_classes)):

    y_test = torch.LongTensor([l]).to(device=device)

    for i in tqdm(
        range(len(x_test)),
        desc=f"inference {a * n_classes + l + 1} / {adversaries * n_classes}",
    ):

        adversarial_network = copy.deepcopy(model)
        adversarial_network.to(device)
        adversarial_network.train()

        opt = optim.Adam(params=adversarial_network.parameters(), lr=lr_adv, weight_decay=weight_decay)
        preds = list()

        c = c_0
        for ad_i in range(adv_iterations):
            for op_s in range(optim_steps):
                idx = torch.randperm(x_train.size(0))[:batch_size]
                train_adv_pred = adversarial_network.forward(x_train[idx])
                penalty = ce(train_adv_pred, y_train[idx])

                test_adv_pred = adversarial_network.forward(x_test[i].unsqueeze(0))
                objective = ce(test_adv_pred, y_test)

                loss = objective + c * (penalty - train_net_ce - gamma)

                with torch.no_grad():
                    adversarial_network.eval()
                    preds.append(
                        torch.softmax(adversarial_network.forward(x_test[i].unsqueeze(0)), dim=1)
                        .cpu()
                        .numpy()
                    )
                    adversarial_network.train()

                opt.zero_grad()
                loss.backward()
                opt.step()

            # update parameter
            c *= eta

        p[i][np.argmax(preds[149][0])] += 1

entropies = []

p = p / 50.0

for i in range(len(p)):
    e = np.sum((-1) * p[i] * np.log(p[i] + 0.0000001))
    entropies.append(e)

print("Mean entropy:", np.mean(np.array(entropies)))

np.save("entropies_ID.npy", entropies)