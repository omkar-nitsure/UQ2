import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

gpu_idx = 7
device = torch.device(f"cuda:{gpu_idx}")

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

def predict(model, x):
    
    model.eval()
    
    with torch.no_grad():
        x = x.to(device)
        y_pred = model(x)
        
    return y_pred

x_test, y_test = np.load('cifar10/test.npz')['x_test'], np.load('cifar10/test.npz')['y_test']

x_test = torch.tensor(x_test).float()
x_test = x_test.unsqueeze(1)
y_test = torch.tensor(y_test).float()

x_test = x_test[:1000]
y_test = y_test[:1000]

y_pred = predict(model, x_test)
y_pred = y_pred.cpu().numpy()
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

ei = np.load("entropies_ID.npy")

ei_idx = np.argsort(-ei)

c = 0
n = len(x_test)
accs = []

for i in range(len(x_test)):
    if y_test[i] == y_pred[i]:
        c += 1
        

for i in range(len(x_test) - 1):
    
    if(y_test[ei_idx[i]] == y_pred[ei_idx[i]]):
        c -= 1
        n -= 1
    else:
        n -= 1
        
    accs.append((100 * c) / n)

acc_plot = []
for i in range(len(accs)):
    if(i % 10 == 0):
        acc_plot.append(accs[i])
        
acc_plot = np.array(acc_plot)

plt.plot(np.arange(0, len(acc_plot)), acc_plot)
plt.xlabel("percent of samples removed")
plt.ylabel("accuracy")
plt.title("Accuracy vs percent of samples removed")
plt.savefig("acc_plot.png")