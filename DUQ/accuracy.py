import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from datasets import get_id
from duq_model import CNN_DUQ

gpu_idx = 6
# device = torch.device(f"cuda:{gpu_idx}")
device = torch.device("cpu")


def prepare_datasets(x_test, y_test):

    x_test = np.expand_dims(x_test, 1)
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()
    
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    train = data.TensorDataset(x_test, y_test)

    return data.DataLoader(train, batch_size=64, shuffle=False)


def accuracy(model, dataloader):

    with torch.no_grad():
        c, t = 0, 0
        r, d = [], []
        for x, y in dataloader:
            x = x.to(device)
            y = np.argmax(y.numpy(), axis=1)

            pred, dist = model(x)

            pred = pred.cpu().numpy()
            dist = dist.cpu().numpy()

            pred = np.argmax(pred, axis=1)
            
            
            c += np.sum(pred == y)
            t += len(y)
            r.append((pred == y).astype(int))
            d.append(np.min(dist, axis=1))
            
        r = np.concatenate(r)
        d = np.concatenate(d)
        
        r = r.flatten()
        d = d.flatten()

    return  c, t, r, d

num_classes = 10
embedding_size = 84
learnable_length_scale = False
gamma = 0.999
length_scale = 0.05

model = CNN_DUQ(
    num_classes,
    embedding_size,
    learnable_length_scale,
    length_scale,
    gamma,
)

model.load_state_dict(torch.load(f"models/LeNet_0.05.pt", map_location=device, weights_only=True))
model.to(device)

x_test, y_test = get_id()

dl = prepare_datasets(x_test, y_test)

acc, tot, results, dists = accuracy(model, dl)

d, id = zip(*sorted(zip(-dists, np.arange(1000))))

accs = []

for i in range(tot - 1):
    
    if(results[id[i]]):
        acc -= 1
        tot -= 1
    else:
        tot -= 1
        
    accs.append((100 * acc) / tot)
    
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

np.save("uv_id.npy", dists)