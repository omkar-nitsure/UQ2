import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import get_ood
from duq_model import CNN_DUQ

gpu_idx = 6
# device = torch.device(f"cuda:{gpu_idx}")
device = torch.device("cpu")

uv_id = np.load("uv_id.npy")
mi = np.mean(uv_id)

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

x_ood = get_ood()

x_ood = np.expand_dims(x_ood, 1)
x_ood = torch.tensor(x_ood).float()

model.eval()
with torch.no_grad():
    x_ood = x_ood.to(device)
    y_pred, dist = model(x_ood)
    
dist = dist.cpu().numpy()
dist = np.min(dist, axis=1)

mo = np.mean(dist)

print("Mean uncertainty value for OOD:", mo)
print("Mean uncertainty value for ID:", mi)

plt.boxplot([uv_id, dist])
plt.xticks([1, 2], ["ID", "OOD"])
plt.ylabel("uncertainty value")
plt.title("uncertainty value distribution for ID and OOD samples")
plt.savefig("uncertainty_vals.png")
