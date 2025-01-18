import numpy as np
import matplotlib.pyplot as plt

ei = np.load("entropies_ID.npy")
eo = np.load("entropy_OOD.npy")

mi = np.mean(ei)
mo = np.mean(eo)

print("Mean entropy ID:", mi)
print("Mean entropy OOD:", mo)

plt.boxplot([ei, eo])
plt.xticks([1, 2], ["ID", "OOD"])
plt.ylabel("Entropy")
plt.title("Entropy of ID and OOD samples")
plt.savefig("analysis_data/entropy.png")
