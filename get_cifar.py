import numpy as np
import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

x_train = np.mean(x_train, axis=-1)
x_test = np.mean(x_test, axis=-1)

n_shards = 10
shard_size = len(x_train) // n_shards

for i in range(n_shards):
    end_idx = min((i + 1) * shard_size, len(x_train))
    x_train_shard = x_train[i * shard_size:end_idx]
    y_train_shard = y_train[i * shard_size:end_idx]
    np.savez_compressed(f'cifar10/x_train/train_{i}.npz', x_train=x_train_shard, y_train=y_train_shard)
    
np.savez_compressed('cifar10/test.npz', x_test=x_test, y_test=y_test)