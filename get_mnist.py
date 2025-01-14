import numpy as np
import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train_resized = np.array([resize(image, [32, 32]).numpy() for image in x_train])
x_test_resized = np.array([resize(image, [32, 32]).numpy() for image in x_test])

x_train = x_train_resized
x_test = x_test_resized
del x_train_resized, x_test_resized

x_train = x_train.squeeze(-1)
x_test = x_test.squeeze(-1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

n_shards = 10
shard_size = len(x_train) // n_shards

for i in range(n_shards):
    end_idx = min((i + 1) * shard_size, len(x_train))
    x_train_shard = x_train[i * shard_size:end_idx]
    y_train_shard = y_train[i * shard_size:end_idx]
    np.savez_compressed(f'mnist/x_train/train_{i}.npz', x_train=x_train_shard, y_train=y_train_shard)
    
np.savez_compressed('mnist/test.npz', x_test=x_test, y_test=y_test)