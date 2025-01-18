import numpy as np

def get_train():

    n_shards = 10
    x = []
    y = []

    for i in range(n_shards):
        x.append(np.load(f'../cifar10/x_train/train_{i}.npz')['x_train'])
        y.append(np.load(f'../cifar10/x_train/train_{i}.npz')['y_train'])

    x = np.concatenate(x)
    y = np.concatenate(y)

    return x, y


def get_id():
    x_test, y_test = np.load('../cifar10/test.npz')['x_test'], np.load('../cifar10/test.npz')['y_test']

    return x_test, y_test


def get_ood():
    x_ood = np.load('../mnist/test.npz')['x_test']

    return x_ood


def get_noisy_id(var):

    x_test, y_test = np.load('../cifar10/test.npz')['x_test'], np.load('../cifar10/test.npz')['y_test']
    x_test = x_test + np.random.normal(0, var, x_test.shape)

    return x_test, y_test