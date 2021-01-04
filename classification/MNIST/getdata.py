import os

import numpy as np
from scipy.io import loadmat

from sklearn.datasets import fetch_openml

def get_data1():
    return os.getcwd()

def get_data():
    def sort_by_target(mnist):
        reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
        reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
        mnist.data[:60000] = mnist.data[reorder_train]
        mnist.target[:60000] = mnist.target[reorder_train]
        mnist.data[60000:] = mnist.data[reorder_test + 60000]
        mnist.target[60000:] = mnist.target[reorder_test + 60000]

    # mnist = fetch_openml('mnist_784', version=1, cache=False)
    # mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
    # sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset

    # with open(mnist_path, "wb") as f:
    #     content = response.read()
    #     f.write(content)

    mnist_path = 'data\mnist\mnist-original.mat'
    mnist_path = os.path.join("..", "..", "data", "mnist", "mnist-original.mat")

    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }

    return mnist

    (
        # Альтернативный способ загрузки
        # from six.moves import urllib
        # from scipy.io import loadmat
        # mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
        # mnist_path = "./mnist-original.mat"
        # response = urllib.request.urlopen(mnist_alternative_url)
        # with open(mnist_path, "wb") as f:
        #     content = response.read()
        #     f.write(content)
        # mnist_raw = loadmat(mnist_path)
        # mnist = {
        #     "data": mnist_raw["data"].T,
        #     "target": mnist_raw["label"][0],
        #     "COL_NAMES": ["label", "data"],
        #     "DESCR": "mldata.org dataset: mnist-original",
        # }
    )
