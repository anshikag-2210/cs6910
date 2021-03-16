from keras.datasets import fashion_mnist
import numpy as np


def load():
    # load dataset
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
    test_data = normalise(test_data)
    train_data = normalise(train_data)

    validation = 0.1 # 10% for validation
    split = int(len(train_data) * (1 - validation))
    validation_data, validation_labels = train_data[split:], train_labels[split:]

    train_data, train_labels = train_data[:split], train_labels[:split]
    return (train_data , train_labels , validation_data , validation_labels , test_data , test_labels)

def normalise(data):
    normalised_data = ((data.reshape((data.shape[0], 784))).astype('float64'))/255.0
    return normalised_data
