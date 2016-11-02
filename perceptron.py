import numpy as np
from datasets import generate_dataset

# PERCEPTRON
def train_perceptron(w, x, eta):

    data_point = np.zeros((3,1))
    print data_point.shape
    data_point[:2,0] = x[:2]
    data_point[2,0] = 1
    target = x[2] - 1
    threshold_func = lambda x: 0 if x < 0 else 1
    summed_weights = np.dot(w, data_point)
    error = target - threshold_func(summed_weights)
    w += eta * error * x
    return w

def evaluate_perceptron(w, D):
    error_sum = 0
    for data_point in D:
        if data_point[2] == 1 or data_point[2] == 2:
            
            summed_weights = np.dot(w, data_point)
            error = target - threshold_func(summed_weights)
if __name__ == "__main__":
    train, valid, test = generate_dataset(False)



    # initalise weights
    w = np.random.uniform(low = -0.5, high = 0.5, size = 3)
    x = np.array([3.16965834,1.64847082,2.])
    print x[2]

    eta = 0.2
    for data_point in train:
        if data_point[2] == 1 or data_point[2] == 2:
            w = train_perceptron(w, data_point, eta)
            print w

