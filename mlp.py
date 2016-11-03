import numpy as np
import matplotlib.pyplot as plt

from datasets import generate_dataset, Dataset


class ActivationFunction(object):

    TANH = 1
    SIGMOID = 2

    def __init__(self, act_type):
        self.act_type = act_type

    def function(self, x):
        if self.act_type == ActivationFunction.TANH:
            return self._tanh(x)
        elif self.act_type == ActivationFunction.SIGMOID:
            return self._sigmoid(x)
        else:
            raise Exception("Unrecognised function.")

    def derivative(self, x):
        if self.act_type == ActivationFunction.TANH:
            return self._tanh_deriv(x)
        elif self.act_type == ActivationFunction.SIGMOID:
            return self._sigmoid_deriv(x)
        else:
            raise Exception("Unrecognised function.")

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_deriv(x):
        return 1.0 - np.tanh(x)**2

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x)*(1-self._sigmoid(x))


def train_mlp(w, D, eta = 0.2, h = 5):
    # Network topology
    # 2 input units + 1 bias
    # h hidden units
    # 4 output units

    # Get the weight matrices
    weight_matrix_1, weight_matrix_2 = w[0], w[1]

    # Get the input vector and target output vector
    l0_in, target = parse_data_point(D)

    # For now, just use same sigmoid function for all units (easily changed below)
    activation_function = ActivationFunction(ActivationFunction.SIGMOID)

    # Do the forward pass
    l0 = l0_in
    l1 = activation_function.function(np.dot(l0, weight_matrix_1))
    l2 = activation_function.function(np.dot(l1, weight_matrix_2))

    # Do the backwards pass
    # Check the error and adjust for layer 2
    l2_error = target.transpose() - l2
    l2_delta = l2_error * activation_function.derivative(l2)
    # Check the error and adjust for layer 1
    l1_error = l2_delta.dot(weight_matrix_2.transpose())
    l1_delta = l1_error * activation_function.derivative(l1)

    # Update the weights accordingly and proportionally to the learning rate
    weight_matrix_2 += eta * (l1.transpose().dot(l2_delta))
    weight_matrix_1 += eta * (l0.transpose().dot(l1_delta))

    return [weight_matrix_1, weight_matrix_2]


def evaluate_mlp(w, D, h):
    # Get the weights
    weight_matrix_1, weight_matrix_2 = w[0], w[1]

    # Get the input vector and target output vector
    l0_in, target = parse_data_point(D)

    # For now, just use same sigmoid function for all units (easily changed below)
    activation_function = ActivationFunction(ActivationFunction.SIGMOID)

    # Do the forward pass
    l0 = l0_in
    l1 = activation_function.function(np.dot(l0, weight_matrix_1))
    l2 = activation_function.function(np.dot(l1, weight_matrix_2))
    target_class = target.transpose().argmax(axis=1) + 1
    output_class = l2.argmax(axis=1) + 1
    if target_class != output_class:
        return 1
    else:
        return 0
    #if(target.transpose() - l2) != 0 :


def parse_data_point(x):
    """ 
    Normalises the data, so that classes lie between 0 and 1. Adds an
    additional input value - the bias, always 1.
    """
   # Add the bias to the input vector
    input_vector = np.zeros((3,1))
    input_vector[:2,0] = x[:2]
    input_vector[2,0] = 1
    input_vector = input_vector.transpose()

    # Create target vector - data provides as float, so convert to int and
    # use as index for output activations
    target_value = x[2]
    target_vector = np.zeros((4,1))
    target_vector[int(target_value) - 1] = 1

    return input_vector, target_vector


if __name__ == "__main__":
    train_set, valid_set, test_set = generate_dataset(False)

    number_of_epochs = 50
    h = 4   
    np.random.seed(1)
    # Initialise weights randomly
    # weight matrix 1 = 3 * h
    # weight matrix 2 = h * 4
    weight_matrix_1 = np.random.uniform(low = -0.5, high = 0.5, size = 3 * h).reshape((3,h))
    weight_matrix_2 = np.random.uniform(low = -0.5, high = 0.5, size = 4 * h).reshape((h,4))

    weights = [weight_matrix_1, weight_matrix_2]
    for n in xrange(number_of_epochs):
        for x in train_set:
            weights = train_mlp(weights, x, eta = 0.04, h = 5)

    error = 0
    for x in train_set:
        error += evaluate_mlp(weights, x, 999)
    error = float(error)/float(len(train_set))
    print "Total Errors: {}%".format(str(error * 100))
    # data_point = np.zeros((3,1))
    # data_point[:2,0] = x[:2]
    # data_point[2,0] = 1
    # print data_point.transpose()
    # print x[2]
    # target_vector = np.zeros((4,1))
    # target_vector[int(x[2]) - 1] = 1
    # print target_vector
