import numpy as np
import matplotlib.pyplot as plt

from datasets import generate_dataset

class ActivationFunction(object):
    """
    Represents the activation function for a particular unit.
    Currently implements Hyperbolic Tangent and Sigmoid functions.
    To call the function on input vector x, simply call function() member. 
    Derivative can be called similarly with derivative().
    """
    # Static types
    TANH = 1
    SIGMOID = 2

    def __init__(self, act_type):
        self.act_type = act_type

    def function(self, x):
        """
        Returns the evaluated activation function on the input vector x.
        """
        if self.act_type == ActivationFunction.TANH:
            return self._tanh(x)
        elif self.act_type == ActivationFunction.SIGMOID:
            return self._sigmoid(x)
        else:
            raise Exception("Unrecognised function.")

    def derivative(self, x):
        """
        Returns the evaluated derivative of the activation function on the input
        vector x.
        """
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
    """
    Given a set of weight matrices (1 for the synaptic connections between each
    layer), and a new input vector/ target vector pair D and learning rate eta,
    updates the weight matrices to reflect evidence from the new training sample.

    Note, the number of hidden units is not necessary as this is implicit in the
    dimensionality of the weight matrices.
    """
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
    """
    Given a set of weight matrices (1 for the synaptic connections between each
    layer), and a new input vector/ target vector pair D, evaluates if the resulting
    MLP output corresponds to that of the target vector.

    Takes the unit activated the most in the output layer as the network's output.
    """
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

    # Compare the target class to the output class
    target_class = target.transpose().argmax(axis=1) + 1
    output_class = l2.argmax(axis=1) + 1
    if target_class != output_class:
        return 1
    else:
        return 0


def parse_data_point(x):
    """ 
    Takes a Dataset data point x and parses into the required format for the MLP.

    Outputs an input vector, with additional bias column (always 1), and a target 
    vector representing the output class e.g. output class = 3, produces target 
    vector [0,0,1,0].
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
    # Get the data sets
    train_set, valid_set, test_set = generate_dataset(False)

    # Number of epochs
    number_of_epochs = 50

    # Number of hidden layers
    h = 4   
    
    # Network topology:
    # 2 input units + 1 bias = 3 units
    # h hidden units
    # 4 output units (1 per class)

    # Make the calculations deterministic
    np.random.seed(1)

    # Initialise weights randomly
    # weight matrix 1 = 3 * h
    # weight matrix 2 = h * 4
    weight_matrix_1 = np.random.uniform(low = -0.5, high = 0.5, size = 3 * h).reshape((3,h))
    weight_matrix_2 = np.random.uniform(low = -0.5, high = 0.5, size = 4 * h).reshape((h,4))

    # For each sample in the training set, train the MLP. Repeat every epoch.
    weights = [weight_matrix_1, weight_matrix_2]
    for n in xrange(number_of_epochs):
        for x in train_set:
            weights = train_mlp(weights, x, eta = 0.04, h = 5)

    # Now count the number of mis-classifications based on the test set
    error = 0
    for x in test_set:
        error += evaluate_mlp(weights, x, 999)
    error = float(error) / float(len(test_set))
    print "Total Error Percentage: {}%".format(str(error * 100))
