import numpy as np
import matplotlib.pyplot as plt

from datasets import generate_perceptron_dataset, Dataset

# Perceptron functions
def train_perceptron(w, x, eta):
    """ Adjusts the weight matrix for an individual data point. """
    # Add bias and normalise
    data_point, target = parse_data_point(x)

    # Train the perceptron for this data point
    summed_weights = np.dot(w, data_point)
    error = target - threshold_function(summed_weights)
    w += eta * error * x
    return w

def evaluate_perceptron(w, D):
    """ 
    Returns the number of errors encountered, using the given weight matrix
    for a given data set. 
    """
    error_sum = 0
    for data_point in D:
        # Add bias and normalise
        data_point, target = parse_data_point(data_point)

        summed_weights = np.dot(w, data_point)
        error = target - threshold_function(summed_weights)
        if(error != 0):
            error_sum += 1
    return error_sum

def parse_data_point(x):
    """ 
    Normalises the data, so that classes lie between 0 and 1. Adds an
    additional input value - the bias, always 1.
    """
   # Add the bias
    data_point = np.zeros((3,1))
    data_point[:2,0] = x[:2]
    data_point[2,0] = 1

    # Normalise the target values (classes in this case are 1,2 so easy - subtract 1)
    target = x[2] - 1
    return data_point, target

def threshold_function(x):
    """ Step threshold function. """
    return 0 if x < 0 else 1

def show_decision_boundary_plot(w):
    """
    Shows the decision boundary orthogonal to the weight vector. For visualisation,
    the original data points are recalculated and displayed, with their labels
    represented by colour.
    """
    # Recreate the original datasets for visualisation
    class_one = Dataset.create_uniform_distribution(x_scale = 3, 
                                                    y_scale = 3, 
                                                    x_shift = 2, 
                                                    y_shift = 1,
                                                    label = 1).rotate(75)

    class_two = Dataset.create_uniform_distribution(x_scale = 2, 
                                                    y_scale = 4, 
                                                    x_shift = 1, 
                                                    y_shift = -5,
                                                    label = 2).rotate(75)
    plt.scatter(class_one.x1, class_one.x2, color='red')
    plt.scatter(class_two.x1, class_two.x2, color='blue')

    # Calculate the equation for the decision boundary
    x1 = -(w[2]/w[0])
    x2 = 0
    y1 = 0
    y2 = -(w[2]/w[1])
    grad = (y1 - y2) / (x1 - x2)

    # Plot the decision boundary
    y_list = []
    x_list = [x - 4 for x in xrange(11)]
    for x in x_list:
        y_list.append(grad * x + y1)
    plt.plot(x_list, y_list, linestyle='-')

    plt.show()

def train_and_evaluate(eta = 0.6, epochs = 200):
    """
    Trains the percepron using the given learning rate (eta) for the given number
    of epochs.
    """

    # Get the data
    train_set, valid_set, test_set = generate_perceptron_dataset()

    # Initalise weights
    np.random.seed(12)
    w = np.random.uniform(low = -0.5, high = 0.5, size = 3)

    # Keep training until the errors in the test set are below the threshold
    acceptable_error_threshold = 5

    # Just a modulo value, for how often to plot error
    error_plot_interval = 1

    # Array to store error percentages per iteration
    error_values = []
    # Check against the test set for errors, keep training until error
    # number below threshold
    while evaluate_perceptron(w, test_set) > acceptable_error_threshold:
        # Train the perceptron for all train data
        for n in xrange(epochs):
            i = 0
            for data_point in train_set:
                w = train_perceptron(w, data_point, eta)
                if i % error_plot_interval == 0 and n == 0:
                    # Append the percentage error
                    error_values.append(float(evaluate_perceptron(w, test_set)) / float(test_set.shape[0]))
    samples = [x * error_plot_interval for x in range(len(error_values))]
    return samples, error_values, w

if __name__ == "__main__":
    # Try the different learning rates - currently going from 0.01 to 0.61
    for eta in xrange(1, 61, 10):
        eta *= 0.01 #xrange only allows for integers, so scale down
        samples, error_values, w = train_and_evaluate(eta = eta)

        # Accuracy is 1 minus percentage of errors in final iteration
        print "Learning Rate: {}, Accuracy: {}%".format(str(eta), str((1 - error_values[-1])*100))

        name = "ETA-"+str(eta)
        plt.plot(samples, error_values, linestyle='-', label=name)

    # Show learning rates
    plt.legend()
    plt.show()

    show_decision_boundary = True
    if(show_decision_boundary):
        show_decision_boundary_plot(w)
