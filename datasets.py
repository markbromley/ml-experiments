import numpy as np
import matplotlib.pyplot as plt

class Dataset(object):

    def __init__(self, data):
        # TODO: Add asserts
        self._data = data

    @staticmethod
    def create_uniform_distribution(x_scale = 1, 
                                    y_scale = 1, 
                                    x_shift = 1, 
                                    y_shift = 1,
                                    label = None, 
                                    data_size = 500):
        # Make the set deterministic
        np.random.seed(1)
        data = np.random.rand(data_size, 3)
        data[:,0] *= x_scale
        data[:,1] *= y_scale
        data[:,0] += x_shift
        data[:,1] += y_shift
        # Add the label
        data[:,2] = label
        data = Dataset(data)
        return data

    @staticmethod
    def create_multivariate_normal_distribution(mean = (0,0), 
                                                covariance_matrix = ((0,0),(0,0)), 
                                                data_size = 500,
                                                label = None):
        # Make the set deterministic
        np.random.seed(1)
        data_points = np.random.multivariate_normal(mean, covariance_matrix, (data_size))
        data = np.zeros((data_size, 3))
        data[:,:2] = data_points
        data[:,2] = label
        return Dataset(data)


    def rotate(self, theta_degrees = 75):
        return_mat = np.zeros_like(self._data)
        rot_mat = self._get_rotation_matrix(theta_degrees)
        return_mat[:,:2] = np.dot(self._data[:,:2], rot_mat.transpose())
        # Add the labels back on after the rotation
        return_mat[:,2] = self._data[:,2]
        return Dataset(return_mat)

    def _get_rotation_matrix(self, theta_degrees):
        theta = np.radians(theta_degrees)
        cos, sin = np.cos(theta), np.sin(theta)
        rotation_matrix = np.matrix([[cos, -sin], [sin, cos]])
        return rotation_matrix

    @property
    def x1(self):
        return self._data[:,0]

    @property
    def x2(self):
        return self._data[:,1]

    @property
    def target(self):
        return self._data[:,2]

    @property
    def raw_data(self):
        return self._data

def split_data_set(data_set):
    np.random.shuffle(data_set)
    # Split the data into training, validation and test sets
    # Assumes number of samples multiple of 4
    # TODO: Add asserts
    train_ratio, validation_ratio, test_ratio = 0.5, 0.25, 0.25
    train_split = data_set.shape[0] * train_ratio
    validation_split = data_set.shape[0] * validation_ratio
    test_split = data_set.shape[0] * test_ratio

    training_set = data_set[:train_split, :]
    validation_set = data_set[:validation_split, :]
    test_set = data_set[:test_split, :]
    return training_set, validation_set, test_set

def generate_perceptron_dataset():
    """
    Provides a train/ validation/ test set for the perceptron model e.g.
    only the classes that are linearly separable.
    """
    # Create the uniform distributions
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
    all_data = np.concatenate((class_one.raw_data,
                       class_two.raw_data),
                      axis=0)
    return split_data_set(all_data)



def generate_dataset(show_plot = False):
    """
    Provides a train/ validation/ test set for the perceptron model e.g.

    """
    # Create the uniform distributions
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

    # Create the multivariate gaussian distributions
    normal_mean_1 = np.array((-2, -3))
    normal_cov_1 = np.array(((0.5, 0),(0, 3)))
    class_three = Dataset.create_multivariate_normal_distribution(mean = normal_mean_1, 
                                                                  covariance_matrix = normal_cov_1,
                                                                  label = 3)
    
    normal_mean_2 = np.array((-4, -1))
    normal_cov_2 = np.array(((3, 0.5),(0.5, 0.5)))
    class_four = Dataset.create_multivariate_normal_distribution(mean = normal_mean_2, 
                                                                 covariance_matrix = normal_cov_2,
                                                                 label = 4)
    # Concatenate and shuffle the four classes
    all_data = np.concatenate((class_one.raw_data,
                   class_two.raw_data,
                   class_three.raw_data,
                   class_four.raw_data),
                  axis=0)
    training_set, validation_set, test_set = split_data_set(all_data)

    if(show_plot):
        plt.scatter(class_one.x1, class_one.x2, color='red')
        plt.scatter(class_two.x1, class_two.x2, color='blue')
        plt.scatter(class_three.x1, class_three.x2, color = 'green')
        plt.scatter(class_four.x1, class_four.x2, color = 'cyan')
        plt.show()
    return training_set, validation_set, test_set

if __name__ == "__main__":
    generate_dataset(show_plot = True)


