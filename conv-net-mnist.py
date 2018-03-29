# Convolutional Networks

from matplotlib import pylab
pylab.rcParams['figure.figsize'] = (3.0, 3.0)

from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ### Initialisations
# 
# Helper functions to initialise weights and biases. Initialise weights as Gaussian random variables with mean 0 and variance 0.0025. Initialise biases with a constant 0.1. Primarily using ReLU non-linearities.

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# ### Model
# 
# Define model as follows:
# * An input that is 728 dimensional vector. 
# * Reshape the input as 28x28x1 images (only 1 because they are grey scale) 
# * A convolutional layer with 25 filters of shape 12x12x1 and a ReLU non-linearity (with stride (2, 2) and no padding)
# * A convolutional layer with 64 filters of shape 5x5x25 and a ReLU non-linearity (with stride (1, 2) and padding to maintain size)
# * A max_pooling layer of shape 2x2
# * A fully connected layer taking all the outputs of the max_pooling layer to 1024 units and ReLU nonlinearity
# * A fully connected layer taking 1024 units to 10 no activation function (the softmax non-linearity will be included in the loss function rather than in the model)

x = tf.placeholder(tf.float32, shape=[None, 784])
x_ = tf.reshape(x, [-1, 28, 28, 1]) # batch examples, width, height, num channels
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define the first convolution layer
# TODO strides = [1, x, y, 1]
W_conv1 = weight_variable([12, 12, 1, 25])
b_conv1 = bias_variable([25])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_, W_conv1, strides=[1, 2, 2, 1], padding='VALID')
                     + b_conv1)

# Define the second convolution layer
W_conv2 = weight_variable([5,5,25,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,1,2,1], padding='SAME') 
                     + b_conv2)

# Define maxpooling
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# All subsequent layers fully connected ignoring geometry so flatten layer
# Flatten the h_pool2_layer (as it has a multidimensional shape) 
h_pool2_flat = tf.reshape(h_pool2, [-1, 15*64])

# Define the first fully connected layer
W_fc1 = weight_variable([15 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# The final fully connected layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# #### Loss Function, Accuracy and Training Algorithm
# 
# * Cross entropy loss function. Accuracy as the fraction of data correctly classified
# Adam optimiser with learning rate at 1e-4.

# Cross entropy loss function 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Classification accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Adam optimiser
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)


# Load the mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Visualise the first 16 data points from the MNIST training data
fig = plt.figure()
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.imshow(mnist.train.images[i].reshape(28, 28), cmap='Greys_r')  

# Start a tf session and run the optimisation algorithm
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0}, session = sess)
        print('Step %d, Training Accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, session=sess)

print('Test Accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}, session=sess))

print ('Test Accuracy: %g' % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# #### Visualise Filters
# 
# Visualise all 25 filters in the first convolution layer. Each of shape 12x12x1, (they may themselves be viewed as greyscale images).

# Visualise the filters in the first convolutional layer
with sess.as_default():
    W = W_conv1.eval()
    
    # Add code to visualise filters here
    pylab.rcParams['figure.figsize'] = (5.0, 5.0)

    fig = plt.figure()
    fig.suptitle('Visualisation of 25 Filters in Convolutional Layer 1', fontsize=10)
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.imshow(W[:,:,:,i].reshape(W.shape[0], W.shape[1]), cmap='Greys_r')  


#Identifying image patches that activate the filters
# 
# Identifying the image patches in the test-set that activate each of the first 5 filters that maximise the activation for that filter.

def show_patch_with_high_activation(H, feature_map_id):
    # Select a particular feature map
    feature_map_acts = H[:,:,:,feature_map_id]

    # List of indices in ascending order
    activation_indices = np.unravel_index(
        np.argsort(feature_map_acts, axis=None), feature_map_acts.shape)

    # Constants
    w,p,s = 28, 12, 2 # image width/ height, patch size, step size

    # Get the indices
    num_examples = 12 # Number examples to display
    im_ids = activation_indices[0][-num_examples:]
    x_idxs = activation_indices[1][-num_examples:]
    y_idxs = activation_indices[2][-num_examples:]

    fig = plt.figure()
    fig.suptitle('Patches with High Activation for Feature Map {}'.format(feature_map_id), fontsize=10)
    for i in range(num_examples):
        # Calculate patch coords
        im_id, x_id, y_id = im_ids[i], x_idxs[i], y_idxs[i]
        x_coord_1 = s * x_id
        y_coord_1 = s * y_id
        x_coord_2 = x_coord_1 + p
        y_coord_2 = y_coord_1 + p

        # Display
        ax = fig.add_subplot(4, 3, i + 1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.imshow(mnist.test.images[im_id].reshape((w,w))
                  [x_coord_1:x_coord_2, y_coord_1:y_coord_2], cmap='Greys_r')  

# Calculate activations over test set
H = sess.run(h_conv1, feed_dict={x: mnist.test.images})

# Display results
# show_patch_with_high_activation(H, feature_map_id=20)
show_patch_with_high_activation(H, feature_map_id=0)
# show_patch_with_high_activation(H, feature_map_id=8)
# show_patch_with_high_activation(H, feature_map_id=26)
# show_patch_with_high_activation(H, feature_map_id=14)

