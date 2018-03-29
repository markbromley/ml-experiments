import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as pl


# Test data
n = 150
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# Define the kernel function
def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

param = 0.1
K_ss = kernel(Xtest, Xtest, param)

# Get cholesky decomposition (square root) of the
# covariance matrix
L = scipy.linalg.cholesky(K_ss + 1e-13*np.eye(n), lower=True)

L = np.linalg.cholesky(K_ss + 1e-13*np.eye(n))
# Sample 150 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L, np.random.normal(size=(n,150)))


# Plot the sampled functions
pl.plot(Xtest, f_prior,alpha=0.3)
pl.axis([-5, 5, -6, 6])
pl.title('150 Samples from Prior Distribution')
pl.show()


#### Noiseless training data
Xtrain = np.array([-4, -3, -2, -1, 0,1,2,3,4]).reshape(9,1)
Xtrain = np.arange(-4,4,.8).reshape(-1,1)
ytrain = np.sin(Xtrain)

# Apply the kernel function to our training points
K = kernel(Xtrain, Xtrain, param)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_s = kernel(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)
# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,100)))


# Plot the posterior distribution
pl.rc('figure', figsize=(12, 8))
pl.plot(Xtest, f_post,alpha=0.3)
pl.gca().fill_between(Xtest.flat, mu-3*stdv, mu+3*stdv, color="#eeeeee")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.plot(Xtrain, ytrain, 'bo', ms=6)
pl.axis([-5, 5, -6, 6])
pl.title('100 Samples from Posterior Distribution')
pl.show()

