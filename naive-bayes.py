import warnings
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cp


# Import data
x, y = cp.load(open('/Users/mark/Desktop/voting.pickle', 'rb'))

N, D = x.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
xtrain = x[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
xtest = x[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]
print(xtrain[0:2])
print(ytrain[0:2])
print(xtrain.shape)
print(ytrain.shape)


# NBC Classifier
class NBC(object):
    def __init__(self, feature_types = None, num_classes = -1):
        self._num_classes = num_classes
        # Check valid feature types
        if not np.all(np.in1d(feature_types, ['b','r'])):
            raise ValueError("Invalid feature types.")
        # Assign feature types
        self._feature_types = feature_types
    
    def fit(self, xtrain, ytrain):
        # Count the number of classes and prepare list to hold parameters for Gaussian model in each class
        self._classes = np.unique(ytrain)
        # Should have correct # of features/ classes by this stage
        # TODO: Test set might not have all classes!!
        # assert self._num_classes == len(self._classes), "Different number of classes in data, than specified."
        assert self._feature_types.shape[0] == xtrain.shape[1], "Different number of features to feature types."
        self._means = [None] * self._num_classes
        self._varis = [None] * self._num_classes
        self._feature_probs = [None] * self._num_classes
        self._class_probs = np.bincount(ytrain.astype(int)) / ytrain.shape[0]
        
        # Find the mean, variance and probability for each class
        for idx, class_label in enumerate(self._classes):
            # Identify the training records for the current class
            inds = np.array(np.where(ytrain == class_label))
            cur_label_data = np.take(xtrain, inds, axis =0).reshape(inds.shape[1], xtrain.shape[1])
            cur_mean = cur_label_data.mean(axis=0)
            cur_var = (np.square((cur_label_data - cur_mean))).mean(axis=0)
        
            # Replace variance zero with small non -zero value, to prevent division by zero errors
            min_var_val = 1e-6
            cur_var[cur_var == 0] = min_var_val

            # Use Laplace Sampling with alpha = alpha, to prevent log(<=0) errors
            alpha = 1.
            feat_prob_numerator = cur_label_data.sum(axis=0) + alpha
            feat_prob_denominator = cur_label_data.shape[0] + (2 * alpha) # TODO: 2 is number of options for feature
            cur_feature_probs = feat_prob_numerator / feat_prob_denominator            
            
            # Store features for prediction
            self._means[idx] = cur_mean
            self._varis[idx] = cur_var
            self._feature_probs[idx] = cur_feature_probs
    
    def predict(self, xtest):
        # predict
        class_likelihoods=[]
        class_labels=[]

        # For each class, calculate the likelihoods
        for idx, class_label in enumerate(self._classes):
            mean = self._means[idx]
            var = self._varis[idx]
            feat_prob = self._feature_probs[idx]
            class_prob = self._class_probs[int(class_label)]
            
            # Select the distribution used, based on feature type            
            # Both Bernoulli and Gauss calculated, irrelvant of feature type. May result in log-zero error,
            # even though not used in calculation. Zero errors, handled manually on lines below.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bern_likelihood = self.bernoulliPDF(xtest, feat_prob)
                gauss_likelihood = self.gaussianPDF(xtest, mean, var)
            likelihood = self._get_feature_specific_likelihood(bern_likelihood, gauss_likelihood, self._feature_types)
            likelihood = likelihood.mean(axis=1)

            isNan = np.isnan(likelihood).any()
            if(isNan):
                raise ValueError("Zero-Error. Log zero or variance zero error.")

            # Calculate final likelihood for each class
            likelihood = class_prob * likelihood # TODO: ADD IN class_prob * likelihood
            class_likelihoods.append(likelihood)
            class_labels.append(class_label)

        # Find the maximum likelihood among classes for each data point, record the index
        max_likelihood_class_indices = np.argmax(np.array(class_likelihoods),0)
        # Use the found index, to lookup the associated class type
        predicted_labels = np.take(np.array(class_labels), max_likelihood_class_indices)
        return predicted_labels
    
    def bernoulliPDF(self, xtest, feature_probs):
        # Log likelkihood Bernoulli posterior prob (matrix form)
        val = ((xtest * np.log(feature_probs)) + ((1-xtest)*np.log(1-feature_probs)))
        return val
    
    def gaussianPDF(self, x_val, mean, var):
        # Log matrix interpretation of Gaussian model
        # Designed to take whole dataset in one go
        feature_likelihoods = np.tile(((-1/2)*np.log(2*np.pi*var)),(x_val.shape[0],1)).T - (np.square(x_val - mean)/(2*var)).sum(axis=1)
        return feature_likelihoods.T
    
    def accuracy(self, preds, real):
        return (preds==real).mean()
    
    def _get_feature_specific_likelihood(self, bern_likelihood, gauss_likelihood, feats):
        assert bern_likelihood.shape[1] == gauss_likelihood.shape[1] == feats.shape[0], "Error in likelohood shapes."
        likelihood = np.zeros(gauss_likelihood.shape)

        for idx, feature in enumerate(feats):
            if(feature == 'b'):
                # Binary => Bernoulli
                likelihood[:,idx] = bern_likelihood[:,idx]
            elif(feature == 'r'):
                # Real => Gaussian
                likelihood[:,idx] = gauss_likelihood[:,idx]
            else:
                raise ValueError('Invalid feature type (feature {})'.format(idx))
        return likelihood


# In[243]:


# nBc = NBC(feature_types = np.array(['r','r','r','r']), num_classes=3)
# nBc = NBC(feature_types = np.array(['b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b']), num_classes=2)
nBc = NBC(feature_types = np.array(['r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r']), num_classes=2)

nBc.fit(xtrain, ytrain)
predicted_labels = nBc.predict(xtest)
print("Accuracy: {0}".format(nBc.accuracy(predicted_labels, ytest)*100))
print("Number of mislabeled points out of a total %d points : %d"
      % (xtest.shape[0],(ytest != predicted_labels).sum()))


# Quick comparison between NBC implementation and Logistic Regression (SKLearn)
from sklearn.linear_model import LogisticRegression

def compare_lr_nbc(xtrain, ytrain, xtest, ytest, features, number_classes):
    logreg = LogisticRegression(C=1e5)

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(xtrain, ytrain)
    yhat = logreg.predict(xtest)
    logRegAccuracy = np.array(yhat == ytest).mean()*100

    # NBC
    nBc = NBC(feature_types = features, num_classes=number_classes)
    nBc.fit(xtrain, ytrain)
    yhat = nBc.predict(xtest)
    nBcAccuracy = np.array(yhat == ytest).mean()*100

#     print("Logisitc Regression/ NBC Accuracy: {0:3.2f}/{1:3.2f}".format(logRegAccuracy, nBcAccuracy))
    return logRegAccuracy, nBcAccuracy

# Show plot comparison between NBC implementation and logistic regression
def show_plot(IRIS, title):
    if not IRIS:
        x, y = cp.load(open('/Users/mark/Desktop/voting.pickle', 'rb'))
#         FEATURES = np.array(['b','b','b','b','b','b','b','b','b','b','b','b','b','b','b','b'])
        FEATURES = np.array(['r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r'])
        NUM_CLASSES = 2
    else:
        from sklearn.datasets import load_iris
        iris = load_iris()
        x, y = iris['data'], iris['target']
        FEATURES = np.array(['r','r','r', 'r'])
        NUM_CLASSES = 3

    NUM_RUNS = 500
    mean_log_accs = []
    mean_nbc_accs = []
    for j in range(NUM_RUNS):
        N, D = x.shape
        Ntrain = int(0.8 * N)
        shuffler = np.random.permutation(N)
        xtrain = x[shuffler[:Ntrain]]
        ytrain = y[shuffler[:Ntrain]]
        xtest = x[shuffler[Ntrain:]]
        ytest = y[shuffler[Ntrain:]]

        log_accs = np.array([])
        nbc_accs = np.array([])
        # Calculate for each percentage
        for i in range(10):
            percentage = (i+1)*.1
            idx_max = int(percentage*xtrain.shape[0])
            cur_xtrain = xtrain[0:idx_max]
            cur_ytrain = ytrain[0:idx_max]
            c_log_acc, c_nbc_acc = compare_lr_nbc(cur_xtrain, cur_ytrain, xtest, ytest, FEATURES, NUM_CLASSES)
            log_accs = np.append(log_accs, c_log_acc)
            nbc_accs = np.append(nbc_accs, c_nbc_acc)

        mean_log_accs.append(log_accs)
        mean_nbc_accs.append(nbc_accs)

    final_mean_log_accs = np.array(mean_log_accs).T.mean(axis=1)
    final_mean_nbc_accs = np.array(mean_nbc_accs).T.mean(axis=1)

    plt.plot(final_mean_log_accs, label="Logisitc Regression")
    plt.plot(final_mean_nbc_accs, label = "Naive Bayes")
    # plt.xticks(np.arange(0, 10, 10.0))
    plt.xlabel("Percentage of Training Data Used")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.show()
    
show_plot(True, "Iris Data")
show_plot(False, "Voting Data")

