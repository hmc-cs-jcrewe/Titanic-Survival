"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2017 Jan 11
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *

from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        vals, counts = np.unique(y, return_counts=True)
        majority_val, majority_count = max(zip(vals, counts), key=lambda (val, count): count)
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        # Generate the counts and probabilities for the majority and minority values 
        vals, counts = np.unique(y, return_counts=True)
        majority_val, majority_count = max(zip(vals, counts), key=lambda (val, count): count)
        minority_val, minority_count = min(zip(vals, counts), key=lambda (val, count): count)
        total_count = majority_count + minority_count
        majority_probability = float(majority_count) / float(total_count)
        minority_probability = float(minority_count) / float(total_count)

        # Generate dictionary using the values calculated above
        self.probabilities_ = {majority_val : majority_probability, minority_val : minority_probability}

        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        # np.random.choice assigns the keys into the array at the probability specified in its last parameter 
        y = np.random.choice(self.probabilities_.keys(), len(X), True, self.probabilities_.values())
        return y


######################################################################
# functions
######################################################################

def plot_histogram(X, y, Xname, yname) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    plt.figure()
    n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
    plt.xlabel(Xname)
    plt.ylabel('Frequency')
    plt.legend() #plt.legend(loc='upper left')
    plt.show()


def plot_scatter(X, y, Xnames, yname):
    """
    Plots scatter plot of values in X grouped by y.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,2), feature values
        y      -- numpy array of shape (n,), target classes
        Xnames -- tuple of strings, names of features
        yname  -- string, name of target
    """
    
    # plot
    targets = sorted(set(y))
    plt.figure()

    # set up data and labels
    ageData = X[:,0]
    fareData = X[:,1]
    labels = ["Did not survive", "Survived"]

    # create scatter plot with labels
    for target in targets:
        age =  [ageData[i] for i in xrange(len(y)) if y[i] == target]
        fare = [fareData[i] for i in xrange(len(y)) if y[i] == target]
        plt.scatter(age, fare, label=labels[int(target)])

    plt.autoscale(enable=True)
    plt.xlabel(Xnames[0])
    plt.ylabel(Xnames[1])
    plt.legend()
    plt.show()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    print X
    
    
    #========================================
    # plot histograms of each feature
    print 'Plotting...'
    for i in xrange(d) :
       plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
    
    
    
    
    # part b: make scatterplot of age versus fare
    Xscatter = X[:,2:6:3]
    plot_scatter(Xscatter, y, Xnames = Xnames[2:6:3], yname = yname)
    
    
    
    #========================================
    # train Majority Vote classifier on data
    print 'Classifying using Majority Vote...'
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    
    #========================================
    # train Random classifier on data
    print 'Classifying using Random...'
    clf2 = RandomClassifier()   # create a Random classifier which includes all model parameters
    clf2.fit(X,y)               # fit training data using the classifier 
    y_pred = clf2.predict(X)    # take the classifier and run it on the training data 
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    
    
    print 'Done'


if __name__ == "__main__":
    main()