#-------------------------------------------------------------------------------
# Name:        svm for breast cancer data
# Purpose:
#
# Author:      Shi
#
# Created:     25/03/2016
# Copyright:   (c) Administrator 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import smo
import matplotlib.pyplot as plt

MIN_SUPPORT_VECTOR_MULTIPLIER=0.01

class SVMTrainer(object):
    def __init__(self, kernel, c, id):
        self._kernel = kernel
        self._c = c
        self.id=id

    def train(self, X, y):
        """Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """
        #[lagrange_multipliers,bias] = self._compute_multipliers(X, y)
        #print lagrange_multipliers
        #print bias
        #return self._construct_predictor(X, y, lagrange_multipliers,bias)
        return self._compute_multipliers(X,y)

    def construct_predictor(self, X, y, lagrange_multipliers,bias):
        support_vector_indices = \
            lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        # Thus we can just predict an example with bias of zero, and
        # compute error.

        return SVMPredictor(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def _compute_multipliers(self, X, y):
        tol=0.001
        passes=10
        s=smo.smo(self._c, tol, passes, X, y, self._kernel)
        s.saveM(self.id)
        return s.opt()

class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._weights = weights
        self._bias=bias
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):

            result += z_i * y_i * self._kernel(x_i, x)
        return [result, np.sign(result).item()]

def main():
    import kernel
    import os
    data = np.genfromtxt("breast-cancer-wisconsin1.data",delimiter=",")
    #You can either remove the samples with missing features or replace the missing features with their column-wise medians or means.

    #samples=np.array([[2,1],[2,2],[0,1],[0,2],[0,4],[5,1],[2,3],[1,3]])
    #labels=np.array([1,1,-1,-1,-1,1,1,-1])
    samples=data[150:660,1:10]
    labels=data[150:660,10]-3


    # initialize the samples, y and a in svm_model


    svm = SVMTrainer(kernel.Kernel.gaussian(4), 20, '01')

    [multipliers, b]=svm.train(samples,labels)

    svm_predictor=svm.construct_predictor(samples,labels,multipliers,b)
    print "training done..."
    print multipliers
    print b
    #print samples
    #print "predict: ", svm_predictor.predict([1.5,500])
    #print res
    #plt.subplot(2, 1, 1)
    #plt.scatter(samples[:,0].ravel(), samples[:,1].ravel(), c=labels, alpha=0.5)

    #plt.subplot(2, 1, 2)
    #plt.scatter(samples[:,0].ravel(), samples[:,1].ravel(), c=res[0], alpha=0.5)
    #plt.show()
    #test_samples=data[200:len(data)-1,1:10]

    test_confidence=[]
    test_samples=data[0:150,1:10]
    test_labels=[]

    test_samples=np.array(test_samples)

    for row in test_samples:
        [a,b]=svm_predictor.predict(row)
        test_labels.append(b+3)
        test_confidence.append(a)

    test_labels_real=data[0:150,10]

    #print test_labels
    #print test_labels_real
    #print test_confidence
    indices=test_confidence>0
    indices_1=test_confidence<0
    a=test_confidence[indices]
    b=test_confidence[indices_1]
    print "average confidence >0 : ", np.mean(a)
    print "average confidence <0 : ", np.mean(b)

    right=0
    wrong=0
    colors=[]
    for i, j,k in zip(test_labels,test_labels_real,test_confidence):
        if (i==j):
            right+=1
            colors.append(0)
        else:
            wrong+=1
            colors.append(1)

    print "right: ", right
    print "wrong: ", wrong
    #plt.plot(range(len(test_confidence)),test_confidence, 'o')
    plt.scatter(range(len(test_confidence)),test_confidence, c=colors, alpha=0.5)
    plt.show()



if __name__ == '__main__':
    main()
