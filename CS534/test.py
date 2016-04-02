#-------------------------------------------------------------------------------
# Name:        test1
# Purpose:
#
# Author:      Administrator
#
# Created:     31/03/2016
# Copyright:   (c) Administrator 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import trainer
import predictor
import random
import numpy as np
import matplotlib.pyplot as plt
import kernel

def main():
    num_samples=100
    num_features=2
    samples = np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features)
    labels = 2 * (samples.sum(axis=1) > 0) - 1.0

    svm01=trainer.trainer(kernel.Kernel.linear(), 20, '01')
    svm01.train(samples,labels)
    predictor01=predictor.predictor(samples,labels,kernel.Kernel.linear(),'01')
    print predictor01.predict(np.array([0,0]))
    print predictor01.predict(np.array([-1,-1]))
    print predictor01.predict(np.array([-2,-2]))
    print predictor01.predict(np.array([-5,-5]))

    #plt.subplot(2, 1, 1)
    #plt.scatter(samples[:,0].ravel(), samples[:,1].ravel(), c=labels, alpha=0.5)

    plt.show()

if __name__ == '__main__':
    main()
