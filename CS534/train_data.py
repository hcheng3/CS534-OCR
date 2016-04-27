import kernel
import numpy as np
import multiclass

data = np.genfromtxt("features_direction.csv",delimiter=",")

samples=data[:,:-1]
labels=data[:,-1]

k=kernel.Kernel.gaussian(6)
str="classifiers-direction"
m=multiclass.multiclass(samples,labels,k,20,str)



print "training done..."