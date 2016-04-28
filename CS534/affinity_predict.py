# is_vague() pass to svm
# predict(x)
import numpy as np
path="features_all_compress5/"

class affinity_predict():
    def __init__(self,test_sample):
        center=np.load(path+"af_all.npz")['center']
        center_labels=np.load(path+"af_all.npz")['center_labels']
        mini=100000000
        mini_1=100000000
        mini_2=100000000
        j=0
        k=0
        kk=0
        for i in range(center.shape[0]):
            iii=self.compute_distance(test_sample,center[i])
            if mini>iii:
                mini=iii
                j=i
            elif mini_1>iii and iii>mini:
                mini_1=iii
                k=i
            elif mini_2>iii and iii>mini_1:
                mini_2=iii
                kk=i
        print "return the cluster: ", j, "distance is ", mini
        self.second_nearest=center_labels[k]
        self.first_nearest=center_labels[j]
        self.third_nearest=center_labels[kk]

    def compute_distance(self,x,y):
        from math import sqrt
        return sqrt(sum((x-y)**2))

    def is_vague(self):
        if self.second_nearest!=self.first_nearest or self.first_nearest!=self.third_nearest or self.second_nearest!=self.third_nearest:
            return True
        else:
            return False


