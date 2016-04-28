#-------------------------------------------------------------------------------
# Name:        multi-class  using max win
# Purpose:
#
# Author:      Shi
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
# kernel :1: linear 2: gaussian
class multiclass():
    def __init__(self,samples,labels,kernel,c,strr):
        self.u= np.unique(labels)
        self.k=self.u.size
        self.kernel=kernel
        print "labels are: ", self.u
        dic=[]
        for i in range(self.k):
            indices1= labels==self.u[i]
            samples1= samples[indices1]
            labels1=np.ones(samples1.shape[0])
            for j in range(i+1, self.k):
                indices2= labels==self.u[j]
                samples2= samples[indices2]
                labels2=np.ones(samples2.shape[0])*(-1)
                t=trainer.trainer(kernel, c, str(self.u[i])+","+str(self.u[j]),strr+"/")
                t.train(np.vstack((samples1,samples2)), np.hstack((labels1,labels2)))
        np.savez(strr+'/labels', ulabels=self.u, k=1)

class multipredict():
    def __init__(self, test_sample,multiclass):
        u=multiclass.u
        self.k=u.size
        pp=np.zeros(self.k)
        for i in range(self.k):
            for j in range(i+1, self.k):
                p1=predictor.predictor(multiclass.kernel,str(u[i])+","+str(u[j]))
                p=p1.predict(test_sample)
                if p==1:
                    pp[i]+=1
                else:
                    #print i, j
                    pp[j]+=1
        print "pp: ", pp
        self.value=u[np.argmax(pp)]

class multipredict1():
    def __init__(self, test_sample):   # u is the array of  unique labels
        u=np.load('multipliers/labels.npz')['ulabels']
        if(np.load('multipliers/labels.npz')['k']==1):
            k=kernel.Kernel.gaussian(6)
        else:
            print "kernel in multipredict1 error"
        self.k=u.size
        pp=np.zeros(self.k)
        for i in range(self.k):
            for j in range(i+1, self.k):
                p1=predictor.predictor(k,str(u[i])+","+str(u[j]))
                p=p1.predict(test_sample)
                #print u[i],u[j], p
                if p==1:
                    pp[i]+=1
                    pp[j]-=1
                else:
                    #print i, j
                    pp[j]+=1
                    pp[i]-=1
        print u
        print pp
        self.value=u[np.argmax(pp)]



def predict_new(test_sample, c, path):
        dict={'0.0':0,'1.0':1,'2.0':2,'3.0':3,'4.0':4,'5.0':5,'6.0':6,'7.0':7,'8.0':8,'9.0':9,'a':11,'b':12, 'c':13,'d':14,
              'e':15,'f':16,'g':17,'h':18,'i':19,'j':20,'k':21,'l':22,'m':23,'n':24,'o':25}
        u=c.get('labels')
        if(c.get('kernel')==1):
            ker=kernel.Kernel.gaussian(6)
        elif(c.get('kernel')==0):
            ker=kernel.Kernel.linear()
        else:
            print "kernel in multipredict1 error"

        pp=np.zeros(u.size)
        for k in c.keys():
            if k=='kernel' or k=='labels':
                continue
            i=dict.get(k.split(",")[0])
            j=dict.get(k.split(',')[1])
            #p=predictor.predictor_new(ker,c.get(k),test_sample)
            p1=predictor.predictor(ker,k, path)
            p=p1.predict(test_sample)
            if p==1:

                pp[i]+=1
                #pp[j]-=1
            else:
                pp[j]+=1
                #pp[i]-=1
            #print i,j, p
        print u
        print pp
        return u[np.argmax(pp)]
        #res=u[np.argmax(pp)]
        # if ((res!=5 and res!=2 and res!=3 ) or (res==5 and pp[2]<pp[5]-1 and pp[3]<pp[5]-1) or (res==2 and pp[5]<pp[2]-1 )\
        #          or (res==3 and pp[5]<pp[3]-1) ):
        #     return res
        # path="features_all_compress5/"
        # #path="multipliers/"
        # center=np.load(path+"af235.npz")['center']
        # center_labels=np.load(path+"af235.npz")['center_labels']
        # mini=10000000000
        # j=0
        # print center.shape
        # for i in range(1,center.shape[0]):
        #     iii=compute_distance(test_sample,center[i])
        #     if mini>iii:
        #         mini=iii
        #         j=i
        # print "return the cluster: ", j, "distance is ", mini
        # return center_labels[j]


def compute_distance(x,y):
    from math import sqrt
    return sqrt(sum((x-y)**2))

def main():
    data = np.genfromtxt("optdigits.tra",delimiter=",")

    samples=data[:,0:-1]
    labels=data[:,-1]
#    i1=data[:,-1]==0
#    s1=data[i1,0:-1]
#    l1=data[i1,-1]
#    i2=data[:,-1]==1
#    s2=data[i2,0:-1]
#    l2=data[i2,-1]
#    i3=data[:,-1]==2
#    s3=data[i3,0:-1]
#    l3=data[i3,-1]
#    s11=np.vstack((s1,s2))
#    samples=np.vstack((s11,s3))
#    l11=np.hstack((l1,l2))
#    labels=np.hstack((l11,l3))

    k=kernel.Kernel.gaussian(6)
    #m=multiclass(samples,labels,k,20)
    #test_sample=np.array([0,0,10,16,6,0,0,0,0,7,16,8,16,5,0,0,0,11,16,0,6,14,3,0,0,12,12,0,0,11,11,0,0,12,12,0,0,8,12,0,0,7,15,1,0,13,11,0,0,0,16,8,10,15,3,0,0,0,10,16,15,3,0,0])
    test_data=np.genfromtxt("optdigits.tes",delimiter=",")
    test_samples=data[:,0:-1]
    test_labels_real=data[:,-1]
    test_labels=[]
    for t in test_samples:
        p=multipredict1(t)
        test_labels.append(p.value)
    #print p.value
    print test_labels
    print test_labels_real

    right=0
    wrong=0
    colors=[]
    for i, j in zip(test_labels,test_labels_real):
        if (i==j):
            right+=1
        else:
            wrong+=1

    print "right: ", right
    print "wrong: ", wrong

#if __name__ == '__main__':
#    main()