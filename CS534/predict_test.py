import numpy as np
import multiclass


class all_classifiers():
    def __init__(self,strr):
        self.path=strr
        #self.path="multipliers"
        u=np.load(self.path+'/labels.npz')['ulabels']
        print "loading classifiers...labels are: ", u
        self.classifier={}
        self.classifier['kernel']=np.load(self.path+'/labels.npz')['k']
        self.classifier['labels']=u
        uu=list(u)
        for i in u:
            uu.remove(i)
            for j in uu:
                id=str(i)+","+str(j)
                bias=np.load(self.path+'/'+id+'.npz')['bias']
                support_multipliers = np.load(self.path+'/'+id+'.npz')['support_multipliers']
                support_vectors = np.load(self.path+'/'+id+'.npz')['support_vectors']
                support_vector_labels = np.load(self.path+'/'+id+'.npz')['support_vector_labels']
                c=classifier(bias,support_vectors,support_multipliers,support_vector_labels)
                self.classifier[id]=c
        print len(self.classifier)-2, "classifiers are loaded..."

class classifier():
    def __init__(self, bias, support_vectors, support_vector_multipliers, support_vector_labels):
        self.bias=bias
        self.support_vectors=support_vectors
        self.support_vectors_multipliers=support_vector_multipliers
        self.support_vector_labels=support_vector_labels

def main():
    #strr="features_all_without_skeleton"
    #strr="features_density_direction"
    #data=np.genfromtxt("optdigits.tes",delimiter=",")
    strr="features_all_compress5"
    data = np.genfromtxt("test_"+strr+".csv",delimiter=",")
    print data.shape
    c=all_classifiers(strr).classifier



    test_samples=data[:,0:-1]
    test_labels_real=data[:,-1]
    test_labels=[]
    for t in test_samples:
        p=multiclass.predict_new(t,c)
        test_labels.append(p)
        print p
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


    print "testing done..."


    def calculate_confusion(labels,Y):
        Y=Y
        a=np.unique(labels)
        from sklearn.metrics import confusion_matrix
        print "confusion matrix is: "
        print a
        c=confusion_matrix(Y, labels)
        print c
        print "overall accuracy is: ", np.trace(c)*1.0/size
        return c

    size=test_samples.shape[0]
    calculate_confusion(test_labels_real,test_labels)
#if __name__ == '__main__':
#    main()