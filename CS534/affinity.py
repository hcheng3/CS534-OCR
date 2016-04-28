import numpy as np


def test_labels(labels,Y):
    Y=Y
    labels_1=np.zeros(labels.size)
    for i in np.unique(labels):
        index=labels==i
        a=0
        b=labels[0]
        for j in range(0,10):
            c=sum(Y[index]==j)
            if a<c:
                b=j
                a=c
        labels_1[index]=b
        print "for class ", i, " the label is ", b, " . The accuracy is: ", a*1.0/labels[index].size*100, "%"
    return labels_1

def calculate_confusion(labels,Y):
    Y=Y
    size=Y.size
    a=np.unique(labels)
    from sklearn.metrics import confusion_matrix
    print "confusion matrix is: "
    print a
    c=confusion_matrix(Y, labels)
    print c
    print "overall accuracy is: ", np.trace(c)*1.0/size
    return c

def calculate_fm(cm, size):
    size=size
    t=np.sum(cm**2)-size
    p=sum(np.sum(cm, axis=0)**2)-size
    q=sum(np.sum(cm, axis=1)**2)-size
    #print "++++++", t, p ,q
    import math
    return t/math.sqrt(p*q)

def main():
    import numpy as np

    #str="features_all_without_skeleton"
    str="features_all_compress5"
    data = np.genfromtxt(str+".csv",delimiter=",")

    X=data[:,:-1]
    Y=data[:,-1]
    size=X.shape[0]

    print "++++++++++++++++++++Affinity propagation+++++++++++++"
    from sklearn.cluster import AffinityPropagation
    # Compute Affinity Propagation

    af = AffinityPropagation(preference=-150).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels_af = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    print "number of cluster is: ", n_clusters_
    labels_af=test_labels(labels_af, Y)
    cm=calculate_confusion(labels_af,Y)
    print " Fowlkes Mallows index index is: ", calculate_fm(cm,size)




    # Plot result
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold, datasets
    from sklearn.decomposition import PCA



    n_neighbors=30
    n_components=2

    fig = plt.figure(figsize=(15, 8))

    plt.suptitle("Manifold Learning with %i points, %i neighbors, t-SNE"
                 % (size, n_neighbors), fontsize=14)

    color = Y

    xx = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
    ax = fig.add_subplot(121)
    plt.scatter(xx[:, 0], xx[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("data with true label")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')


    color=labels_af
    xx = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
    ax = fig.add_subplot(122)
    plt.scatter(xx[:, 0], xx[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("data with clustering label")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    plt.show()

# if __name__ == '__main__':
#     main()