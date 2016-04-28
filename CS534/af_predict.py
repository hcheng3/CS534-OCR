import numpy as np


#path="multipliers/"
path="features_all_compress5/"
#data=np.genfromtxt("optdigits.tes",delimiter=",")
data = np.genfromtxt(str+".csv",delimiter=",")
test_samples=data[:,0:-1]
test_labels_real=data[:,-1]
test_labels=[]

center=np.load(path+"af_all.npz")['center']
center_labels=np.load(path+"af_all.npz")['center_labels']

print center_labels

def compute_distance(x,y):
    from math import sqrt
    return sqrt(sum((x-y)**2))

for t in test_samples:
    mini=10000000000
    j=0
    for i in range(center.shape[0]):
        iii=compute_distance(t,center[i])
        if mini>iii:
            mini=iii
            j=i
    print "return the cluster: ", j, "distance is ", mini
    p=center_labels[j]
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


import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold


n_neighbors=30
n_components=2

fig = plt.figure(figsize=(15, 8))

plt.suptitle("Manifold Learning with %i points, %i neighbors, t-SNE"
             % (data.shape[0], n_neighbors), fontsize=14)

color = test_labels_real

xx = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(test_samples)
ax = fig.add_subplot(121)
plt.scatter(xx[:, 0], xx[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("data with true label")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


color=test_labels
xx = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(test_samples)
ax = fig.add_subplot(122)
plt.scatter(xx[:, 0], xx[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("data with clustering label")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()



