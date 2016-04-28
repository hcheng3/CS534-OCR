import numpy as np
str_1="features_all_without_skeleton"
data_1 = np.genfromtxt(str_1+".csv",delimiter=",")

str_2="features_density_direction"
data_2 = np.genfromtxt(str_2+".csv",delimiter=",")

str_3="features_profiling_direction"
data_3 = np.genfromtxt(str_3+".csv",delimiter=",")

str_4="features_profiling_density"
data_4 = np.genfromtxt(str_4+".csv",delimiter=",")

str_5="features_all_2345"
data_5 = np.genfromtxt(str_5+".csv",delimiter=",")

X_1=data_1[:,:-1]
Y_1=data_1[:,-1]
size_1=X_1.shape[0]

X_2=data_2[:,:-1]
Y_2=data_2[:,-1]
size_2=X_2.shape[0]
X_3=data_3[:,:-1]
Y_3=data_3[:,-1]
size_3=X_3.shape[0]
X_4=data_4[:,:-1]
Y_4=data_4[:,-1]
size_4=X_4.shape[0]
X_5=data_5[:,:-1]
Y_5=data_5[:,-1]
size_5=X_5.shape[0]



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
             % (size_1, n_neighbors), fontsize=14)

color_1 = 9-Y_1
color_2 = 9-Y_2
color_3 = 9-Y_3
color_4 = 9-Y_4
color_5 = 9-Y_5

xx = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X_1)
ax = fig.add_subplot(321)
plt.scatter(xx[:, 0], xx[:, 1], c=color_1, cmap=plt.cm.Spectral)
plt.title("features_all_without_skeleton")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

xx = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X_2)
ax = fig.add_subplot(322)
plt.scatter(xx[:, 0], xx[:, 1], c=color_2, cmap=plt.cm.Spectral)
plt.title("features_density_direction")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


xx = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X_3)
ax = fig.add_subplot(323)
plt.scatter(xx[:, 0], xx[:, 1], c=color_3, cmap=plt.cm.Spectral)
plt.title("features_profiling_direction")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


xx = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X_4)
ax = fig.add_subplot(324)
plt.scatter(xx[:, 0], xx[:, 1], c=color_4, cmap=plt.cm.Spectral)
plt.title("features_profilling_density")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


xx = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X_5)
ax = fig.add_subplot(325)
plt.scatter(xx[:, 0], xx[:, 1], c=color_5, cmap=plt.cm.Spectral)
plt.title("features_all_2345")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


plt.show()