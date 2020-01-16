import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from pandas.plotting import scatter_matrix

import pandas as pd
from pandas import read_csv
import numpy as np

filename = '***/Datasets/Tetra.csv'
names = ['C1','C2','C3','label']
dataset = read_csv(filename, names=names)

print(dataset.shape)
print(dataset)

y = dataset['label']
x = dataset.drop(columns='label')

print(x)
print()
print(y)
#histogram

x.hist()
plt.show()

# density
x.plot(kind='density', subplots=True, layout=(2,3), sharex=False)
plt.show()

# scatter plot matrix
scatter_matrix(x)
plt.show()

# box and whisker plots
x.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
plt.show()

# correlation matrix
correlations = x.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)
plt.show()

#set size of the plot
plt.figure(figsize=(14,7))

#create colormap
colormap = np.array(['red', 'lime', 'black'])

#Plot sepal
plt.subplot(1, 2, 1)
plt.scatter(x.C1, x.C2, c=y, s=40)
plt.title('C1')
#plt.show()
plt.subplot(1, 2, 2)
plt.scatter(x.C1, x.C2, c=y, s=40)
plt.title('C1')
plt.show()

#K means cluster
model = KMeans(n_clusters=4)
model.fit(x)

model.labels_

print(model.labels_)

#view the results
#set the size of the plot
plt.figure(figsize=(14,7))

#plot original classifications
plt.subplot(1, 2, 1)
plt.scatter(x.C1, x.C2, c=y, s=40)
plt.title('Real Classification')
#plt.show()
#Plot the models classifications
plt.subplot(1, 2, 2)
plt.scatter(x.C1, x.C2, c=model.labels_, s=40)
plt.title('K Mean Classification Labels Not Adjusted')
plt.show()

#the fix, we convert all 1s to 0s and 0s to 1s
predY = np.choose(model.labels_, [3, 2, 1, 4]).astype(np.int64)
print('MODEL LABELS')
print  (model.labels_)
print('PRED Y')
print (predY)
print('YYYYYYYY')
print(y)
#view results
#set the size of plot
plt.figure(figsize=(14,7))

#plot original classifications
plt.subplot(1, 2, 1)
plt.scatter(x.C1, x.C2, c=y, s=40)
plt.title('Real Classification')
#Plot the models classifications
plt.subplot(1, 2, 2)
plt.scatter(x.C1, x.C2, c=predY, s=40)
plt.title('K Mean Classification')
plt.show()

#performance metrics
print(sm.accuracy_score(y, predY))

# Confusion matrix
print(sm.confusion_matrix(y, predY))