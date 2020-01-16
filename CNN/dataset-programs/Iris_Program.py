import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np


#import some data to play with
iris = datasets.load_iris()

#print(iris.data)
#print(iris.feature_names)
#print(iris.target)
#print(iris.target_names)

x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

#5 additional options plotted below
#Box and Whiskers plot
x.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
plt.show()

#Density plot
x.plot(kind='density', subplots=True, layout=(2,7), sharex=False)
plt.show()

# histograms
x.hist()
plt.show()

# scatter plot matrix
scatter_matrix(x)
plt.show()

#Correlation_Matrix_Generic
correlations = x.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
plt.show()


#Set the size of the plot
plt.figure(figsize=(14,7))
#create a colormap
colormap = np.array(['red', 'lime', 'black'])
#plot Sepal
plt.subplot(1,2,1)
plt.scatter(x.Sepal_Length, x.Sepal_Width, c=colormap[y.Targets], s=40)
plt.title('Sepal')

plt.subplot(1,2,2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Petal')
plt.show()
#k Means Cluster
model = KMeans(n_clusters=3)
model.fit(x)

print(model.labels_)
#view the results
#set the size of the plot
plt.figure(figsize=(14,7))

#create a colormap
colormap = np.array(['red','lime','black'])

#Plot the Origiinal Classifications
plt.subplot(1,2,1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')

#plot the models classifications
plt.subplot(1,2,2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.show()

#The fix, we convert all the 1s to 0s and 0s to 1s
predY = np.choose(model.labels_,[1,0,2]).astype(np.int64)
print (model.labels_)
print(predY)

#View the results
#Set the size of the plot
plt.figure(figsize=(14,7))

#Create a colormap
colormap = np.array(['red','lime','black'])

#plot Original
plt.subplot(1,2,1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')

#plot Predicted with corrected values
plt.subplot(1,2,2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[predY], s=40)
plt.title('K Mean Classification')
plt.show()

#performance Metrics
print(sm.accuracy_score(y, predY))

#Confusion Matrix
print(sm.confusion_matrix(y, predY))





















