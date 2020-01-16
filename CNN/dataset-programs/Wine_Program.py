import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from pandas.plotting import scatter_matrix
import pandas as pd
from pandas import read_csv
import numpy as np

filename = '***/Datasets/wine.csv'
names = ['Cls','Alchohol', 'Maltic_Acid', 'Ash', 'Alcalinity', 'Magnesium','Total_Phenols','Flavanoids',
         'Nonfalvanoid_Phenols','Proanthocyanins','Color_Intensity','Hue','OD280_OD315','Proline']
dataset = read_csv(filename, names=names)

print(dataset.shape)
print(dataset)

y = dataset['Cls']
x = dataset.drop(columns='Cls')
print(x)
print(y)

#histogram
x.hist()
plt.show()

# density
x.plot(kind='density', subplots=True, layout=(2,7), sharex=False)
plt.show()

# scatter plot matrix
scatter_matrix(x)
plt.show()

# box and whisker plots
x.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
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

#Plotnmm
plt.subplot(1, 2, 1)
plt.scatter(x.Alcalinity, x.Ash, c=y, s=40)
plt.title('Alcalinity')

plt.subplot(1, 2, 2)
plt.scatter(x.Magnesium, x.Ash, c=y, s=40)
plt.title('Magnesium')
plt.show()

#K means cluster
model = KMeans(n_clusters=3)
model.fit(x)

model.labels_

print(model.labels_)

#view the results
#set the size of the plot
plt.figure(figsize=(14,7))

#plot original classifications
plt.subplot(1, 2, 1)
plt.scatter(x.Alcalinity, x.Ash, c=y, s=40)
plt.title('Real Classification')
#plt.show()
#Plot the models classifications
plt.subplot(1, 2, 2)
plt.scatter(x.Alcalinity, x.Ash, c=model.labels_, s=40)
plt.title('K Mean Classification Labels Not Adjusted')
plt.show()

#the fix, we convert all 1s to 0s and 0s to 1s
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)
print('MODEL LABELS')
print  (model.labels_)
print('PRED Y')
print (predY)
print('Y')
print(y)
#view results
#set the size of plot
plt.figure(figsize=(14,7))

#plot original classifications
plt.subplot(1, 2, 1)
plt.scatter(x.Alcalinity, x.Ash, c=y, s=40)
plt.title('Real Classification')
#Plot the models classifications
plt.subplot(1, 2, 2)
plt.scatter(x.Alcalinity, x.Ash, c=predY, s=40)
plt.title('K Mean Classification')
plt.show()

#performance metrics
print(sm.accuracy_score(y, predY))

# Confusion matrix
print(sm.confusion_matrix(y, predY))
