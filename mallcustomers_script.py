import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn import linear_model,preprocessing,svm,cluster,pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score,mean_squared_error

#Used to encode nominal data
le = preprocessing.LabelEncoder()
#Ridge regression data
reg = linear_model.Ridge(alpha=.5)
#Support vector machine using the rbf kernel
regr = svm.SVR(kernel='rbf')
clf = pipeline.make_pipeline(preprocessing.StandardScaler(),regr)
#Clustering
kmeans = cluster.KMeans(n_clusters=8)

mall_data = pd.read_csv("Mall_Customers.csv")
mall_data['Genre'] = le.fit_transform(mall_data['Genre'])
print(mall_data.head())

#Separate target and predictor data
target = mall_data['Spending Score (1-100)']
data = mall_data.drop(['Spending Score (1-100)','CustomerID'],axis=1)

#Split into test and training data
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state=10)
reg.fit(data_train,target_train)
pred = reg.predict(data_test)
print("Ridge Regression Mean error - %f" %(mean_squared_error(target_test,pred,squared=False)))

clf.fit(data_train,target_train)
pred = clf.predict(data_test)
print("SVR with RBF kernel mean error - %f" % (mean_squared_error(target_test,pred,squared=False)) )

#Assign class depending on clusters
kmeans.fit(data_train)
#Generate regression model for each class
cluster_class = kmeans.labels_
print(kmeans.predict((data_test.iloc[0].to_numpy()).reshape(1,-1))[0])
#breakpoint()
data_train = data_train.to_numpy()
data_test = data_test.to_numpy()
target_train = target_train.to_numpy()

#These two functions are used to retrieve data and predictors with class y
cluster_data_train = lambda y: [ data_train[x] for x in range(len(data_train)) if int(kmeans.predict(data_train[x].reshape(1,-1))[0]) == y]
cluster_target_train = lambda y: [ target_train[x] for x in range(len(data_train)) if int(kmeans.predict(data_train[x].reshape(1,-1))[0]) == y]
#Store SVR for every class
svc_array = [ pipeline.make_pipeline(preprocessing.StandardScaler(),svm.SVR(kernel='linear',gamma='auto')).fit(cluster_data_train(x),cluster_target_train(x)) for x in set(cluster_class)]
data_class = kmeans.predict(data_test)
pred = []
for output in range(len(data_class)):
    pred.append(svc_array[data_class[output]].predict(data_test[output].reshape(1,-1)))
print("Clustering with SVR mean error - %f" % (mean_squared_error(target_test,pred,squared=False)))
