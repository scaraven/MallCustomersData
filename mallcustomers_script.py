from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random
from itertools import cycle


##########################################
#Visualising pdata
#draws Graphs for multiple attributes
def visualiseAttr():
    for x in range(0,len(attributes)-1):
        arrayfile = csvfile[attributes[x]].to_numpy()
        pdata.append(arrayfile)
        plt.plot(arrayfile,spendingscore,"ro")
        plt.ylabel("Spending Score (1-100)")
        plt.xlabel(attributes[x])
        plt.show()
    plt.plot(pdata[2],pdata[3],"ro")
    plt.xlabel(attributes[2])
    plt.ylabel(attributes[3])
    plt.show()
def plot3DAttr(xdata,ydata,zdata):
    ax = plt.axes(projection="3d")
    ax.scatter3D(xdata,ydata,zdata,cmap="Greens")
    ax.set_zlabel(attributes[4])
    ax.set_xlabel(attributes[2])
    ax.set_ylabel(attributes[3])
    plt.show()


###############################################
#Preparing data
#Does cross validation with X folds
def Xfoldcrossvalidation(csvfile,foldnum):
    csvfile = remobeObselete(csvfile)
    #split the data into ten samples
    samples = []
    length = len(csvfile.index)
    if(foldnum >= length):
        return csvfile
    while length!=0:

        onefold = pandas.DataFrame(columns=column_names)
        for i in range(foldnum):
            try:
                rand = random.randint(0,length-1)
                onefold = onefold.append(csvfile.iloc[rand])
                csvfile = csvfile.drop(csvfile.index[rand])
                length -=1
            except:
                breakpoint()
        if(length < foldnum):
            for i in range(length):
                onefold.append(csvfile.iloc[i])
                samples.append(onefold)
            return samples
        samples.append(onefold)
def remobeObselete(csvfile):
    #Removes customerID, gender and Spending score
    return csvfile.drop([attributes[0],attributes[1],attributes[4]],axis=1)
def crossValidation():
    trialnum = 10
    samples = Xfoldcrossvalidation(csvfile,trialnum)
    npsamples = []
    for series in samples:
        npsamples.append(series.to_numpy())
    breakpoint()
    for x in range(trialnum):
        #Creates DataFrames with only Age and Annual Income
        training = pandas.DataFrame(columns=column_names)
        test = pandas.DataFrame(columns=column_names)
        if(x > 0):
            training = training.append(npsamples[x:x+trialnum-1]+samples[0:x-1])
        else:
            training = training.append(npsamples[x:x+trialnum-1]+samples[0])
        test = test.append(samples[x-1])
        training = training.to_numpy()
        test = test.to_numpy()
        Clustering(training,test)

###############################################
#Machine learning algorithms


#############################
#Clustering
def Clustering(training,test):
    clusternum = 4
    #default 8 clusters
    kmeans = KMeans(n_clusters=clusternum).fit(training)
    centroids = kmeans.cluster_centers_
    distarray = []
    dist = lambda centr,s1,x : ((centr[x][0]-s1[0])**2 + (centr[x][1]-s1[1])**2)**0.5
    for x in range(len(centroids)):
        distarray.append(dist(centroids,test[0],x))

    #This is a list which keeps holds of all the points sorted into clusters
    clusters = [[[0,0]] for x in range(clusternum)]

    #This loops through every instance in the training data
    for x in range(len(kmeans.labels_)):
        #Find which cluster instance X belongs to
        val = kmeans.labels_[x]

        #Add the instance to its cluster
        if(clusters[val][0][0] == 0 & clusters[val][0][1] == 0):
            clusters[val][0] = training[x]
        else:
            clusters[val].append(training[x])
    #evalClustering(clusters)
    evalClusteringResults(clusters,kmeans,csvfile,test,clusternum)
    CompareSpendingmean(test,kmeans,clusters)
def evalClustering(clusters):
    marker = 0
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for cluster, col in zip(clusters,colors):
        for coord in cluster:
            plt.plot(coord[0],coord[1],col+'.')
        marker+=1
    plt.show()
#Tests how effective our results from the clustring algorithm has been
def evalClusteringResults(clusters,kmeans,csvfile,test,clusternum):
    labels_true = []
    labels_clust = []
    for coord in test:
        spendingscore = lookUpSpending(coord)
        instanceid = int(spendingscore) // (100 / clusternum)
        labels_true.append(instanceid)
        cluster_instanceid = kmeans.predict(coord.reshape(1,-1))[0]
        labels_clust.append(cluster_instanceid)
    print("Metric score - %f" %(metrics.adjusted_rand_score(labels_true, labels_clust)))

    #Function, given age and annual income, will output the spending score from observations



def calcDeviation(mean, instances):
    total = 0
    num = len(instances)
    for x in range(num):
        total += (mean - instances[x])**2
    return (total / num)**0.5
def calculateSpendingMean(clusters,csvfile):
    num = len(clusters)
    total = 0
    for coord in clusters:
        total += csvfile.loc[(csvfile['Age'] == coord[0]) & (csvfile['Annual Income (k$)'] == coord[1])]['Spending Score (1-100)'].iloc[0]
    return total/num

def CompareSpendingmean(test,kmeans,clusters):
    for x in range(len(test)):
        print("Spending mean of - %d belongs to cluster - %s" %(lookUpSpending(test[x]),kmeans.predict(test[x].reshape(1,-1))))
    #Iterates through every single cluster

    for cluster in range(len(clusters)):
        individ = clusters[cluster]
        spendingscore = []
        #Iterates through every point in cluster
        for atr in individ:
            spendingscore.append(lookUpSpending(atr))
        mean = calculateSpendingMean(individ,csvfile)
        deviation = calcDeviation(mean,spendingscore)
        print("Cluster %d has mean %f and deviation %f" %(cluster,mean,deviation))
##############################################
#Main code
attributes = ["CustomerID","Genre","Age","Annual Income (k$)","Spending Score (1-100)"]
pdata = []
lookUpSpending = lambda coord : csvfile.loc[(csvfile['Age'] == coord[0]) & (csvfile['Annual Income (k$)'] == coord[1])]['Spending Score (1-100)'].iloc[0]
column_names = [attributes[2],attributes[3]]
csvfile = pandas.read_csv(".\Mall_Customers.csv")
spendingscore = csvfile['Spending Score (1-100)'].to_numpy()
for x in range(0,len(attributes)):
    arrayfile = csvfile[attributes[x]].to_numpy()
    pdata.append(arrayfile)
crossValidation()
