from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random


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


###############################################
#Machine learning algorithms


#############################
#Clustering
def Clustering(samples,csvfile):
    clusternum = 6

    #Creates DataFrames with only Age and Annual Income
    training = pandas.DataFrame(columns=column_names)
    test = pandas.DataFrame(columns=column_names)
    for i in range(len(samples)-1):

        training = training.append(samples[i])
    test = test.append(samples[-1])
    training = training.to_numpy()
    test = test.to_numpy()
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
    evalClustering(clusters,kmeans,csvfile)
#Tests how effective our clustring algorithm has been
def evalClustering(clusters,kmeans,csvfile):

    #Function, given age and annual income, will output the spending score from observations
    lookUpSpending = lambda coord,csvfile : csvfile.loc[(csvfile['Age'] == coord[0]) & (csvfile['Annual Income (k$)'] == coord[1])]['Spending Score (1-100)'].iloc[0]
    #Iterates through every single cluster
    for cluster in range(len(clusters)):
        individ = clusters[cluster]
        spendingscore = []
        #Iterates through every point in cluster
        for atr in individ:
            spendingscore.append(lookUpSpending(atr,csvfile))
        mean = calculateSpendingMean(individ,csvfile)
        deviation = calcDeviation(mean,spendingscore)
        print("Cluster %d has mean %f and deviation %f" %(cluster,mean,deviation))

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

def testSpendingmean(test,kmeans,csvfile):
    for x in range(len(test)):
        print("Spending mean of - %d belongs to cluster - %s" %(lookUpSpending(test[x], csvfile),kmeans.predict(test[x].reshape(1,-1))))
##############################################
#Main code
attributes = ["CustomerID","Genre","Age","Annual Income (k$)","Spending Score (1-100)"]
pdata = []

column_names = [attributes[2],attributes[3]]
csvfile = pandas.read_csv(".\Mall_Customers.csv")
spendingscore = csvfile['Spending Score (1-100)'].to_numpy()
for x in range(0,len(attributes)):
    arrayfile = csvfile[attributes[x]].to_numpy()
    pdata.append(arrayfile)
samples = Xfoldcrossvalidation(csvfile,10)
#plot3DAttr(csvfile[attributes[2]],csvfile[attributes[3]],csvfile[attributes[4]])
Clustering(samples,csvfile)
