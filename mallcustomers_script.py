from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random
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
    """ax.set_zlabel(attributes[4])
    ax.set_xlabel(attributes[2])
    ax.set_ylabel(attributes[3])
    plt.show()"""

#Does cross validation with X folds
def Xfoldcrossvalidation(csvfile,foldnum):
    csvfile = remobeObselete(csvfile)
    #split the data into ten samples
    samples = []
    length = len(csvfile.index)
    if(foldnum >= length):
        return csvfile
    while length!=0:

        onefold = pandas.DataFrame(columns=[attributes[2],attributes[3],attributes[4]])
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
    return csvfile.drop([attributes[0],attributes[1]],axis=1)
def Clustering(samples):

    training = pandas.DataFrame(columns=[attributes[2],attributes[3],attributes[4]])
    test = pandas.DataFrame(columns=[attributes[2],attributes[3],attributes[4]])
    for i in range(len(samples)-1):

        training = training.append(samples[i])
    test = test.append(samples[-1])
    training = training.to_numpy()
    test = test.to_numpy()
    kmeans = KMeans().fit(training)
    centroids = kmeans.cluster_centers_
attributes = ["CustomerID","Genre","Age","Annual Income (k$)","Spending Score (1-100)"]
pdata = []
csvfile = pandas.read_csv(".\Mall_Customers.csv")
spendingscore = csvfile['Spending Score (1-100)'].to_numpy()
for x in range(0,len(attributes)):
    arrayfile = csvfile[attributes[x]].to_numpy()
    pdata.append(arrayfile)
samples = Xfoldcrossvalidation(csvfile,10)
Clustering(samples)
