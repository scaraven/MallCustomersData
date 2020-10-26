from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas

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
def plot3DAttr():
    ax = plt.axes(projection="3d")
    ax.scatter3D(pdata[2],pdata[3],pdata[4],cmap="Greens")
    ax.set_zlabel(attributes[4])
    ax.set_xlabel(attributes[2])
    ax.set_ylabel(attributes[3])
    plt.show()

attributes = ["CustomerID","Genre","Age","Annual Income (k$)","Spending Score (1-100)"]
pdata = []
csvfile = pandas.read_csv(".\Mall_Customers.csv")
spendingscore = csvfile['Spending Score (1-100)'].to_numpy()
for x in range(0,len(attributes)):
    arrayfile = csvfile[attributes[x]].to_numpy()
    pdata.append(arrayfile)
