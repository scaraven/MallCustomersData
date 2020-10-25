from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas

attributes = ["CustomerID","Genre","Age","Annual Income (k$)","Spending Score (1-100)"]
pdata = []
#performing linear regression
regr = linear_model.LinearRegression()
csvfile = pandas.read_csv(".\Mall_Customers.csv")
spendingscore = csvfile['Spending Score (1-100)'].to_numpy()

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
