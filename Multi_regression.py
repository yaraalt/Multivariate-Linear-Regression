import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing as pp



# find hypothesis
def calculateHypothesis(x, theta):
    hypothesis = np.dot(x, theta)
    return hypothesis



# MSE cost function
def calculateCost(x, y, theta):
    m= len(y)
    hypothesis= calculateHypothesis(x, theta)
    cost= (sum((hypothesis - y)**2))/(2*m)
    return cost



# gradient descent function
def gradientDescent(x, y, theta, alpha):
    cost= []
    tempTheta= []
    m= len(y)
    iterations= 4000
    columns= np.shape(x)[1]

    for i in range(iterations):
        hypothesis = calculateHypothesis(x, theta)
        cost.append(calculateCost(x, y, theta))
        for j in range(columns):
            theta[j]= theta[j] - alpha*(1/m)*(sum((hypothesis - y)*x[:,j]))

    return cost, theta



# Plotting
def plot(iterations, cost, title):
    plt.plot(list(range(0, iterations)), cost, color= "blue")
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()



# loading dataset
training_data= pd.read_csv("Admission_Predict_Ver1.1.csv")
training_data= training_data.dropna()

x= training_data.iloc[:, 1:8].values
y= training_data.iloc[:, 8].values

ones= [1 for i in range(len(y))]


# normalizing features then adding column of ones for x0
x_normalized= pp.normalize(x)
x_normalized= np.column_stack((ones,x_normalized))


# adding column of ones for x0 (non-normalized features)
x= np.column_stack((ones,x))


# defining theta
columns= np.shape(x)[1]
theta= np.array([0]*columns)
theta= theta.astype(np.float64)

iterations= 4000
costsNot= []
costsNormalized= []
cost= []


# Plotting
# Normalized
# alpha= 0.1
cost, results= gradientDescent(x_normalized, y, theta, 0.1)
costsNormalized.append(cost[-1])
print("The theta value for normalized and alpha= 0.1 is: ", results)
plot(iterations,cost,"Normalized and alpha= 0.1")


# alpha= 0.001
theta= np.array([0]*columns)
theta= theta.astype(np.float64)
cost, results= gradientDescent(x_normalized, y, theta, 0.001)
costsNormalized.append(cost[-1])
print("The theta value for normalized and alpha= 0.001 is: ", results)
plot(iterations,cost,"Normalized and alpha= 0.001")


# alpha= 0.00001
theta= np.array([0]*columns)
theta= theta.astype(np.float64)
cost, results= gradientDescent(x_normalized, y, theta, 0.00001)
costsNormalized.append(cost[-1])
print("The theta value for normalized and alpha= 0.00001 is: ", results)
plt.plot(list(range(0, iterations)), cost)
plot(iterations,cost,"Normalized and alpha= 0.00001")


plt.scatter([0.1, 0.001, 0.00001], costsNormalized, color= "blue")
plt.title("Cost against alpha (normalized)")
plt.xlabel("Alpha")
plt.ylabel("Cost")
plt.show()



# Not normalized
# alpha= 0.1
theta= np.array([0]*columns)
theta= theta.astype(np.float64)
cost, results= gradientDescent(x, y, theta, 0.1)
costsNot.append(cost[-1])
print("\n\nThe theta value for not normalized and alpha= 0.1 is: ", results)
plot(iterations, cost, "Not normalized and alpha= 0.1")


# alpha= 0.001
theta= np.array([0]*columns)
theta= theta.astype(np.float64)
cost, results= gradientDescent(x, y, theta, 0.001)
costsNot.append(cost[-1])
print("The theta value for not normalized and alpha= 0.001 is: ", results)
plot(iterations,cost,"Not normalized and alpha= 0.001")


# alpha= 0.00001
theta= np.array([0]*columns)
theta= theta.astype(np.float64)
cost, results= gradientDescent(x, y, theta, 0.00001)
costsNot.append(cost[-1])
print("The theta value for not normalized and alpha= 0.00001 is: ", results)
plot(iterations, cost, "Not normalized and alpha= 0.00001")


plt.scatter([0.1, 0.001, 0.00001], costsNot, color= "blue")
plt.title("Cost against alpha (not normalized)")
plt.xlabel("Alpha")
plt.ylabel("Cost")
plt.show()