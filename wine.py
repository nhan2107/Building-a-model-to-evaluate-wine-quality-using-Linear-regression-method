import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ols_linear_regression(A, b):
    A_pinv = np.linalg.inv(A.T @ A) @ (A.T @ b)
    x = np.linalg.norm(A @ A_pinv - b)
    return x

def LinearRegression(A, b):
    theta = np.linalg.inv(A.T @ A) @ (A.T @ b)
    y = "y = "
    for i in range(11):
        if theta[i] >= 0:
            if i == 0:
                y += str(theta[i]) + " * x1"
            else:
                y += " + " + str(theta[i]) + " * x" + str(i + 1)
        else:
            if i == 0:
                y += " - " + str(-theta[i]) + " * x1"
            else:
                y += " - " + str(-theta[i]) + " * x" + str(i + 1)
    return y

def Best_standard_molecular_vector(A,b):
    y = np.empty(11)
    for i in range(11):
        x = A.values[:,i].reshape((-1,1))
        x = np.concatenate((np.ones((len(df),1)), x.reshape(-1,1)), axis=1)
        m = ols_linear_regression(x, b)
        y[i] = m
    print("Standard molecular vector each feature: ", y)
    return np.min(y)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    df = pd.read_csv('wine.csv', sep=';')
    A = np.empty((len(df),11))
    for i in range(11):
        A[:,i] = df.values[:, i]
    b = df.values[:,11]
    y = LinearRegression(A, b)
    print("Linear regression expression: " + y)
    print("standard molecular vector: ", ols_linear_regression(A, b))
    print("Chuan vector phan tu cua dac trung tot nhat: ",Best_standard_molecular_vector(df,b))