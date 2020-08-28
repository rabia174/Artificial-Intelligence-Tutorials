# importing dependencies
import numpy as np
import matplotlib.pyplot as plt




# linear regression using gradient descent


# function to compute hypothesis / predictions
def hypothesis(X, theta):
    return np.dot(X, theta)
# function to compute gradient of error function w.r.t. theta
def gradient(X, y, theta):
    h = hypothesis(X, theta)
    grad = 2*np.dot(X.transpose(), (h - y))/X.shape[0]
    return grad
# function to compute the error for current values of theta
def cost(X, y, theta):
    h = hypothesis(X, theta)
    J = np.dot((h - y).transpose(), (h - y))**2
    J /= X.shape[0]
    return J[0]
# function to perform gradient descent
def gradientDescent(X, y, learning_rate=0.01):
    # theta = np.zeros((X.shape[1], 1))
    theta = np.array([-2,5]).reshape((X.shape[1], 1))

    error_list = []
    max_iters = 500
    for itr in range(max_iters):
        theta = theta - learning_rate * gradient(X, y, theta)
        error_list.append(cost(X, y, theta))
    return theta, error_list

data = np.loadtxt('GD_Example.txt')

# visualising data
plt.scatter(data[:, 0], data[:, 1], marker='.')
plt.show()

# Adding a column of ones to create a vector X that will be multiplied by the two parameters m and b as b+mx = X(dot)theta where (dot) is the matrix multiplication, X=[1,x] and theta = [b,m]
data = np.hstack((np.ones((data.shape[0], 1)), data))

# Split data into training and validation data
split_factor = 0.90
split = int(split_factor * data.shape[0])
X_train = data[:split, :-1]
y_train = data[:split, -1].reshape((-1, 1))
X_test = data[split:, :-1]
y_test = data[split:, -1].reshape((-1, 1))

# Call gradient descent method
theta, error_list = gradientDescent(X_train, y_train)

# Print results
print("b = ", theta[0])
print("m = ", theta[1:])

# visualising gradient descent optimization process
plt.close()
plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()

# predicting output for X_test
y_pred = hypothesis(X_test, theta)
plt.scatter(X_test[:, 1], y_test[:, ], marker='.')
plt.plot(X_test[:, 1], y_pred, color='orange')
plt.show()

# calculating error in predictions
error = np.sum(np.abs(y_test - y_pred) / y_test.shape[0])
print("Mean absolute error = ", error)