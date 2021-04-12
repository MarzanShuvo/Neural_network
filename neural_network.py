import numpy as np


class NeuralNetwork():
    def __init__(self, learning_rate, X, Y):
        self.X = X
        self.Y = Y
        self.w1 = np.random.rand(4, self.X.shape[0])*0.01
        self.b1 = np.zeros(shape=(4,1))
        self.w2 = np.random.randn(self.Y.shape[0], 4)*0.01
        self.b2 = np.zeros(shape =(self.Y.shape[0], 1))
        self.learning_rate = learning_rate
        self.a1 = 0
        self.a2 = 0


    def relu(self, z):
        return np.maximum(z, 0)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def feedforward(self):
        z1= np.dot(self.w1, self.X) + self.b1
        self.a1= np.tanh(z1)
        z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = self.sigmoid(z2)

    def compute_cost(self):
        m = self.Y.shape[1]
        cost_sum = np.multiply(np.log(self.a2), self.Y) + np.multiply((1 - self.Y), np.log(1 - self.a2))
        cost = - np.sum(cost_sum) / m
        
        cost = np.squeeze(cost)
        return cost

        

    def back_propagate(self):
        m = self.Y.shape[1]
        dZ2 = self.a2 - self.Y
        dW2 = (1 / m) * np.dot(dZ2, self.a2.T)
        db2 = (1 / m) * np.sum(dZ2, axis = 1, keepdims = True)
    
        dZ1 = np.multiply(np.dot(self.w2.T, dZ2), 1 - np.power(self.a1, 2))
        dW1 = (1 / m) * np.dot(dZ1, self.X.T)
        db1 = (1 / m) * np.sum(dZ1, axis = 1, keepdims = True)
        
        # Updating the parameters according to algorithm
        self.w1 = self.w1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1
        self.w2 = self.w2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2
        

    def training(self):
        for i in range(0, 1000):
    
            # Forward propagation. Inputs: "X, parameters". return: "A2, cache".
            self.feedforward()
            
            # Cost function. Inputs: "A2, Y". Outputs: "cost".
            cost = self.compute_cost()
    
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            self.back_propagate()

            print_cost = True
            
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration % i: % f" % (i, cost))

        
def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y


X, Y = load_planar_dataset()
a = NeuralNetwork(0.5, X, Y)
a.training()




