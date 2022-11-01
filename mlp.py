# Author :- Gaurav Kumar
# Roll No. :- 20074013

import numpy as np
import pandas as pd

class mlp:
    def __init__(self, m, n, p, eta, epoaches):
        self.m = m
        self.n = n
        self.p = p

        self.w1 = np.zeros((n+1, m+1))
        self.w2 = np.zeros((p+1, n+1))
        self.v = np.zeros(n+1)
        self.y = np.zeros(n+1)
        self.z = np.zeros(p+1)
        self.o = np.zeros(p+1)

        self.eta = eta
        self.epoaches = epoaches

    def sigmoid(self, y):
        return 1/(1+np.exp(-y))

    def forward_prop(self, inp=np.ones(1)):
        # inp: np arra y of size (m+1, ) , one biased imp
        self.v = np.dot(self.w1, inp)
        self.y = self.sigmoid(self.v)
        self.y[0] = 1

        self.z = np.dot(self.w2, self.y)
        self.o = self.sigmoid(self.z)

        return self.o
    
    def back_prop(self, d = np.ones(1), inp = np.ones(1)):
        for k in range(1, self.p+1):
            for j in range(self.n+1):
                self.w2[k][j] += self.eta*(d[k]-self.o[k])*self.o[k]*(1-self.o[k])*self.y[j]
        
        for j in range(self.n+1):
            for i in range(self.m+1):
                val = 0
                for k in range(1, self.p+1):
                    val += (d[k]-self.o[k])*self.o[k]*(1-self.o[k])*self.w2[k][j]

                self.w1[j][i] += self.eta*val*self.y[j]*(1-self.y[j])*inp[i]

    def train(self, X= np.ones((150,4)), Y= np.ones((150))):
        patterns = X.shape[0]

        d_matrix = np.zeros((patterns, self.p+1))
        for i in range(patterns):
            d_matrix[i][Y[i]+1] = 1
        
        X = np.c_[np.ones(patterns), X]

        #epoaches
        for i in range(self.epoaches):
            print(f"epoach {i}: ")
            for i in range(patterns):
                self.forward_prop(X[i])
                self.back_prop(d_matrix[i], X[i])
                # print(i, end="->")

        output = np.zeros(d_matrix.shape)
        # print(d_matrix.shape)
        for i in range(patterns):
            output[i] = self.forward_prop(X[i])
            output[i][0] = 0
        
        # for i in range(patterns):
        predicted = output.argmax(axis=1)
        desired = d_matrix.argmax(axis=1)
        accuracy = 0
        for i in range(patterns):
            if predicted[i]==desired[i]:
                accuracy+=1
        
        accuracy = (accuracy/patterns)*100.0
        print(accuracy)

df = pd.read_csv("iris.csv",sep=',')
df.iloc[:,4] = pd.Categorical(df.iloc[:,4])
df.iloc[:,4] = df.iloc[:,4].cat.codes

#shuffling the data
df = df.sample(frac=1)

X = np.array(df.iloc[:,:4])
Y = np.array(df.iloc[:,4])

n = mlp(4,2,3,0.01, 1000)
n.train(X, Y)
