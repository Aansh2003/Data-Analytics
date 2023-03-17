import numpy as np
from scipy import stats

class KNearestNeighbours:
    def __init__(self,k,regression=True):
        self.k = k
        self.regression = regression

    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
    def predict(self,X_test):
        y_pred = []
        for i in range(X_test.shape[0]):
            distances = []
            for j in range(self.X_train.shape[0]):
                d =(np.sum(abs(X_test.iloc[i,:] - self.X_train.iloc[j,:])))
                distances.append((d, self.Y_train[j])) 
            distances = sorted(distances)
            
            neighbors = []
            for item in range(self.k):
                neighbors.append(distances[item][1])
            if self.regression:
                y_pred.append(np.mean(neighbors))
            else:
                y_pred.append(stats.mode(neighbors)[0][0])
        return y_pred
