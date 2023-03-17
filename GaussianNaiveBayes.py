# A package to implement a Gaussian predictive algorithm.
import pandas as pd
import numpy as np

# df = dataframe with pre-processed data(type=pandas.dataframe)
# target = label of the target attribute(type=string)
# pred = test case values(type=list)
# A package to implement a Gaussian predictive algorithm.
import pandas as pd
import numpy as np

# df = dataframe with pre-processed data(type=pandas.dataframe)
# target = label of the target attribute(type=string)
# pred = test case values(type=list)
class GaussianPrediction:
    def predict(self,X_test):
        y_new = []
        for k in range(len(X_test)):
            i = 0
            final_probs = []
            for value in self.X:
                final_probs.append(self.GaussianProbability(self.X[value],X_test.iloc[k][value]))
            arr = []
            for value in final_probs:
                arr.append(self.inverseGaussianCalc(Y,value))
            y_new.append(np.mean(arr))
        return y_new
            
    def fit(self,X,Y,pred=[]):
        self.X = X
        self.Y = Y

    def multiStandardDeviation(self,df):
        return df.std()

    def multiMeanCalculation(self,df):
        return df.mean()

    def GaussianProbability(self,df,x):
        s = self.multiStandardDeviation(df)
        u = self.multiMeanCalculation(df)
        return (1/np.sqrt(2*np.pi*s*s))*np.e**((-1*((x-u)**2))/(2*(s)**2))

    def inverseGaussianCalc(self,df,prob):
        s = self.multiStandardDeviation(df)
        u = self.multiMeanCalculation(df)
        return (np.log(prob*np.sqrt(2*np.pi*s*s))*-2*s*s)+u
