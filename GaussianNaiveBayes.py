# A package to implement a Gaussian predictive algorithm.
import pandas as pd
import numpy as np

# df = dataframe with pre-processed data(type=pandas.dataframe)
# target = label of the target attribute(type=string)
# pred = test case values(type=list)
class GaussianPrediction:
    def GaussianPredictor(self,df,target,pred=[]):
        # All error cases handling
        if(not (type(df))==pd.core.frame.DataFrame):
            print('ERROR: Argument \'df\' must be of type pandas.core.frame.DataFrame')
            exit()
        if(not type(target)==str):
            print('ERROR: argument \'target\' must be of type str')
            exit()
        if(not target in df.columns):
            print('ERROR: target: \''+target+'\' not in dataframe')
            exit()
        if(len(df.columns)-1 != len(pred)):
            print('ERROR: Incorrect size of \'pred\'')
            exit()
        df_vars = df.drop(target,axis=1)
        df_target = df[target]
        i = 0
        final_probs = []
        for value in df_vars:
            final_probs.append(self.GaussianProbability(df_vars[value],pred[i]))
            i = i+1
        arr = []
        for value in final_probs:
            arr.append(self.inverseGaussianCalc(df_target,value))
        print("Prediction: ")
        print(np.mean(arr))
        print("Probability: ")
        print(np.mean(final_probs)*100)

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
        return np.sqrt((np.log(prob*np.sqrt(2*np.pi*s*s))*-2*s*s))+u

