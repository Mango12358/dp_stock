import numpy as np
import pandas as pd


class PastSampler:
    '''
    Forms training samples for predicting future values from past value
    '''

    def __init__(self, N, K, sliding_window=True):
        '''
        Predict K future sample using N previous samples
        '''
        self.K = K
        self.N = N
        self.sliding_window = sliding_window

    def transform(self, A):
        M = self.N + self.K  # Number of samples per row (sample + target)
        # indexes
        if self.sliding_window:
            # setp K
            if (A.shape[0] - M) % self.K == 0:
                I = np.arange(M) + np.arange(0, A.shape[0] - M + 1, self.K).reshape(-1, 1)
            else:
                I = np.arange(M) + np.arange(0, A.shape[0] - M + (A.shape[0] - M) % self.K + 1, self.K).reshape(-1, 1)
        else:
            I = np.arange(M) + np.arange(0, A.shape[0] - M + 1, 1).reshape(-1, 1)
        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]  # Number of features per sample
        return B[:, :ci], B[:, ci:]  # Sample matrix, Target matrix


import json
import os

class ChartsJsonParser:
    '''
    Echarts Json Data Parser
    '''

    def __init__(self, T, R, P):
        '''
        T: Timestamp
        R: Real Data
        P: Predict Data
        '''
        self.T = T
        self.R = R
        self.P = P
        self.model = {'title':{'text':'Print_Result'},'tooltip':{'trigger':'axis'},'legend':{'data':['Real','Predict']},'grid':{'left':'3%','right':'4%','bottom':'3%','containLabel':True},'toolbox':{'feature':{'saveAsImage':{}}},'xAxis':{'type':'category','boundaryGap':False,'data':[]},'yAxis':{'type':'value'},'series':[{'name':'Real','type':'line','stack':'Real','data':[]},{'name':'Predict','type':'line','stack':'Pre','data':[]}]}
        self.model_json = self.model

    def get_data(self):
        self.model_json['xAxis']['data'] = self.T
        self.model_json['series'][0]['data'] = self.R
        self.model_json['series'][1]['data'] = self.P

        return json.dumps(self.model_json)

    def save_data(self,file):
        self.model_json['xAxis']['data'] = self.T
        self.model_json['series'][0]['data'] = self.R
        self.model_json['series'][1]['data'] = self.P

        try:
            f = open(file,'w+')
            f.write(json.dumps(self.model_json))
        finally:
            if f:
                f.close()

        return json.dumps(self.model_json)
