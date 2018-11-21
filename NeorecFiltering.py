# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:19:53 2018

@author: Александр
"""
import numpy as np
from scipy.signal import lfilter



 
class NotchFilter():    
    def __init__(self):
        self.init=0
        
    def initParams(self, f0=50, fs=2048, n_channels=64, mu=0.05):
        self.n_channels = n_channels
        w0 = 2*np.pi*f0/fs
        self.a = np.array([1., 2 * (mu - 1) * np.cos(w0), (1 - 2 * mu)])
        self.b = np.array([1., -2 * np.cos(w0), 1.]) * (1 - mu)
        self.zi = np.zeros((max(len(self.b), len(self.a)) - 1, n_channels))
        self.init=1
    
    def CommonAverage(self, chunk):
        avg=np.mean(chunk,1)[:,None]
        chunk-=avg
        return chunk
    
    def apply(self, chunk):
        if(not self.init):
            self.initParams(n_channels=chunk.shape[1])
        y, self.zi = lfilter(self.b, self.a, chunk, axis=0, zi=self.zi)
        return y
    
    def reset(self):
        self.zi = np.zeros((max(len(self.b), len(self.a)) - 1, self.n_channels))
        self.init=0

