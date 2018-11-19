# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:39:34 2018

@author: voxxys
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn import linear_model


from scipy import stats

from sklearn.metrics import confusion_matrix

import pylab

import math

def imagesc(x):
    plt.imshow(x,extent = [0,1,0,1])

#%matplotlib inline
plt.rcParams['figure.figsize'] = [20, 10]

#%%

# setting parameters we know about the data in advance

srate = 2048

# electrode stips information for plotting

stripborders = [[0,5],[6,11],[12,17],[18,21],[22,27],[28,35],[36,41],[41,47],[48,51]]
striplabels = ['Left Frontal 1','Left Frontal 2','Left Frontal 3','Left Motor', \
              'Right Frontal 1','Right Frontal 2','Right Frontal 3','Right Frontal 4','Right Motor']

strips = [list(range(stripborders[i][0],stripborders[i][1]+1)) for i in range(len(stripborders))]
strips

#%%

# load trials data
name_prefix = 'pol_van_co_16_left_step4herz' # to save figures

matfile_path ='co_van_16_left_step4herz.mat'
matfile = loadmat(matfile_path)

ecog_trials = matfile['ecog_trials']
feat_trials = matfile['feat_trials']
posX_trials = matfile['posX_trials']
posY_trials = matfile['posY_trials']
par_trials = matfile['par_trials'][0]

del matfile

#%%

# choose features that will be analyzed

numch_ecog = ecog_trials[0].shape[1]
fbandmins_len = feat_trials[0].shape[1]//numch_ecog

ch_idxs_motor = [18,19,20,21,48,49,50,51]
#ch_idxs_cihosen = [0]
ch_idxs_chosen = range(numch_ecog) #ch_idxs_motor #[ch_idxs_motor[0]]

ch_idxs_all_chosen = []

for c in ch_idxs_chosen:
    feat_idxs_all = [c + numch_ecog*i for i in range(fbandmins_len)]
    for f in feat_idxs_all:
        ch_idxs_all_chosen.append(f)
        
ch_idxs_all_chosen.sort()
#ch_idxs_all_motor = ch_idxs_all_motor[-16:]

#print(ch_idxs_all_chosen)

#%%

# define moments in time that will be analyzed

step_div = 32

num_w = int(2.5*step_div)

win_starts = np.arange(0,num_w)*srate//step_div
print(win_starts)

win_ends = win_starts + srate//4
print(win_ends)

#%%

# convert trial parameters (position on circle) to angles
par_trials_rad = (math.pi/4)*par_trials
par_trials_rad_sc = np.column_stack([np.sin(par_trials_rad), np.cos(par_trials_rad)])
par_trials_sin = par_trials_rad_sc[:,0]
par_trials_cos = par_trials_rad_sc[:,1]

par_trials_rad_sc_tile = np.tile(par_trials_rad_sc,[512,1]).T.reshape([-1])

#%%

# plant a fake feature encoding angle to test code below
#for tr in range(40):
#    feat_trials[tr][0:srate,17] = np.array([par_trials_sin[tr]]*srate)

#%%

# find features that correlate with trial parameter (angle)

numch_feat = feat_trials.shape[-1] 
res_ch_w = np.zeros([len(ch_idxs_all_chosen),num_w])

chnum = 0

par_trials_rad_sc_tile = np.tile(par_trials_rad_sc,[512,1]).T.reshape([-1])

for ch in ch_idxs_all_chosen:

    #print('channel ', ch)

    for w in range(len(win_starts)):

        feat_trials_red = [feat_trials[i][win_starts[w],ch].T for i in range(len(feat_trials))]
        feat_trials_red = np.array(feat_trials_red)
        #feat_trials_red = [feat_trials[i][win_ends[w],ch].T - \
        #                feat_trials[i][win_starts[w],ch].T for i in range(len(feat_trials))]
        

        '''
        numit = 100
    
        res_true = []
        res_pred = []

        for it in range(numit):
            X_train, X_test, y_train, y_test = train_test_split(feat_trials_red, par_trials_sin, test_size=0.2)

            svr = SVR(kernel='rbf', C=1e2, gamma=0.1)

            pred = svr.fit(X_train, y_train).predict(X_test)

            res_pred.append(pred)
            res_true.append(y_test)

        res_pred_flat = np.array(res_pred).reshape(-1)
        res_true_flat = np.array(res_true).reshape(-1)
        '''
        
        #print(feat_trials_red.shape)
        #print(par_trials_sin.shape)

        slope, intercept, r_value_1, p_value, std_err = stats.linregress(feat_trials_red, par_trials_sin)
        slope, intercept, r_value_2, p_value, std_err = stats.linregress(feat_trials_red, par_trials_cos)
        
        #r2 = r2_score(res_true_flat, res_pred_flat)
        
        r2 = r_value_1**2 #(r_value_1**2 + r_value_2**2)/2    #r_value_1**2 #/(1 - r_value_1**2)
        
        
        #print("r-squared:", round(r2,2))
        res_ch_w[chnum,w] = r2
        
        
        #print(res_true_flat)
        #print(res_pred_flat)

    chnum = chnum + 1
    

#%%
    
res_ch_w_re = res_ch_w.reshape([len(ch_idxs_chosen),fbandmins_len,num_w],order='F')
#res_ch_w_re.shape

res_ch_w_re_maxtime = np.max(res_ch_w_re,axis=-1)


fig, axes = plt.subplots(nrows=len(striplabels), ncols=1, figsize=(15,20))

s = 0
for ax in axes.flat:
    figheight = len(strips[s])/2
    im = ax.imshow(res_ch_w_re_maxtime[strips[s],:],extent = [0,5,0,0.1*figheight],vmin=0,vmax=0.9)
    
    s = s + 1
    
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

fig.colorbar(im, cax=cbar_ax)

plt.show()

