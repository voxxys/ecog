# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:37:29 2018

@author: voxxys
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn import svm

from sklearn.metrics import confusion_matrix

import pylab

def imagesc(x):
    plt.imshow(x,extent = [0,1,0,1])

#%matplotlib inline
plt.rcParams['figure.figsize'] = [20, 10]


#%%

name_prefix = 'pol_van_co_16_left_' # to save figures

matfile_path ='co_van_16_left.mat'
matfile = loadmat(matfile_path)

ecog_trials = matfile['ecog_trials']
feat_trials = matfile['feat_trials']
posX_trials = matfile['posX_trials']
posY_trials = matfile['posY_trials']
par_trials = matfile['par_trials'][0]

del matfile


#%%

srate = 2048

numch_ecog = ecog_trials[0].shape[1]
fbandmins_len = feat_trials[0].shape[1]//numch_ecog

ch_idxs_motor = [18,19,20,21,48,49,50,51]

ch_idxs_all_motor = []

for c in ch_idxs_motor:
    feat_idxs_all = [c + numch_ecog*i for i in range(fbandmins_len)]
    for f in feat_idxs_all:
        ch_idxs_all_motor.append(f)
ch_idxs_all_motor.sort()
ch_idxs_all_motor = ch_idxs_all_motor[-16:]
print(ch_idxs_all_motor)


#%%

step_div = 64

num_w = int(2.5*step_div)

win_starts = np.arange(num_w)*srate//step_div
print(win_starts)

win_ends = win_starts + srate//4
print(win_ends)


cms = np.zeros([num_w,8,8])


#%%

for w in range(len(win_starts)):

    feat_trials_red = [feat_trials[i][win_starts[w]:win_ends[w]:200,ch_idxs_all_motor].T for i in range(len(feat_trials))]

    feat_trials_red_flat = [feat_trials_red[i].reshape(-1) for i in range(len(feat_trials))]
    feat_trials_red_flat = np.array(feat_trials_red_flat)


    # find optimal C
    Cc_list = [50, 100, 500, 1000, 2000, 5000]

    Cc_scores = np.zeros(len(Cc_list))

    numit = 500

    k = 0
    for Cc in Cc_list:

        scores = np.zeros(numit)

        res_true = []
        res_pred = []

        for it in range(500):
            X_train, X_test, y_train, y_test = train_test_split(feat_trials_red_flat, par_trials, test_size=0.2)

            clf =  svm.SVC(kernel='rbf', C = Cc, shrinking = True).fit(X_train, y_train)

            scores[it] = clf.score(X_test, y_test)

            pred = clf.predict(X_test)
            res_pred.append(pred)
            res_true.append(y_test)

        #plt.hist(scores)

        #print(Cc)
        #print(np.mean(scores))
        Cc_scores[k] = np.mean(scores)

        k = k+1


    # use best C to build the classifier
    best_Cc_idx = np.argmax(Cc_scores)

    Cc = Cc_list[best_Cc_idx]

    scores = np.zeros(numit)

    res_true = []
    res_pred = []

    for it in range(500):
        X_train, X_test, y_train, y_test = train_test_split(feat_trials_red_flat, par_trials, test_size=0.2, stratify=par_trials)

        clf =  svm.SVC(kernel='rbf', C = Cc, shrinking = True).fit(X_train, y_train)

        scores[it] = clf.score(X_test, y_test)

        pred = clf.predict(X_test)
        res_pred.append(pred)
        res_true.append(y_test)

    print(Cc)
    print(np.mean(scores))
    
    
    # draw and save confusion matrix
    res_pred_flat = np.array(res_pred).reshape(-1)
    res_true_flat = np.array(res_true).reshape(-1)
    
    cm = confusion_matrix(res_true_flat,res_pred_flat)
    cms[w,:,:] = cm
    #plt.rcParams['figure.figsize'] = [10, 5]
    #plt.figure
    #imagesc(cm)
    #pylab.savefig(name_prefix+'cm_'+str(w))
    #plt.gcf().clear()
    
    print('w = ')
    print(w)
    #print(cm)
    print(np.diag(cm)/np.sum(cm,axis=1))
    

    fig = plt.figure(figsize=(20,5))
    rect = fig.patch
    plt.gcf().clear()

    for p in range(8):
        ax = plt.subplot(1,8,p+1, polar=True)

        angles = [n*3.14/4 for n in range(8)]
        angles += angles[:1]

        cm_to_plot = np.concatenate((cm[p,:],[cm[p,0]]),axis=0)
        cm_to_plot.shape

        plt.plot(angles,cm_to_plot)
        plt.plot([angles[p],angles[p]],[0,200],'m')

        ax.set_rticks([]) 
        
        ms_into_trial = w*srate//step_div
        
        
        win_flash = 128
        if(ms_into_trial > 2048-win_flash)and(ms_into_trial < 2048+win_flash):
            ax.set_facecolor('xkcd:mint green')
        else:
            ax.set_facecolor('white')
            
    if(ms_into_trial > 2048):
        plt.suptitle(str(ms_into_trial-2048)+' ms since GO-signal')
    else:
        plt.suptitle(str(2048-ms_into_trial)+' ms to GO-signal')
    
    pylab.savefig(name_prefix+'polar_'+str(w))
    plt.gcf().clear()
    
    
#%%
    
acc = [np.diag(cms[i])/np.sum(cms[i],axis=1) for i in range(num_w)]

acc_np = np.array(acc)
acc_np.shape

plt.imshow(acc_np.T,extent = [0,160,0,80])
plt.xticks(np.arange(0, 160, step=10), win_starts[::10])
plt.axvline(2048//32,color='xkcd:aqua')
plt.colorbar()




