import numpy as np
import h5py
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from scipy.io import savemat

from scipy.signal import butter, lfilter, spectrogram, welch, iirfilter, filtfilt

from NeorecFiltering import NotchFilter

import pylab

plt.rcParams['figure.figsize'] = [20, 10]

#%%

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandstop_filter(data, lowcut, highcut, fs, order):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y

def filterEMG(MyoChunk,fmin,fmax):
    
    blow, alow = butter_lowpass(cutoff=2, fs=2048, order=5)

    if(fmin > 0):
        bband,aband = butter_bandpass(lowcut=fmin, highcut=fmax, fs=2048, order=3)
    else:
        bband,aband = butter_lowpass(cutoff=fmax, fs=2048, order=3)
        
    MyoChunk = lfilter(bband,aband, MyoChunk.T).T

    np.abs(MyoChunk, out=MyoChunk)
    
    for j in range(MyoChunk.shape[1]):
        MyoChunk[:,j] = lfilter(blow,alow, MyoChunk[:,j])

    return MyoChunk

#%%
    

#workdir='data\\CO_80trials_van_10-01_17-16-52\\'
#workdir='ALL_RAW/CO_80trials_van_10-16_11-20-24/'
workdir='ALL_RAW/CO_80trials_van_10-16_11-27-00/'

datafile='experiment_data.h5'

filepath = workdir + datafile
filepath

# setting parameters we know about the data in advance
srate = 2048

ch_idxs_motor = [18,19,20,21,48,49,50,51]

ch_idxs_ecog = np.arange(52)
ch_idx_photo = 64


# loading the data

with h5py.File(workdir+datafile,'r+') as f1:
    
    #list(f1.keys())[0]
    
    raw_data = np.array(f1['protocol1']['raw_data'])
    posx_data = np.array(f1['protocol1']['posx_data'])
    posy_data = np.array(f1['protocol1']['posy_data'])
    par_data = np.array(f1['protocol1']['par_data'])
    state_data = np.array(f1['protocol1']['state_data'])
    
    
#%%
    
## here we can check that we have correct states for this type of experiment
#states_set = np.unique(state_data)
#states_set = states_set[~np.isnan(states_set)]
#states_set

# for vanilla center-out:
# 0 = ready, wait at the center
# 1 = show target
# 3 = go

# replacing low values from the start of the recording to make plotting more convenient
par_data = np.array([p if p>-100 else -2 for p in par_data]) # to plot
state_data = np.array([p if p>-100 else -2 for p in state_data]) # to plot


#%%

# extracting ecog data and applying notch filter

ecog_data = np.copy(raw_data[:,ch_idxs_ecog])

for i in range(ecog_data.shape[1]):
    for freq in np.arange(50,1000,50):
        ecog_data[:,i] = butter_bandstop_filter(ecog_data[:,i], freq-2, freq+2, srate, 4)

#%%
        
        
# calculate spectrograms to have continuous spectrogram signal to cut trials from later

nperseg = srate*2
noverlap = nperseg*0.99
nfft = nperseg

f_sg, t_sg, Sxx = spectrogram(ecog_data[:,0], fs=srate, nfft=nfft, nperseg=nperseg, noverlap=noverlap) # hack to determine correct Sxx size to allocate sg_data
sg_data = np.zeros([len(ch_idxs_motor),Sxx.shape[0],Sxx.shape[1]])

k = 0
for ch_mot in ch_idxs_motor:
    f_sg, t_sg, Sxx = spectrogram(ecog_data[:,ch_mot], fs=srate, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
    #Sxx, f_sg, t_sg, im = plt.specgram(ecog_data[:,ch_mot], nperseg=nperseg, NFFT=nfft, Fs=srate, noverlap=noverlap)
    
    f_w, Pxx_den = welch(ecog_data[:,ch_mot], fs=srate, nfft=nfft, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    Sxx_norm = Sxx #/Pxx_den[:, None]
    
    sg_data[k,:,:] = Sxx_norm
    k = k+1
    
#plt.rcParams['figure.figsize'] = [10, 3]
#plt.plot(f_w,Pxx_den) #[math.log10(Pxx_den[i]) for i in range(len(Pxx_den))])

#plt.rcParams['figure.figsize'] = [20, 3]
#plt.figure()
#plt.pcolormesh(t[5:500], f[:50], Sxx_norm[:50,5:500])
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
    
    
#%%
    
#    
#fbinmax = 50
#tbinmax = 300
#
#plt.rcParams['figure.figsize'] = [10, 3]
#plt.figure()
##plt.pcolormesh(t_sg[:60], f_sg[:300], sg_data[7][:300,:60])
#plt.pcolormesh(t_sg[:fbinmax], f_sg[:tbinmax], np.log(sg_data[7][:tbinmax,:fbinmax]))
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

#%%
    
# define frequencies to calculate features 
fstep = 4
fbandmins = np.arange(0,200,fstep)
fbandmaxs = fbandmins + fstep
#fbandmins

#%%

feat_data = np.zeros([ecog_data.shape[0],0])

for fsetnum in range(len(fbandmins)):
    
    fbandmin = fbandmins[fsetnum]
    fbandmax = fbandmaxs[fsetnum]
    
    ecog_data_f = filterEMG(ecog_data,fbandmin,fbandmax)
    
    #print(feat_data.shape)
    #print(ecog_data_f.shape)

    feat_data = np.concatenate((feat_data,ecog_data_f),axis=1)

    
scaler = MinMaxScaler([-1, 1])
scaler.fit(feat_data)
feat_data = scaler.transform(feat_data)

# extract and scale photo sensor data to plot along state and par
photo_data = np.copy(raw_data[:,ch_idx_photo])
photo_data = photo_data.reshape(-1,1)
scaler = MinMaxScaler([0, 1])
scaler.fit(photo_data)
photo_data = scaler.transform(photo_data)

con_photo_data = np.convolve(photo_data[:,0], np.ones([20]),mode='same')
diff_con_photo_data = np.diff(con_photo_data)
ph_trigger = np.convolve(diff_con_photo_data, np.ones([20]),mode='same')


#%%

# plot state, par and photo sensor
#sec_to_plot = 60
#
#plt.rcParams['figure.figsize'] = [20, 3]
#
#plt.figure()
#
#plt.plot(ph_trigger[:int(srate)*sec_to_plot])
#plt.plot(state_data[:int(srate)*sec_to_plot])
#
#plt.plot(par_data[:int(srate)*sec_to_plot])

#%%

# find points in time that correspond to go-signal moment captured by photo sensor

srate = int(srate)

start_margin = srate
win = srate//4

photo_points_go = np.unique([np.argmin(ph_trigger[(i-win):(i+win)])+i for i in range(win,len(ph_trigger)) if ph_trigger[i] < -2])
print(photo_points_go)

#%%

print(ecog_data.shape[0])
print(Sxx_norm.shape[1])

print(ecog_data.shape[0]/Sxx_norm.shape[1])

rs_factor = nperseg - noverlap
sg_srate = srate/rs_factor
photo_points_go_RS = photo_points_go/rs_factor

#%%

photo_points_go_sec = photo_points_go/srate
photo_points_go_sg_srate = photo_points_go/rs_factor

#photo_points_go_sg_srate = photo_points_go_sg_srate-nperseg*0.5

win_middles = np.arange(nperseg*0.5,ecog_data.shape[0],rs_factor)


#%%

# plot found points to make sure everything is correct

plt.figure()

plt.plot(ph_trigger)
plt.plot(state_data)

plt.plot(par_data)

for i in range(photo_points_go.shape[0]):
    plt.axvline(photo_points_go[i],color='r')


#%%

# to check time axis is aligned
    
#Sxx_norm = Sxx/Pxx_den[:, None]
#Sxx_norm.shape
#
#plt.rcParams['figure.figsize'] = [20, 3]
#
#plt.pcolormesh(t_sg[:], f_sg[:], Sxx_norm[:,:])
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#
#for i in range(photo_points_go_sec.shape[0]):
#    plt.axvline(photo_points_go_sec[i],color='r')
#    
#plt.show()

#%%
    
T1 = srate*1
T2 = srate*2

trials_idx = [range(i-T1,i+T2) for i in photo_points_go]

trials_idx_sg = [(t_sg > i-1)&(t_sg < i+2) for i in photo_points_go_sec]

# extract trials centered around go-signal
ecog_trials = [ecog_data[p] for p in trials_idx]

feat_trials = [feat_data[p] for p in trials_idx]

sg_trials = [sg_data[:,:,p] for p in trials_idx_sg]

posX_trials = [posx_data[p] for p in trials_idx]
posY_trials = [posy_data[p] for p in trials_idx]

par_trials = [par_data[p] for p in photo_points_go]

#%%

alltrials = [sg_trials[i][:,:200,:63] for i in range(40)]
#alltrials[0].shape

std_across_trials = np.std(np.array(alltrials),axis=0)
#std_across_trials.shape

#te = sg_trials[tr][c,:200,:63]/std_across_trials[c]

#fbinmax = 200
#tbinmax = 63
#
#ch_idxs_motor = [18,19,20,21,48,49,50,51]
#
#for parnum in range(8):
#    
#    #numpartrials = [tr for tr in range(40) if par_trials[tr] == parnum]
#    #partrials = [sg_trials[tr] for tr in numpartrials]
#
#    #std_for_type = np.std(np.array(partrials),axis=0)
#    
#    for c in [1]:#range(len(ch_idxs_motor)):
#        k = 0
#        plt.figure(figsize=(20,10))
#        for tr in range(len(sg_trials)):
#            if(par_trials[tr] == parnum):
#                k = k + 1
#                plt.subplot(1,5,k)
#                plt.pcolormesh(t_sg[:tbinmax]-0.5, f_sg[:fbinmax], sg_trials[tr][c,:fbinmax,:tbinmax]/std_across_trials[c])
#                plt.ylabel('Frequency [Hz]')
#                plt.xlabel('Time [sec]')
#                plt.axvline(1,color='xkcd:aqua')
#        
#    
#        pylab.savefig('par_'+str(parnum)+'_ch_'+str(ch_idxs_motor[c]))
#        
#        #plt.show()
#        #plt.gcf().clear()


#parnum = 0
#numpartrials = [tr for tr in range(40) if par_trials[tr] == parnum]
#partrials = [sg_trials[tr] for tr in numpartrials]
#
#std_for_type = np.std(np.array(partrials),axis=0)


#%% 

# save extracted trials to a mat file

new_matfile = {}

new_matfile['ecog_trials'] = ecog_trials
new_matfile['feat_trials'] = feat_trials
new_matfile['posX_trials'] = posX_trials
new_matfile['posY_trials'] = posY_trials
new_matfile['par_trials'] = par_trials

filepath_save = 'co_van_16_left.mat'

savemat(filepath_save,new_matfile)
