import scipy.io
import numpy as np
from bf import bf 
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt

from scipy import signal


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# ***************************************************** Variables  *****************************************************
pitch = 3e-04
ne = 32
x_channels = (np.arange(0,ne)*pitch) - (ne-1)*pitch/2  # might need to address this
print(x_channels)

# x_axis = (np.arange(0,ne+2)*pitch/2) - (ne+2-1)*pitch/2/2 # might need to address this
# print(x_axis)


x_axis = (np.arange(0,2*ne)*pitch/2) - (2*ne-1)*pitch/2/2 # might need to address this
print(x_axis)
scale = 5
z_axis = np.arange(0,701*scale)*5e-5/scale

f_resample = 20e6
c = 1540
tstart = 0
terror = 0

offset = -min(x_channels)
# offset = 0
x_channels = x_channels + offset
x_axis = x_axis + offset

mask =np.zeros((z_axis.shape[0],x_axis.shape[0],ne))
f_num = 1
ii = 0
for i in x_axis:
    a = z_axis/2*f_num
    # This give negative indexs!!!
    start = np.floor((i - a)/pitch ).astype(int)
    end = np.ceil((i + a)/pitch ).astype(int)
    start[ start< 0] = 0
    end[ end > ne-1] = ne-1
    for j in range(0,len(z_axis)):
        mask[j,ii,start[j]:end[j] ] = 1
    ii = ii+1

# ***************************************************** Import data  *****************************************************
from os import listdir
from os.path import isfile, join
import time

file_name= 'Mice_8_B'
Path = '../Clean/UserSessions/'+ file_name +'/RFArrays/'

files = listdir(Path)

Y = np.zeros((z_axis.shape[0],x_axis.shape[0]))

t0 = time.perf_counter()
for file in files:
    tt0 = time.perf_counter()
    fileName = str(file).replace(".npy","")
    print()
    if 'F' in fileName:
        pass
    else:
        fileNameParts = fileName.replace(",", ".").split("_")
        angle = float(fileNameParts[2])
        
        if angle in [ -10, -5, -2, 0, 2, 5,10 ]:
        # if True:
            print("filename: " + fileName, "Angle : " , angle)
            X = np.load(Path +file)

# ***************************************************** BF and compound  ***************************************************
            print(angle)
            data =  butter_highpass_filter(X.T,1*1e6,20*1e6,order =5).T
            print(data.shape)
            y = bf(data,x_axis,z_axis,x_channels,ne,angle,c,terror,tstart,f_resample,mask)
            Y = Y + y

np.savetxt("outputy.csv", Y, delimiter=",")
img = hilbert(Y.T).T  #Mother fucker
print(np.amax(img))
img= img/np.amax(img)
img = np.abs(img)
print(img.shape)

fig, ax = plt.subplots(1,1,figsize=(4,4), dpi=100)

img_log = 20*np.log10(img)
Image = plt.imshow(img_log,cmap='gray',interpolation='none', extent=[x_axis[0],x_axis[-1],z_axis[-1],z_axis[0]], aspect=1)
# Image = plt.imshow(20*np.log10(img),cmap='gray',interpolation='none',  aspect=1)
# ax.set_xlim(-0.005, 0.005)
# ax.set_ylim(18e-3,7e-3)
Image.set_clim(vmin=-60, vmax=0)
print(np.amax(img_log))

np.savetxt("imgPython.csv", img, delimiter=",")
plt.show()



print(x_channels)
print(x_axis)