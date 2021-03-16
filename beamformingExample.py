import scipy.io
import numpy as np
from bf import bf 
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt

# ***************************************************** Variables  *****************************************************
pitch = 3e-04
ne = 64
x_channels = (np.arange(0,ne)*pitch) - (ne-1)*pitch/2  # might need to address this
print(x_channels)

# x_axis = (np.arange(0,ne+2)*pitch/2) - (ne+2-1)*pitch/2/2 # might need to address this
# print(x_axis)

x_axis = (np.arange(0,2*ne)*pitch/2) - (2*ne-1)*pitch/2/2 # might need to address this
print(x_axis)

z_axis = np.arange(0,401)*5e-5
theta = [-5,0,5]
f_resample = 25000000
c = 1540
tstart = [1.47e-05,	1.496e-05,	1.469e-05]
terror = 6.4939e-07

mask =np.zeros((z_axis.shape[0],x_axis.shape[0],ne))
f_num = 1
ii = 0
for i in x_axis:
    a = z_axis/2*f_num
    # This give negative indexs!!!
    start = np.floor((i - a)/pitch +ne/2).astype(int)
    end = np.ceil((i + a)/pitch +ne/2).astype(int)
    start[ start< 0] = 0
    end[ end > ne-1] = ne-1
    for j in range(0,len(z_axis)):
        mask[j,ii,start[j]:end[j] ] = 1
    ii = ii+1

# ***************************************************** Import data  *****************************************************

Matlab_struct = scipy.io.loadmat('rfData/rfData_1.mat')

rf_data = Matlab_struct['rf_data2']
na = rf_data.shape[1]
print(na)
# data = np.asarray(rf_data[0][2][0])
# print(data, data.shape)

# ***************************************************** BF and compound  ***************************************************

Y = np.zeros((z_axis.shape[0],x_axis.shape[0]))
for i in range(0,na):
    print(i)
    data = np.asarray(rf_data[0][i][0],dtype=np.float64)
    y = bf(data,x_axis,z_axis,x_channels,ne,theta[i],c,terror,tstart[i],f_resample,mask)
    Y = Y + y

np.savetxt("outputy.csv", Y, delimiter=",")
img = hilbert(Y.T).T  #Mother fucker
print(np.amax(img))
img= img/np.amax(img)
img = np.abs(img)
print(img.shape)

fig, ax = plt.subplots(1,1,figsize=(4,4), dpi=100)

Image = plt.imshow(20*np.log10(img),cmap='gray',interpolation='none', extent=[x_axis[0],x_axis[-1],z_axis[-1],z_axis[0]], aspect=1)
ax.set_xlim(-0.005, 0.005)
ax.set_ylim(18e-3,7e-3)
Image.set_clim(vmin=-40, vmax=0)
    # figure, imagesc(x_axis, z_axis,);colormap('gray');caxis([-40 0]);axis image
    # ylim([7e-3 18e-3]);
    # xlim([-0.005 0.005]);
np.savetxt("imgPython.csv", img, delimiter=",")
plt.show()