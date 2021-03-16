import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
A=2
lamb=20
w=2*np.pi*100

fs = 8000
T = 1.0
t = np.arange(int(fs*T)) / fs

signal = A**(-lamb*t)*(np.sin(w*t+5)+np.cos(w*t+5))
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)

plt.plot(t, signal, label='signal')
plt.plot(t, amplitude_envelope, label='envelope')
plt.xlabel('t')
plt.legend(framealpha=1, shadow=True)
plt.grid()
plt.show()