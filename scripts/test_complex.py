import cmath
import math
import random
import numpy as np
PI = np.pi
import matplotlib.pyplot as plt

def remove_linear_trend(phi, t):
    t = t[0:len(phi)]
    t = range(len(phi))
    p = np.poly1d(np.polyfit(t,phi,1)) #linearly fit the instaneous phase
    estimated = p(t) #re-evaluate the offset term using the fitted values                         
    demodulated = phi - estimated
    return demodulated

def rms(x):
    rms = np.sqrt(np.mean(x**2))
    return rms

x = np.random.randint(-4,5, size=(5,8))
fc = 240 #carrier frequency
fc2 = 100
fm = 10 #frequency of modulating signal
alpha = 1 #amplitude of modulating signal
theta = 0 # PI/4 #phase offset of modulating signal
beta = PI/5 #constant carrier phase offset 
receiverKnowsCarrier= False; #If receiver knows the carrier frequency & phase offset

fs = 8*fc #sampling frequency
duration = 0.5 #duration of the signal
t = np.arange(int(fs*duration)) / fs #time base

#Phase Modulation
phi_error = 2*PI*fm*t + theta
m_t = alpha*np.sin(phi_error) #modulating signal
phi = 2*PI*fc*t + beta + m_t
az1 = np.cos( phi ) + 1j * np.sin(phi) #modulated signal
phi = 2*PI*fc2*t + beta + m_t
az2 = np.cos( phi ) + 1j * np.sin(phi) #modulated signal





G = np.zeros((len(az1), 2)) + 1j

G[:, 0] = az1[:]
G[:, 1] = az2[:]


###RITSAR

G_dot = np.diff(G, axis=0)
a = np.array([G_dot[-1,:]])  # last row of the matrix
G_dot = np.append(G_dot,a,axis = 0) # appended to matrix

ri_f1 = (np.conj(G)*G_dot).imag
ri_f2 = np.abs(G)**2
ri_fnum = np.sum(ri_f1, axis = -1)
ri_fdenom = np.sum(ri_f2,axis = -1)
        
#Estimate Spectrum for the derivative of the phase error
phi_dot = np.sum((np.conj(G)*G_dot).imag, axis = -1)/\
            np.sum(np.abs(G)**2, axis = -1)
                
#Integrate to obtain estimate of phase error(Jak)
phi = np.cumsum(phi_dot)


### baseline

dG = G[1:,:] - G[0:-1,:]
# starting here with the second element  
f1 = np.imag(np.conj(G[1:,:]) * dG)
f2 = np.float_power(np.abs(G[1:,:]), 2) #
fnum = np.sum(f1, axis = -1)
fdenom = np.sum(f2,axis = -1)
dPhi = fnum / fdenom 
phi2 = np.cumsum(dPhi)

# FASIH
delta_phase = np.angle( np.sum(  np.conj(G[0:-1, :]) * G[1:, :] , axis=-1) )
phi3 = np.cumsum(delta_phase)


# Amplitude demodulation as done in 
# https://www.gaussianwaves.com/2017/06/phase-demodulation-using-hilbert-transform-application-of-analytic-signal/
# here the phase angle is computed directly (not the derivative)
phi4_matrix = np.angle(G)
np.angle( np.sum(  G.real, axis=-1) / np.sum(G.imag) )
phi4 = np.unwrap(np.angle(G[:, 0])) # don't know how to coherently sum the angles before linear detrending. Is it even possible?


phi = remove_linear_trend(phi, t)
phi2 = remove_linear_trend(phi2, t)
phi3 = remove_linear_trend(phi3, t)
phi4 = remove_linear_trend(phi4, t)

# should get smaller per iteration, now it has no real Aussagekraft
print(rms(phi))
print(rms(phi2))
print(rms(phi3))
print(rms(phi4))
print(rms(m_t))


plt.figure()
plt.subplot(3, 1,1)
plt.plot(t,m_t) #plot modulating signal
plt.title('Modulating signal')
plt.xlabel('t')
plt.ylabel('m(t)')

plt.subplot(3, 1,2)
plt.plot(t[: len(phi2)],phi2) #plot demodulated signal
plt.title('Demodulated signal phi4')
plt.xlabel('t')
plt.ylabel('demod(t)')

plt.subplot(3, 1,3)
plt.plot(t[: len(phi3)],phi3) #plot demodulated signal
plt.title('Demodulated signal phi')
plt.xlabel('t')
plt.ylabel('demod(t)')

plt.show()




print("finished")









