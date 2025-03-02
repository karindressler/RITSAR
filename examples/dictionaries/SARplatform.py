##############################################################################
#                                                                            #
#  This file is where all platform parameters are defined.  All coordinates  #
#  are assumed to have the scene center at the origin of a cartesian         #
#  coordinate system (ECF, [lat,long,zenith], for example).                  #
#                                                                            #
#  If passing in data from an auxiliary file, replace the definitions        #
#  below with expressions that translate the auxiliary data labels           #
#  into the dictionary keys listed at the bottom of the program.             #
#                                                                            #
##############################################################################

#Include dependencies
import numpy as np
from scipy.constants import speed_of_light 

def plat_dict(aux = {}):
    #Define universal constants
    c = speed_of_light
    
    #Define platform parameters
    f_0 = 10.0e9   #center frequency
    wvl = c/f_0
    chirprate = 4.8e12
    T_p = 38.5e-6
    B = T_p*chirprate
    AD_sampling = 45.0e6
    delta_t = 1.0/AD_sampling
    nsamples = 2048
    npulses = 1950
    squint = np.pi/2
    graze = 0.52359877559829893
    
    #Define platform position
    
    #x
    R0_apCenter = 10000.0
    Xc = R0_apCenter*np.cos(graze)
    
    #y
    prf = 1000.0
    v_p = 100.0
    dy = v_p/prf
    Y_c = Xc*np.tan(np.pi/2-squint)
    y = np.linspace(-npulses/2, npulses/2, npulses)*dy+Y_c
    
    #z    
    h = 5000.0
    
    #r[npulses , (x,y,z)]
    pos = np.zeros([npulses, 3])
    pos[:,0] = Xc
    pos[:,1] = y
    pos[:,2] = h
    
    #Synthetic aperture length
    L = np.linalg.norm(pos[-1]-pos[0])
    
    #Vector to scene center at synthetic aperture center
    if np.mod(npulses,2)>0:
        R_c = pos[npulses//2]
    else:
        R_c = np.mean(
                pos[npulses//2-1:npulses//2+1],
                axis = 0)
                    
    #Coherent integration angle
    p1_hat = pos[0]/np.linalg.norm(pos[0]); p2_hat = pos[-1]/np.linalg.norm(pos[-1])
    Delta_theta = np.arccos(np.dot(p1_hat, p2_hat))
    
    #Look angle w.r.t. velocity vector for each pulse (minus pi/2)
    theta = np.zeros(npulses)
    R_vel = pos[-1]-pos[0]
    R_vel_norm = np.linalg.norm(R_vel)
    for i in range(npulses):
        num = np.dot(pos[i], R_vel)
        den = np.linalg.norm(pos[i])*R_vel_norm
        theta[i] = np.arccos(num/den)-np.pi/2
        
    #Obtain frequency locations of range samples
    #Assuming demodulated linear FM signal
    t = np.linspace(
        -nsamples/2, nsamples/2, nsamples)*delta_t #demodulated fast time
    B_IF = (t.max()-t.min())*chirprate
    delta_r = c/(2*B_IF)
    freq = f_0+chirprate*t
    omega = 2*np.pi*freq
    k_r = 2*omega/c
    k_y = np.linspace(-npulses/2,npulses/2,npulses)*2*np.pi/L
    
    
    #Construct dictionary and return to caller
    platform = \
    {
        'f_0'       :   f_0,
        'wvl'       :   wvl,
        'chirprate' :   chirprate,
        'T_p'       :   T_p,
        'B'         :   B,
        'B_IF'      :   B_IF,
        'AD_sampling':  AD_sampling,
        'delta_t'   :   delta_t,
        'nsamples'  :   nsamples,
        'npulses'   :   npulses,
        'squint'    :   squint,
        'graze'     :   graze,
        'pos'       :   pos,
        'L'         :   L,
        'delta_r'   :   delta_r,
        'Delta_theta':  Delta_theta,
        'theta'     :   theta,
        'R_c'       :   R_c,
        't'         :   t,
        'freq'      :   freq,
        'omega'     :   omega,
        'k_r'       :   k_r,
        'k_y'       :   k_y
    }
    
    return(platform)