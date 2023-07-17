import numpy as np
from . import signal as sig
import matplotlib.pyplot as plt
from scipy.stats import linregress

def pga_ritsar(img, win = 'auto', win_params = [100,0.5]):
    """
    This program autofocuses an image using the Phase Gradient Algorithm.     
    If the parameter win is set to auto, an adaptive window is used.          
    Otherwise, the user sets win to 0 and defines win_params.  The first      
    element of win_params is the starting windows size.  The second element   
    is the factor by which to reduce it by for each iteration.  This version  
    is more suited for an image that is mostly focused.  Below is the paper   
    this algorithm is based off of.

    D. Wahl, P. Eichel, D. Ghiglia, and J. Jakowatz, C.V., \Phase gradient    
    autofocus-a robust tool for high resolution sar phase correction,"        
    Aerospace and Electronic Systems, IEEE Transactions on, vol. 30,          
    pp. 827{835, Jul 1994.  

    Args:
        img: calibrated complex SAR image [AZIMUTH_PULSES, RANGE_SAMPLES]
        win (str): default = "auto"
        win_params: initial size of window, to cut out point targets

    """
   
    #Derive parameters
    npulses = int(img.shape[0]) # azimuth direction
    nsamples = int(img.shape[1]) # range direction
    
    #Initialize loop variables
    img_af = 1.0*img
    max_iter = 30
    af_ph = 0
    rms = []
    
    #Compute phase error and apply correction
    for iii in range(max_iter):
        
        #Find brightest azimuth sample in each range bin
        index = np.argsort(np.abs(img_af), axis=0)[-1] # sort along azimuth
        
        #Circularly shift image so max values line up   
        f = np.zeros(img.shape)+0j
        for i in range(nsamples):
            f[:,i] = np.roll(img_af[:,i], npulses//2-index[i])
        
        if win == 'auto':
            #Compute window width    
            s = np.sum(f*np.conj(f), axis = -1)
            s = 10*np.log10(s/s.max())
            width = np.sum(s>-30)
            window = np.arange(npulses//2-width//2,npulses//2+width//2)
        else:
            #Compute window width using win_params if win not set to 'auto'    
            width = int(win_params[0]*win_params[1]**iii)
            window = np.arange(npulses//2-width//2,npulses//2+width//2)
            if width<5:
                break
        
        #Window image
        g = np.zeros(img.shape)+0j
        g[window] = f[window]
        
        #Fourier Transform
        G = sig.ift(g, ax=0) # window along azimuth (per range bin) 
        
        #take derivative
        G_dot = np.diff(G, axis=0)
        a = np.array([G_dot[-1,:]])
        G_dot = np.append(G_dot,a,axis = 0)
        
        #Estimate Spectrum for the derivative of the phase error
        phi_dot = np.sum((np.conj(G)*G_dot).imag, axis = -1)/\
                  np.sum(np.abs(G)**2, axis = -1)
                
        #Integrate to obtain estimate of phase error(Jak)
        phi = np.cumsum(phi_dot)
        
        #Remove linear trend
        t = np.arange(0,npulses) #originally np.arange(0,nsamples)
        slope, intercept, r_value, p_value, std_err = linregress(t,phi)
        line = slope*t+intercept
        phi = phi-line
        rms.append(np.sqrt(np.mean(phi**2)))
        
        if win == 'auto':
            if rms[iii]<0.1:
                break
        
        #Apply correction
        phi2 = np.tile(np.array([phi]).T,(1,nsamples))
        IMG_af = sig.ift(img_af, ax=0)
        IMG_af = IMG_af*np.exp(-1j*phi2)
        img_af = sig.ft(IMG_af, ax=0)
        
        #Store phase
        af_ph += phi    
       
    # fig = plt.figure(figsize = (12,10))
    # ax1 = fig.add_subplot(2,2,1)
    # ax1.set_title('original')
    # ax1.imshow(10*np.log10(np.abs(img)/np.abs(img).max()), cmap = cm.Greys_r)
    # ax2 = fig.add_subplot(2,2,2)
    # ax2.set_title('autofocused')
    # ax2.imshow(10*np.log10(np.abs(img_af)/np.abs(img_af).max()), cmap = cm.Greys_r)
    # ax3 = fig.add_subplot(2,2,3)
    # ax3.set_title('rms phase error vs. iteration')
    # plt.ylabel('Phase (radians)')
    # ax3.plot(rms)
    # ax4 = fig.add_subplot(2,2,4)
    # ax4.set_title('phase error')
    # plt.ylabel('Phase (radians)')
    # ax4.plot(af_ph)
    # plt.tight_layout()
    

    print('number of iterations: %i'%(iii+1))
                     
    return(img_af, af_ph)


def pga_hippler(img, win = 'auto', win_params = [100,0.5]):

    #Derive parameters
    npulses = int(img.shape[0]) # azimuth direction
    nsamples = int(img.shape[1]) # range direction
    
    #Initialize loop variables
    img_af = 1.0*img
    max_iter = 30
    af_ph = 0
    rms = []
    
    #Compute phase error and apply correction
    for iii in range(max_iter):
        
        #Find best azimuth sample for each range bin
        # az_index, quality = find_point_targets(img_af)

        #Find brightest azimuth sample in each range bin
        
        point_targets = get_point_targets(img_af)
        num_pt_targets = len(index_az)
        quality = np.ones(num_pt_targets) # for now assuming one point target per azimuth line
        index_range = range(num_pt_targets) # for now assuming one point target per azimuth line
        
        #Circularly shift image so max values line up
        
        f = np.zeros(num_pt_targets, nsamples) + 0j

        for i in range(num_pt_targets):
            
            f[:,i] = np.roll(img_af[:,i], npulses//2-index_az[i])

        
        #Circularly shift image so max values line up   (or cut out image) 

        # Adapt window width
       
        #Compute window width using win_params if win not set to 'auto'    
           
        #Window image
        
        
        #Fourier Transform
        G = sig.ift(g, ax=0) # window along azimuth (per range bin) 
        # window along azimuth (one FT per range bin) 
        
        #take derivative
        dG = G[1:,:] - G[0:-1,:]
        # starting here with the second element  
        f1 = np.imag(np.conj(G[1:,:]) * dG)
        f2 = np.float_power(np.abs(G[1:,:]), 2) #
        fnum = np.sum(f1, axis = -1)
        fdenom = np.sum(f2,axis = -1)
        dPhi = fnum / fdenom # phi_dot

        #Integrate to obtain estimate of phase error(Jak)
        # Hippler: remove mean -> will be done in linear trend removal anyways
        phi = np.cumsum(dPhi)
        
        #Remove linear trend
        t = np.arange(0,npulses) #originally np.arange(0,nsamples)
        slope, intercept, r_value, p_value, std_err = linregress(t,phi)
        line = slope*t+intercept
        phi = phi-line
        
        if win == 'auto':
            if rms[iii]<0.1:
                break
        
        #Apply correction
        phi2 = np.tile(np.array([phi]).T,(1,nsamples))
        IMG_af = sig.ift(img_af, ax=0)
        IMG_af = IMG_af*np.exp(-1j*phi2)
        img_af = sig.ft(IMG_af, ax=0)
        
        #Store phase
        af_ph += phi  

def  find_point_targets(img, min_ququality = 0.5, min_magn = 5, iii):
    pt_list = list()
    

    min_quality = 0.5 # orig 0.4 targetQualityLimit
    min_magn = 5 # orig very small
    maxNumberOfTargets = 5000
    maxTargetAzExt = 1000
    num_targets_per_line = 3
    spectralTargetAzWindow = np.ones(1000, 1)
        
    npulses = int(img.shape[0]) # azimuth direction
    nsamples = int(img.shape[1]) # range direction

    #Find num_targets_per_line brightest azimuth samples in each range bin
    abs_img = np.abs(img)
    index = np.argsort(abs_img, axis=0)[-num_targets_per_line:-1, :] # sort along azimuth


    # check for mimimum magnitude condition
    for i_maximum in range(num_targets_per_line):
        for r in range(nsamples):
            magn = abs_img[index[i_maximum, r], r]
            if( magn > min_magn):
                point_target = {"i_az": index[i_maximum, r], "i_range": r, "magn": magn}
                pt_list.append(point_target)
                

    #Circularly shift azimuth lines so max values line up 
    f = np.zeros(npulses, len(pt_list))+0j
    for i, pt in enumerate(pt_list):
        f[:,i] = np.roll(img[:,pt["i_range"]], npulses//2-pt["i_az"])
    

    #Compute window width
    # Karin: FIXME check how it is done in Hippler algorithm
    s = np.sum(f*np.conj(f), axis = -1)
    s = 10*np.log10(s/s.max())
    width = np.sum(s>-30)
    window = np.arange(npulses//2-width//2,npulses//2+width//2)

    #Window image
    g = np.zeros(img.shape)+0j
    g[window] = f[window]
    
    # for Hippler smaller, here zero padding
    #Fourier Transform
    G = sig.ift(g, ax=0) # window along azimuth (per range bin)
    
    abs_G = np.abs(G)
    # for Hippler smaller window is used to estimate quality
    # Karin: does zero padded FFT have implication for quality estimation
    # Second guess: No
    quality =  np.std(abs_G)/np.mean(abs_G)
    for i, pt in enumerate(pt_list):
        pt["quality"] = quality[i]
        # append each spectrum to pt? -> check performance!
    
    return pt, G
    




