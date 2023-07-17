##############################################################################
#                                                                            #
#  This is a demonstration of the ritsar autofocusing algorithm.  The Sandia #
#  dataset is used to generate the original image.  A randomly generated     #
#  10th order polynomial is used to create a phase error which is            #
#  subsequently used to degrade the image in the along-track direction.      #
#  This degraded image is then passed to the auto_focusing algorithm.        #
#  The corrected image as well as the phase estimate is returned and relevant #
#  plots are generated.                                                      #
#                                                                            #
##############################################################################

#Add include directories to default path list
from sys import path

path.append('../')

#Include standard library dependencies
import os
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
from scipy.stats import linregress

#Include SARIT toolset
from ritsar import imgTools, phsRead, phsTools
from ritsar import signal as sig

#Define directory containing *.au2 and *.phs files
directory = os.path.join(Path(__file__).parent, "data/Sandia/")# './data/Sandia/'

#Import phase history and create platform dictionary
#Karin:  phs: phase history samples (?)
[phs, platform] = phsRead.Sandia(directory)

#Correct for residual video phase
#Karin: This seams to do a mixing of the received signal with the transmitted signal
# nothing to see after this step...
phs_corr = phsTools.RVP_correct(phs, platform)
# plt.figure()
# plt.imshow(10*np.log10(np.abs(phs_corr)/np.abs(phs_corr).max()), cmap = cm.Greys_r)
# plt.title("phase RVP corrected")
# plt.show()

#Import image plane dictionary from './parameters/img_plane'
img_plane = imgTools.img_plane_dict(platform,
                           res_factor = 1.0, n_hat = platform['n_hat'])

#Apply polar format algorithm to phase history data
#(Other options not available since platform position is unknown)
# Karin: not sure what this does
# it returns the Fourier transform of the image that was interpolated onto 
# a new grid (?)
img_pf = imgTools.polar_format(phs_corr, platform, img_plane, taylor = 30)
# plt.figure()
# plt.imshow(10*np.log10(np.abs(img_pf)/np.abs(img_pf).max()), cmap = cm.Greys_r)
# plt.title("polar format")
# plt.show()

#Degrade image with random 10th order polynomial phase
# Karin: Don't understand why the coefficients are multiplied with the image shape ...
# The polynomial coefficients become HUGE this way and the phase error even bigger,
# making the error wrap around 2 pi multiple times with every step 
# this way it becomes more of a random phase error!
coeff = (np.random.rand(10)-0.5) * img_pf.shape[0] 

# Karin: okay, x - range is in between -1 and 1, so the huge coeffs do not do much ... 
# I still think the phase error is to big in between samples (min -300, max 800)
# In the original Eichel paper, the phase error has min/max of -10/15 over the range of 512
# pixels

x = np.linspace(-1,1,img_pf.shape[0])
y = np.poly1d(coeff)(x)
slope, intercept, r_value, p_value, std_err = linregress(x,y)
line = slope*x+np.mean(y)
# Karin: any linear component (bias and slope) is deleted from the error, as 
# the autofocus does not correct the linear component
# Karin: should be the same as setting the first two coeffs to zero
y = y-line
# Karin. numpy tile repeats (similar to matlab repmat) the array (all range pixels get the same error)
ph_err = np.tile(np.array([y]).T,(1,img_pf.shape[1])) # [in radians?]
# twiddle around spectrum? I am not sure, how the error is applied
img_err = sig.ft(sig.ift(img_pf,ax=0)*np.exp(1j*ph_err),ax=0)

#Autofocus image
print('autofocusing')
img_af, af_ph = imgTools.autoFocus2(img_err, win = 'auto')
#img_af, af_ph = imgTools.autoFocus2(img_err, win = 0, win_params = [500,0.8])

# #Output image
# # because of the heavy wrap-arounds of the phase, the true error and the estimated error are not really comparable
# plt.figure()
# plt.plot(x,y,x,af_ph); plt.legend(['true error','estimated error'], loc = 'best')
# plt.ylabel('Phase (radians)')
# plt.show()

# #Output image
# plt.figure()
# imgTools.imshow(img_af, dB_scale = [-45,0])
# plt.show()
