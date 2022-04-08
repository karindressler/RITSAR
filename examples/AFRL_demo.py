'''
##############################################################################
#                                                                            #
#  This is a demonstration of the ritsar toolset using AFRL Gotcha data.     #
#  Algorithms can be switched in and out by commenting/uncommenting          #
#  the lines of code below.                                                  #
#                                                                            #
##############################################################################
'''

#%%
#Add include directories to default path list
from sys import path
path.append('../')

#Include standard library dependencies
import matplotlib.pylab as plt

#Include RITSAR toolset
from ritsar import phsRead
from ritsar import imgTools

#%%
#Define top level directory containing *.mat file
#and choose polarization and starting azimuth
pol = 'HH'
directory = './data/AFRL/pass1'
start_az = 1

#Import phase history and create platform dictionary
[phs, platform] = phsRead.AFRL(directory, start_az, pol=pol, n_az = 3)

#Create image plane dictionary
img_plane = imgTools.img_plane_dict(platform, res_factor = 1.4, upsample = True, aspect = 1.0)

#Apply algorithm of choice to phase history data
img_bp = imgTools.backprojection(phs, platform, img_plane, taylor = 20, upsample = 6,prnt=50)
img_pf = imgTools.polar_format(phs, platform, img_plane, taylor = 20)

#Output image
imgTools.imshow(img_bp, dB_scale = [-30,0])
plt.title('Backprojection')
imgTools.imshow(img_pf, dB_scale = [-30,0])
plt.title('Polar Formatting')


#%%
#Define top level directory containing *.mat file
#and choose polarization and starting azimuth
directory ='./data/AFRL/Wide_Angle_SAR'
start_az = 214

#Import phase history and create platform dictionary
[phs, platform] = phsRead.AFRL(directory, start_az, n_az = 1)

#Create image plane dictionary
#img_plane = imgTools.img_plane_dict(platform, res_factor = 1.4, upsample = True, aspect = 1.0)
#img_plane = imgTools.img_plane_dict(platform, numPixels = 160, length=22, checkme=True)
img_plane = imgTools.img_plane_dict(platform, force_xy=True, numPixels = 160, length=22, checkme=True)

#For wide angle SAR the backprojection algorithm is recommended
# -> TODO: Figure out why polar format breaks sometimes (k_y problem, R_c problem)
img_bp = imgTools.backprojection(phs, platform, img_plane, taylor = 20, upsample = 6,prnt=1000)

#Output image
imgTools.imshow(img_bp, dB_scale = [-15,0])
plt.title('Backprojection')

#%%

#Use the same file to create and display subapertures
directory ='./data/AFRL/Wide_Angle_SAR'
start_az = 214
[phs, platform] = phsRead.AFRL(directory, start_az, n_az = 1)

# Create subapertures with angle less than 360 to use polar format algorithm
phs_list,plat_list=imgTools.subaperture(phs,platform,angle=15,keep_R_c=False)
# Or compare to using backprojection on the same (keep_R_c can be true or false with bp)
#phs_list,plat_list=imgTools.subaperture(phs,platform,angle=15,keep_R_c=True)

print(len(phs_list))
images=[]
#######
for i in range(len(phs_list)):
    checkme=True if i==0 else False
    lil_img_plane=imgTools.img_plane_dict(plat_list[i],checkme=checkme,numPixels=160,\
                                          length=22,force_xy=True)
    img = imgTools.polar_format(phs_list[i], plat_list[i], lil_img_plane, taylor = 20,prnt=False)
    #img = imgTools.backprojection(phs_list[i], plat_list[i], lil_img_plane, taylor = 20,prnt=False)
    
    images.append(img)

#TODO: make this plot an animation, move this demo to jupyter notebook
for image in images:
    plt.figure()
    imgTools.imshow(image, dB_scale = [-15,0])
    plt.title('')



# %%
