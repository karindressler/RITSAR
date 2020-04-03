jks7592: I was using this a lot to learn about radar processing and wound up updating a few small portions of the code because I use python 3.

March 27th 2020: I wanted to focus the Wide Angle SAR dataset from AFRL and decided to modify the
phsRead.AFRL (warning on compatibility, I made polarization a keyword argument which changes the order) and imgTools.img_plane_dict functions. While fighting with it I noticed that numpy 
now has the ability to interpolate complex numbers directly instead of doing two separate interpolations for real and imaginary parts. I only updated functions that used numpy.interp such as backprojection and polar format algorithm. I did not update those that used scipy.interp1d as I'm not familiar with the extra settings and that function currently only takes real values. 

I also added a "checkme" option to `imgTools.img_plane_dict` based on the code packaged with AFRL's Wide Angle SAR dataset which will calculate and print the theoretical max resolution and scene limits then print those which the image plane is currently set to.

March 29th 2020: Found out I was doing something dumb with 'isinstance' when checking input types as it can check against a tuple of types instead of needing to check each type individually. Also decided to add a `subaperture` function to `imgTools.py` which separates the phase history and platform into a list of corresponding smaller histories and platforms. 

~-----------~

# RITSAR
Synthetic Aperture Radar (SAR) Image Processing Toolbox for Python

Before installation, please make sure you have the following:
- SciPy. Comes with many Python distributions such as Enthought Canopy, Python(x,y), and Anaconda.  Development was done using the Anaconda distribution which can be downloaded for free from https://store.continuum.io/cshop/anaconda/. 
- OpenCV (optional). If using the omega-k algorithm, OpenCV is required. Instructions for installing OpenCV for Python can be found at  https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html#py-table-of-content-setup.
- Spectral (optional).  Needed to interface with .envi files.  Can be installed from the command line using

  $ pip install spectral

  alternatively, Spectral can be downloaded here: http://www.spectralpython.net/ 
  
To get started, first make sure your SciPy and NumPy libraries are up-to-date.  With Anaconda, this can be done by typing the following into a terminal or command prompt:

$ conda update conda

$ conda update anaconda

Once you've ensured the required libraries are up-to-date, download the zip file and extract it to a directory that from here on will be referred to as \<ritsar_dir\>.  Open up a command line or terminal and type:

$ cd \<ritsar_dir\>

$ python setup.py install

then...

$ cd ./examples

$ ipython --pylab

From the ipython console, type:

In [1]: %run FFBPmp_demo

In [2]: import matplotlib.pylab as plt; plt.show()

or run any other demo.  Alternatively, you can open up the demos in an IDE of your choice to experiment with the different options available.

Current capabilities include modeling the phase history for a collection of point targets as well as processing phase histories using the polar format, omega-k, backprojection, digitally spotlighted backprojection, fast-factorized backprojection, and fast-factorized backprojection with multi-processing algorithms.  Autofocusing can also be performed using the Phase Gradient Algorithm.  The current version can interface with AFRL Gotcha and DIRSIG data as well as a data set provided by Sandia.

Data included with this toolset includes a small subset of the AFRL Gotcha data provided by AFRL/SNA.  The full data set can be downloaded separately from https://www.sdms.afrl.af.mil/index.php?collection=gotcha after user registration.  Also included is a single dataset from Sandia National Labs.

If anyone is interested in collaborating, I can be reached at dm6718@g.rit.edu. Ideas on how to incorporate a GUI would be greatly appreciated.
