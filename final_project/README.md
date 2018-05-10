AY250 Final Project:

---------------------

Description of the contents of this folder:

PlotPreimages.ipynb is a Jupyter notebook which can generate the plots which are the final product of this project. It uses a number of python files also contained in the folder. Code previously written for my research is in qubitmatrices.py, params.py, hoppings.py, and hamiltonian.py. Additional code written specifically for this project is in preimage.py. In the end, I discovered that matplotlib had a key flaw which did not allow ideal plotting of the preimages (discussed in video presentation) - I thus wrote my results to a csv file preimages.csv and very quickly plotted them in the Mathematica notebook preimages.nb, just for completeness sake.

The video/screencapture presentation of my code is contained in the video directory and is broken into 3 parts to satisfy github's file size restriction. I apologize that it ran a little long! (~23 minutes) Hopefully you can play at > 1x speed to get through it quicker. Much of the beginning is explaining code written for previous research, so you can skip some amount there if you desire.

------------------

The following was the project proposal I submitted:

My research project involves the simulation of a certain topological insulator, the "Hopf insulator", realized in an ultracold molecule experiment. For all practical purposes, just consider this to be a function from 3D to the 2D Bloch sphere. Using numpy & scipy concepts (some of which I learned in this class!), I already have code to calculate this function given a variety of continuous model parameters.

I would like to plot the preimages of given points on the 2-sphere in 3D (i.e. the set of all points k such that f(k) = n, where n is a chosen pt. on the Bloch sphere). Since we are mapping from 3D to 2D, these preimages are 1D objects, and there doesn't seem to be a good publicly available package/function to solve for this. I propose to i) approximate these preimages by evaluating the function on a coarse grid in 3D, ii) smooth this coarse preimage approximation using scipy.interpolate and scipy.ndimage.filter, and iii) plot this using 3D plots in matplotlib.

USES: 02_Plotting_and_Viz, 03_Numpy_Scipy_Stats, 05_scikit-image (a small bit from the guest lecture)


