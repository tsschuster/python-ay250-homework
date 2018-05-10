import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import CubicSpline

def getPreImage(ns,Rs,v,filter_thresh = [.07,.07,.89],del_thresh = 1,return_ptsQ = False,width = 2./100.,num_pts = 1000):
    """
    Computes preimage of n = v given a grid of n values ns at positions Rs. Works in several steps:
        1) Rotate ns such that preimage v is aligned along z direction
        2) Compute density of points whose ns are within filter_thresh[0] distance of desired preimage v
        3) Filter to keep only points within filter_thresh[1]*(1-filter_thresh[2]*density/np.max(density)) of preimage v (this filters more strongly at high density points - the goal here is to not over(under)sample regions where ns vary slowly (quickly) away from preimage
        4) Find appropriate "order" of the remaining points (our discrete approximation of the 1D preimage) using Order1DSet
        5) Interpolate smoothly between points using scipy.interpolate.CubicSpline
        6) Filter the 1D image using scipy.ndimage.filters.gaussian_filter to make more smooth. Tune for a balance between smoothness & precise interpolation - goal is an accurate reconstruction of the preimage despite our coarse approximation
    
    Input:
        ns: 3 x N x N x N array of n_i(k_j)
        Rs: 3 x N x N x N array of [X,Y,Z] where X,Y,Z form a 3d meshgrid of k_j in the Brillouin zone
        v: 3-vector of desired preimage (e.g. [0,0,1] for preimage of n = z)
        filter_thresh: list of length 3, see above description Step 3)
        del_thresh: float, determines maximum distance to allow in Step 4) - all points considered after this threshold is met will be deleted from the preimage
        return_ptsQ: if True, returns the discrete set of points between Steps 4) and 5) used for the interpolation
        width: fraction of total curve length to be used in gaussian smoothing
        num_pts: number of points at which to sample final curve
        
    Output:
        If return_ptsQ = False:
            xp,yp,zp: 1D arrays of length num_pts of x,y,z values along curve
        If return_ptsQ = True:
            xp,yp,zp: 1D arrays of length num_pts of x,y,z values along curve
            x_im0,y_im0,z_im0: points kept after Step 3)
            x_im_all,y_im_all,z_im_all: points kept after Step 2)    
    """
    Xs,Ys,Zs = np.copy(Rs[0,:,:,:]),np.copy(Rs[1,:,:,:]),np.copy(Rs[2,:,:,:])

    #Rotate ns
    ns_rot = rotateNs(ns,v)

    #Get parallel and perpendicular components of ns after rotation
    n_par = ns_rot[2,:,:,:]
    n_perp1,n_perp2 = ns_rot[0,:,:,:],ns_rot[1,:,:,:]
    n_perps = np.sqrt(n_perp1**2 + n_perp2**2)

    #Filtering in two steps:
    #First: Isolate points "near" preimage (have perp components
    n_bool = np.logical_and(np.abs(n_perps) < filter_thresh[0],n_par < 0)
    
    #Only done for illustrating the filtering steps
    x_im_allpts = Xs[n_bool]
    y_im_allpts = Ys[n_bool]
    z_im_allpts = Zs[n_bool]
    
    #Second: Define tighter filter for regions with a higher density of points near v
    dens = gaussian_filter(n_bool.astype("float"),1)
    thresh_mat = filter_thresh[1]*(1-filter_thresh[2]*dens/np.max(dens)) #parameters chosen for best look at N = 100

    n_bool = np.logical_and(np.abs(n_perps) < thresh_mat,n_par < 0)

    #Get list of satisfactory points - our grid approximation of the preimage
    x_im0 = Xs[n_bool]
    y_im0 = Ys[n_bool]
    z_im0 = Zs[n_bool]
    
    #"Sort" such that each point is the closest point to the previous point
    r_im_unsorted = np.asarray([x_im0,y_im0,z_im0])
    r_im = order1DSet(r_im_unsorted,del_thresh)

    #print("getPreImage used " + str(np.shape(r_im)[1]) + " out of " + str(len(x_im0)) + " points in interpolation.")
    x_im = np.copy(r_im[0,:])
    y_im = np.copy(r_im[1,:])
    z_im = np.copy(r_im[2,:])
    x_im = np.append(x_im,x_im[0])
    y_im = np.append(y_im,y_im[0])
    z_im = np.append(z_im,z_im[0])
    t_im = np.linspace(0,1,len(x_im))

    fx = CubicSpline(t_im,x_im,bc_type="periodic") 
    fy = CubicSpline(t_im,y_im,bc_type="periodic") 
    fz = CubicSpline(t_im,z_im,bc_type="periodic")
    
    #Get points from interpolation functions
    t_p = np.linspace(0,1,num_pts)
    x_p = fx(t_p)
    y_p = fy(t_p)
    z_p = fz(t_p)

    x_p = gaussian_filter(x_p,width*num_pts,mode='wrap')
    y_p = gaussian_filter(y_p,width*num_pts,mode='wrap')
    z_p = gaussian_filter(z_p,width*num_pts,mode='wrap')
    
    if return_ptsQ:
        return x_p,y_p,z_p,x_im0,y_im0,z_im0,x_im_allpts,y_im_allpts,z_im_allpts
    return x_p,y_p,z_p

def order1DSet(r_im_unsorted,del_thresh):
    """
    Finds "best" ordering of a set of points in 3D for a 1D interpolation, by choosing an initial point and successively adding the nearest as-of-yet unincluded point to the ordering. This process is terminated if the nearest point to include is greater than del_thresh from the most recent point, in which case all unincluded points are deleted.
    Input: 
        r_im_unsorted: 3 x N array of undordered points
        del_thresh: threshold for terminating set
    Output:
        r_im: 3 x M array of ordered points, M may be less than N if some points did not fit in algorithm (were a distance > del_thresh from most recent point)
    """
    argmin = 0
    r_im = np.copy(r_im_unsorted)
    for i in range(1,np.shape(r_im)[1]):
        argmin = np.argmin(np.sum((r_im[:,i:]-r_im[:,i-1][:,None])**2,axis=0))
        temp = np.copy(r_im[:,argmin+i])
        if np.sum((r_im[:,argmin+i]-r_im[:,i-1])**2) > del_thresh:
            r_im = np.copy(r_im[:,:i-1])
            break
        r_im[:,argmin+i] = np.copy(r_im[:,i])
        r_im[:,i] = temp
    return r_im

def rotateNs(ns,v):
    """
    Rotates array of 3-vectors ns such that ns = v is aligned with the z-axis after rotation
    Input:
        ns: 3 x N1 x ... array of 3-vectors to be rotated
        v: length 3 array of 3-vector to be rotated to z-direction
    Output:
        ns_rot: 3 x N1 x ... array of rotated 3-vectors
    """
    #Normalize v, get perpendicular vectors, and construct rotation matrix to rotate v into z = (0,0,1)
    v = v.astype("float")
    v /= np.linalg.norm(v) #normalize if not already
    if v[0] != 0 or v[1] != 0:
        v_perp1 = np.array([-v[1],v[0],0])
        v_perp1 /= np.linalg.norm(v_perp1)
    else:
        v_perp1 = np.array([1.,0,0])
    v_perp2 = np.cross(v,v_perp1)
    R = np.array([v_perp1,v_perp2,v]) #rotation matrix's columns are v and two perpendicular vectors

    #Rotate ns
    ns_rot = np.inner(R,np.moveaxis(ns, 0, -1))
    return ns_rot

def test_1():
    #make sure output of order1DSet is of appropriate shape
    shape = np.shape(order1DSet(np.random.random((3,20)),1))
    assert shape[0] == 3 and shape[1] <= 20
    
def test_2():
    #test that rotateNs handles (unnormalized) 
    assert (rotateNs(np.asarray([[1,0,0],[0,0,1]]).T,np.asarray([1,0,0]))[:,0] == np.asarray([0,0,1.])).all()
    
def test_3():
    #test that rotateNs handles no rotation correctly
    ns = np.asarray([[1,.3,0],[0,.5,1]]).T
    assert (rotateNs(ns,np.asarray([0,0,1])) == ns).all()
