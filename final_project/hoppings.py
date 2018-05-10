import numpy as np

#Bare (without Floquet damping) dipole-dipole hoppings
def tAA(p,x,y,z):
    """
    Returns bare (without Floquet damping) sublattice A -> A hopping
    Input:
        p: params object for model
        x,y,z: coordinates of unit cell to hop to, can't accept x=y=z=0 (no exception is raised because this function was designed to be composed of only numpy operations
    Output:
        hopping: float
    """
    r,t,ph = xyz2rtp(x,y,p.a*z)
    return (1.5/r**3)*(3*np.cos(t)**2-1)
    
def tAB(p,x,y,z):
    """
    Returns bare (without Floquet damping) sublattice A -> B hopping
        p: params object for model
        x,y,z: coordinates of unit cell to hop to
    Output:
        hopping: complex float
    """
    r,t,ph = xyz2rtp(x,y,p.a*z+p.b)
    return (1.5*3./r**3)*(np.sin(t)*np.cos(t))*np.exp(-1j*ph)

def xyz2rtp(x,y,z):
    """
    Converts between Cartesian and radial coordinates
    Input:
        x,y,z: cartesian coordinates
    Output: 
        r,t,phi: radius, theta, and phi in spherical coordinates
    """
    r=np.sqrt( x**2+y**2+z**2)
    t=np.arccos( z/r )
    ph=np.arctan2( y, x )
    return (r,t,ph)
