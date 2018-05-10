import numpy as np
import qubitmatrices as qbm
import scipy.linalg
import params

from hoppings import *

def ns(kx,ky,kz,p):
    """
    return n (on Bloch sphere) and E (energy) at a given pt in momentum-space for a model
    Input:
        p: params instance describing model
        kx,ky,kz: point in Brillouin zone at which to compute, ki in [-pi,pi]
    Output: 4 real-valued outputs
        nx,ny,nz: components of n vector
        E: energy
    """
    
    #calculate HAA (real) & HAB (complex) components of the 2x2 Hamiltonian
    HAA = (np.moveaxis(np.broadcast_to(p.hopsAA,np.shape(kx)+np.shape(p.hopsAB)),-1,0)*np.exp(1.j*(np.tensordot(p.xs,kx,axes=0)+np.tensordot(p.ys,ky,axes=0)+np.tensordot(p.zs,kz,axes=0)))).sum(axis=0)
    HAB = (np.moveaxis(np.broadcast_to(p.hopsAB,np.shape(kx)+np.shape(p.hopsAB)),-1,0)*np.exp(1.j*(np.tensordot(p.xs,kx,axes=0)+np.tensordot(p.ys,ky,axes=0)+np.tensordot(p.zs,kz,axes=0)))).sum(axis=0)
    
    #calculate ns and E
    nx = HAB.real
    ny = HAB.imag #sign is Pauli matrix convention
    nz = HAA.real
    E = np.sqrt(nx**2 + ny**2 + nz**2)

    return nx,ny,nz,E

def test_1():
    #make sure output of ns is of correct shape
    p = params.params()
    kx,ky,kz = np.random.random((3,4))
    shape = np.shape(ns(kx,ky,kz,p)[0])
    assert shape == np.shape(kx)
    
def test_2():
    #test that it can also support float inputs for kis, not arrays, and that outputs are real floats
    assert type(ns(1,1,1.,params.params())[0]) == np.float64