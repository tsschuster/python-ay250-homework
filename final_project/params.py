import scipy
import scipy.special
import numpy as np
import pickle

import hoppings

class params:
    """
    A class whose attributes contain all of the variables describing a given dipolar hopping model
    Because some important variables are computed after initialization as functions of input variables, always re-run param.updateHoppings() if changing any parameters after instantiation!
    
    Input attributes:
    
    R: maximum distancen(in a given dimension) hopping to include
    vec: List of length 5 or 13 containing a list of values to be set, in order [mu,a,b,g_eo,g_eo_SL,ts[0],ts[1],g[0],g[1],g[2],g_SL[0],g_SL[1],g_SL[2]]. If supplied, will override the values of the following inputs. (The list of inputs is convenient for some optimization/minimization functions used.)
    mu: difference in site A and B chemical potential
    a: lattice length in z direction (length in x & y direction set to 1)
    b: inter-sublattice distance (along z direction)
    g_eo: amplitude of even-odd xy Floquet modulation
    g_eo_SL: site-independent difference between sublattices in xy Floquet modulation
    ts: list of length 2 - the two "switch times" in the 3-part piecewise constant gradient z Floquet modulation
    g: list of length 3 - the 3 amplitudes in the gradient z Floquet modulation
    g_SL: list of length 3 - the 3 amplitude differences between sublattices in the gradient z Floquet modulation
    strobeQ: setting to False uses a different, continuous Floquet modulation for the z gradient
    joelQ: if True, loads all hoppings from those saved for Joel's model Hopf insulator
    joel_delmu: adjustment to the chemical potential in Joel's model
    
    Additional useful attributes:
    
    rs: a list of all lattice sites (as 3-tuples (x,y,z)) with nonzero hopping
    xs,ys,zs: the above list, but only with the x/y/z component
    hopsAA: intra-sublattice hoppings, in order of the above lattice sites
    hopsAB: inter-sublattice hoppings, in order of the above lattice sites
    """
    
    def __init__(self,R=2,vec = "None",mu=0,a=1,b=.6,g_eo=0,g_eo_SL=0,ts = [0,0],g=[0,0,0],g_SL=[0,0,0],strobeQ = True,joelQ = False,joel_delmu=0):
        #If parameters for Joel's model
        self.joelQ = joelQ
        self.joel_delmu = joel_delmu
        if self.joelQ:
            self.R = 2
            self.rs,self.hopsAA,self.hopsAB = pickle.load(open("Joels_model_hoppings.p","rb"))
            self.xs,self.ys,self.zs = [np.asarray([r[i] for r in self.rs]) for i in range(3)]
            return
            
        self.R=R
        self.mu=mu
        self.a=a
        self.b=b
        self.g_eo = g_eo
        self.g_eo_SL = g_eo_SL
        self.ts = ts
        self.g = g
        self.g_SL = g_SL
        self.strobeQ = strobeQ

        self.vec = [mu,a,b,g_eo,g_eo_SL] + ts + g + g_SL
        
        if vec != "None":
            self.vec = vec
            self.mu,self.a,self.b,self.g_eo,self.g_eo_SL = vec[:5]
            if len(vec) > 5:
                self.ts = vec[5:7]
                self.g = vec[7:10]
                self.g_SL = vec[10:13]

        #Set up displacements for computing hoppings
        self.sites = range(0,self.R+1)+range(-self.R,0)
        self.sites.reverse()
        #meshgrid of hoppings - z potentially could be different (because we floquet truncate)
        xs,ys,zs = np.meshgrid(self.sites,self.sites,self.sites)

        self.xs = xs.flatten(); self.ys = ys.flatten(); self.zs = zs.flatten()
        self.rs = [(x,y,z) for x,y,z in zip(self.xs,self.ys,self.zs)]
                    
        #Pre-compute hoppings (to speed things up in calling energy, ns for different k-pts)
        self.updateHoppings()
        
    def updateHoppings(self,updateZ=True):
        """
        Calculates all hoppings for current parameters and stores as self.hoppings
        Also sets self.dists to be array of the displacement vectors for each hopping
        Input:
            updateZ = False will not recompute the Z Floquet damping - if beta & beta_SL are unchanged this will save time as these are expensive to compute
        """
        
        #Floquet damping: Calculate betas from gs here
        #XY damping:
        self.beta_eo = scipy.special.jv(0,self.g_eo)
        self.beta_eo_SL = scipy.special.jv(0,self.g_eo_SL)
        
        halfdamp = (1-self.beta_eo)/2.
        self.dampingXYAA = (1-halfdamp) + halfdamp*(-1.)**(self.xs + self.ys)
        halfdamp = (1-self.beta_eo_SL)/2.
        self.dampingXYAB = (1-halfdamp) + halfdamp*(-1.)**(self.xs + self.ys)
        
        #Z damping:
        if updateZ:
            if self.strobeQ:
                self.beta = getBeta(self.ts,self.g,[0 for _ in self.g],self.R)
                self.beta_SL = getBeta(self.ts,self.g,self.g_SL,self.R)
            else:
                self.beta = [(1./(2*np.pi))*scipy.integrate.quad(lambda t: np.cos(dz*( sum([self.g[k]*np.sin((k+1)*t) for k in range(0,len(self.g))]) ) + 0.),-np.pi,np.pi)[0] for dz in self.sites]
                self.beta_SL = [(1./(2*np.pi))*scipy.integrate.quad(lambda t: np.cos(dz*(sum([self.g[k]*np.sin((k+1)*t) for k in range(0,len(self.g))])) + (sum([self.g_SL[k]*np.sin((k+1)*t) for k in range(0,len(self.g_SL))])) ),-np.pi,np.pi)[0] for dz in self.sites]
        
        self.dampingZAA = self.beta*len(self.sites)**2 #Copy to go from sites_z-length vec to total_sites-length vec
        self.dampingZAB = self.beta_SL*len(self.sites)**2
        
        #Compute hoppings (compute 0th element r = 0,0,0 separately for tAA - self-hopping should be mu)
        self.hopsAA = self.dampingZAA*self.dampingXYAA*np.append(hoppings.tAA(self,self.xs[:-1],self.ys[:-1],self.zs[:-1]),-self.mu)
        self.hopsAB = self.dampingZAB*self.dampingXYAB*hoppings.tAB(self,self.xs,self.ys,self.zs)
        
        
def getBeta(ts,g,g_SL,R, g_static = 1):
    """
    Get z-dependent damping coefficients beta from Floquet modulation parameters
    Input:
        ts, g, g_SL, R: as defined in params class description
    Output:
        beta: 1D array of damping coefficients for z = -R,-R+1,...,R
    """
    ts_all = [0] + list(ts) + [np.pi] + [2*np.pi - ti for ti in reversed(list(ts))] + [2*np.pi] #all times at which detuning is shifted
    alphas_no_z = [(g_static + gi) for gi in g] + [(g_static - gi) for gi in reversed(g)] #all values of detuning (from time t[i] to t[i+1])
    alphas_SL = [gi for gi in g_SL] + [-gi for gi in reversed(g_SL)]

    beta = []
    for z in range(1,R+1) + range(-R,1):
        alphas = [z*alpha + alpha_SL for alpha,alpha_SL in zip(alphas_no_z,alphas_SL)]
        #Integral is piecewise sum of integrals of cos( alpha * t ) -> sin( alpha * t)/ alpha
        #Define phase accumulated inside cos up to each time:
        phis = [sum([(alpha)*(t-tm1) for alpha,t,tm1 in zip(alphas[:k],ts_all[1:(k+1)],ts_all[:k])]) for k in range(len(alphas)+1)]
        #Add RHS limits, subtract LHS limits
        Ipieces = [(np.sin(phi2)-np.sin(phi1))/(alpha+.00000000001) for alpha,phi1,phi2 in zip(alphas,phis[:-1],phis[1:])]
        tol = .000001
        for i,a in enumerate(alphas):
            if abs(a) < tol:
                Ipieces[i] = np.cos(phis[i])*(ts_all[i+1]-ts_all[i])
                #print("Alpha was zero")
        I = (1./(2*np.pi))*sum(Ipieces)
        beta.append(I)
    return(beta)
        
        