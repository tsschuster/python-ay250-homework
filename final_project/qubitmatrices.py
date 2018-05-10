import numpy as np
import scipy as sp
import scipy.linalg

Id = np.identity(2)

#define useful single qubit matrices
X = np.array([[0, 1],[1, 0]])
Y = np.array([[0, -1j],[1j, 0]])
Z = np.array([[1,0],[0,-1]])

gen1q = [Id,X,Y,Z]

#define useful single qubit states
Zero = np.array([[1.0],[0.0]])
One = np.array([[0.0],[1.0]])
Plus = (1/np.sqrt(2))*(Zero + One)
Minus = (1/np.sqrt(2))*(Zero - One)
PlusI = (1/np.sqrt(2))*(Zero + 1j*One)
MinusI = (1/np.sqrt(2))*(Zero - 1j*One)
BlochSphere = [Zero,One,Plus,Minus,PlusI,MinusI]

