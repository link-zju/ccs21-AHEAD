import numpy as np
import math
from scipy import stats
from scipy.stats import norm

def OUE(epsilon, VectorSum, dataSize):
    p = 0.5
    q = 1.0 / (1 + math.exp(epsilon))

    # step 1: initial count
    X = VectorSum.copy()
    ESTIMATE_X = np.random.binomial(X, p)

    # step 2: noise
    X = VectorSum.copy()
    X = dataSize - X

    ESTIMATE_X += np.random.binomial(X, q)
    # step 3: normalize
    a = 1.0 / (p - q)
    b = dataSize * q / (p - q)

    return (a * ESTIMATE_X - b)/dataSize

def OUE_Noise(epsilon, VectorSum, dataSize):
#this function perturbs the orignal VectorSum, and return
    p = 0.5
    q = 1.0 / (1 + math.exp(epsilon))

    # step 1: initial count
    X = VectorSum.copy()
    Noise_X = np.random.binomial(X, p)

    # step 2: noise
    X = VectorSum.copy()
    X = dataSize - X

    Noise_X += np.random.binomial(X, q)
    return Noise_X

def Base(Noise_Vector,DomainSize,Datasize,p,q):
#this function estimate the unbiased vector of the noise_vector
    Base_Vector = np.zeros(DomainSize)
    for i in range(DomainSize):
        Base_Vector[i] = (Noise_Vector[i]/Datasize - q) / (p - q)
    return Base_Vector

def Base_Pos(Noise_Vector,DomainSize,Datasize,p,q):
    BasePos_Vector = Base(Noise_Vector,DomainSize,Datasize,p,q)
    for i in range(DomainSize):
        if BasePos_Vector[i] < 0:
            BasePos_Vector[i] = 0
    return BasePos_Vector

def Norm_Sub(Noise_Vector, DomainSize, Datasize, p, q):
    BasePos_Vector = Base_Pos(Noise_Vector, DomainSize, Datasize, p, q)
    NormSub_Vector = np.zeros(DomainSize)
    RemainCount = len(BasePos_Vector[BasePos_Vector > 0])
    delta = (1 - sum(BasePos_Vector)) / RemainCount

    for i in range(DomainSize):
        if BasePos_Vector[i] > 0:
            NormSub_Vector[i] = BasePos_Vector[i] + delta
    return NormSub_Vector
