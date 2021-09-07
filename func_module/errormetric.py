import math

def MNAE_metric(realValue, estimatedValue, sumS):
    return abs(estimatedValue - realValue)/sumS

def MRE_metric(realValue, estimatedValue):
    return abs(estimatedValue - realValue)/abs(estimatedValue)

def MSE_metric(MSEList):
    _temp = 0
    for val in MSEList:
        _temp = _temp + val**2
    return _temp/len(MSEList)
