import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom


class indexmap:
    def __init__(self,b,pruntype,dim):
        self.pruntype = pruntype #type of pruning
        self.b = b #pruning parameter
        self.dim = dim #number of dimensions. For now it is 3

    def get_basis_size(self):
            return  int(binom(self.b + 3, 3)) #(self.b+1) * (self.b +2 ) /2

    def get_pruning_func(self):
        "generate pruning function"
        alpha1 = 1
        alpha2 = 1
        alpha3 = 1
        return alpha1,alpha2,alpha3
            


    def gen_map(self):
        #print(self.get_basis_size())
        maparray = [] #np.zeros((int(self.get_basis_size()),4),dtype=int)
        alpha1,alpha2,alpha3 = self.get_pruning_func()
        i = 0 
        for i1 in range(self.b):
            for i2 in range(self.b):
                for i3 in range(self.b):
                    #print(str(i1)+' '+str(i2)+' '+str(i3))
                    if  i1 * alpha1 + i2 * alpha2 + i3 * alpha3 <= self.b:
                        maparray.append([i1,i2,i3,i+1])
                        i+=1

        return maparray 


if __name__=="__main__":



    simpleMap = indexmap(2,'simple',3)
    #print(simpleMap.gen_map())