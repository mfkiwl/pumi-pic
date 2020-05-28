import sys,os
import numpy as np
import io,libconf
import math
from numpy import linalg as LA

import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LogNorm

class process:
    """
    calculate / plot results
    """

    def __init__(self, path = './'):
        if not (path[-1] == '/'):
            path += '/'
        self.path = path
        self.config = {}
        self.files = ['history.nc', 'compare-history.nc']

        self.data = {}
        for name in self.files:
            if os.path.isfile(name):
                self.data[name] = openNCfile(name)

        self.calculateAll()

    def calculateAll(self):
        self.calculatePathNorm()

    def calculatePathNorm(self):
        if 'history.nc' not in self.data:
            print( 'history.nc not available')
            return
        x = self.data['history.nc']['x']
        y = self.data['history.nc']['y']
        z = self.data['history.nc']['z']
        x[x<0] = 0
        y[y<0] = 0
        z[z<0] = 0
        if 'compare-history.nc' in self.data:
            x2 = self.data['compare-history.nc']['x']
            y2 = self.data['compare-history.nc']['y']
            z2 = self.data['compare-history.nc']['z']
            x2[x2 <= 0] = 0
            y2[y2 <= 0] = 0
            z2[z2 <= 0] = 0
            diffx = x2 - x
            normx = LA.norm(diffx)
            diffy = y2 - y
            normy = LA.norm(diffy)
            diffz = z2 - z
            normz = LA.norm(diffz)
            normx1=LA.norm(x)
            normy1 = LA.norm(y)
            normz1 = LA.norm(z)
            p=3
            print('positions diff. norm', round(normx,p), round(normy,p), round(normz,p))
            print('history',round(normx1,p),round(normy1,p),round(normz1,p))
            print('percent',round(normx/normx1*100,p), round(normy/normy1*100,p),round(normz/normz1*100,p))

def openNCfile(Filename):
    """
    Open netcdf file 'Filename' and parse its contents into dictionary
    return dictionary
    """
    from netCDF4 import Dataset
    from warnings import filterwarnings
    filterwarnings('ignore')

    D = {}
    data = Dataset(Filename, 'r')

    # Dimensions
    varnames = data.dimensions.keys()
    for n in varnames:
        D[n] = len(data.dimensions[n])

    # Variables
    varnames = data.variables.keys()
    for n in varnames:
        if(data.variables[n].size > 1):   # arrays (all types)
            D[n] = np.array(data.variables[n][:])
            if(data.variables[n].dtype == 'S1'): # character array
                if(data.variables[n].size == data.variables[n].shape[0]):  # 1-D                                           
                        D[n] = ''.join(data.variables[n][:]).strip()
                else:                                                      # 2-D
                        D[n] = []
                        for i in xrange(data.variables[n].shape[0]):
                                D[n].append(''.join(data.variables[n][i,:]).strip())
        else:                                                                     # single variable
            if(data.variables[n].dtype in ['float','float32','float64']):   # float
                    D[n] = np.float64(data.variables[n][:])
            elif(data.variables[n].dtype in ['int','int32','int64','long']):# int
                    D[n] = int(data.variables[n][:])
            elif(data.variables[n].dtype == 'S1'):                           # char
                    try: D[n] = ''.join(data.variables[n][:]) #at fixed bndy: mgrid_mode exists but is masked -> error
                    except: D[n] = data.variables[n][:]
            else:                                                     # unknown
                    print( 'unknown datatype in', Filename, ', variable:')
                    print( n, ' type:',data.variables[n].dtype, 'size:', data.variables[n].size)
                    D[n] = data.variables[n][:]
    return D


def main():
    process()

if __name__ == '__main__':
    main()
