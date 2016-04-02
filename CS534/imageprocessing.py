#-------------------------------------------------------------------------------
# Name:        image processing
# Purpose:
#
# Author:      Shi
#
# Created:     31/03/2016
# Copyright:   (c) Administrator 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from PIL import Image
import numpy as np
GRID=8

class readImg():
    def __init__(self, str):
        im = Image.open(str)
        self.width, self.height = im.size

        p = np.array(im)
        self.imageMatrix=np.zeros([GRID,GRID])
        imageMatrixWhole=np.zeros([self.height,self.width])
        for i in range(p.shape[0]-1):
            for j in range(p.shape[1]-1):
                b=np.array_equal(p[i][j],np.array([255,255,255]))
                if b:
                    imageMatrixWhole[i][j]=0
                else:
                    imageMatrixWhole[i][j]=1
                    #print i,j, p.shape, self.convertTo(i,j)
                    self.imageMatrix[self.convertTo(i,j)[0], self.convertTo(i,j)[1]]+=1

    def convertTo(self,i,j):
        m=self.height/GRID
        n=self.width/GRID
        return [i/n,j/m]

    def getFeature(self):
        return self.imageMatrix.ravel()/100