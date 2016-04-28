import kernel
import numpy as np
import multiclass

def main():
    #str="features_all_without_skeleton"
    #str="features_density_direction"
    #str="features_all"
    #str="features_profiling"
    str="features_profiling_density"
    data = np.genfromtxt(str+".csv",delimiter=",")

    X=data[:,:-1]
    y=data[:,-1]


    X_new=X

    k=kernel.Kernel.gaussian(6)
    m=multiclass.multiclass(X_new,y,k,20,str)


    print "training done..."

if __name__ == '__main__':
    main()