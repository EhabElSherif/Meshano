from scipy import linalg
import  numpy as np

""" Class for representing pin-hole cameras. """
class Camera(object):

    def __init__(self,P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center

    def project(self,X):
        """ Project points in X (4*n array)(x,y,z,t) to x (3*n array)(x,y,t) and normalize coordinates. """
        x = np.dot(self.P,X) 
        for i in range(3):
            x[i] /= x[2]
        return x

    #TODO there is a sign error
    def factor(self):
        """ Factorize the camera matrix into K,R,t as P = K[R|t]. """
        # factor first 3*3 part
        K,R = linalg.rq(self.P[:,:3])
        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:   
            T[1,1] *= -1
        self.K = np.dot(K,T)
        self.R = np.dot(T,R) # T is its own inverse
        self.t = np.dot(linalg.inv(self.K),self.P[:,3])
        return self.K, self.R, self.t

    def center(self):
        """ Compute and return the camera center. """
        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.factor()
            self.c = -np.dot(self.R.T,self.t)
            return self.c
    
    
#I think not needed #TODO
def rotation_matrix(a):
    """ Creates a 3D rotation matrix for rotation
    around the axis of the vector a. """
    R = np.eye(4)
    R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return R
