import numpy as np
from scipy import linalg

#TODO #Not tested
def compute_P_from_essential(E):
    """ Computes the second camera matrix (assuming P1 = [I 0])
    from an essential matrix. Output is a list of four
    possible camera matrices. """
    # make sure E is rank 2
    U,S,V = linalg.svd(E)
    if linalg.det(np.dot(U,V))<0:
        V = -V
    E = np.dot(U,np.dot(diag([1,1,0]),V))
    # create matrices (Hartley p 258)
    Z = skew([0,0,-1])
    W = array([[0,-1,0],[1,0,0],[0,0,1]])
    # return all four solutions
    P2 = [vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
    vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T,
    vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
    vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]
    return P2

def compute_essential_from_F(F):

    return E

def triangulate_point(x1,x2,P1,P2):
    """ Point pair triangulation from least squares solution. """
    M = zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2
    U,S,V = linalg.svd(M)
    X = V[-1,:4]
    return X / X[3]

def triangulate(x1,x2,P1,P2):
    """ Two-view triangulation of points in x1,x2 (3*n homog. coordinates). """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don’t match.")
    X = [ triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return array(X).T

# compute camera matrices (P2 will be list of four solutions)
P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
#E = P1 #test
P2 = compute_P_from_essential(E)
#From the list of camera matrices, we pick the one that has the most scene points
#in front of both cameras after triangulation.

# pick the solution with points in front of cameras
ind = 0
maxres = 0
for i in range(4):
# triangulate inliers and compute depth for each camera
    X = triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[i])
    d1 = dot(P1,X)[2]
    d2 = dot(P2[i],X)[2]
    if sum(d1>0)+sum(d2>0) > maxres:
        maxres = sum(d1>0)+sum(d2>0)
        ind = i
        infront = (d1>0) & (d2>0)
# triangulate inliers and remove points not in front of both cameras
X = triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[ind])
X = X[:,infront]