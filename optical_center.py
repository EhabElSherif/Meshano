import numpy as np
import sys


'''
P = KR[I|-C]
optical center dimention: [x y z t]
'''
def getOpticalCenter(m_projection):

    if(m_projection.shape != (3,4)):
        print("Projection matrix should be of size 3*4")
        sys.exit()

    #TODO #orthographic case #t=0?

    m_KR = m_projection[:,:3]
    col_4 = m_projection[:,3]
    col_4 = col_4*-1

    m_KR_inverse = np.linalg.inv(m_KR)
    optical_center = np.dot(m_KR_inverse, col_4)
    optical_center = np.append(optical_center,1)
    
    return optical_center