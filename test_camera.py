import camera
import numpy as np

""" Test the factor func ..should print the same results"""
K = np.array([[1000,0,500],[0,1000,300],[0,0,1]])
tmp = camera.rotation_matrix([0,0,1])[:3,:3]
Rt = np.hstack((tmp,np.array([[50],[40],[30]])))
cam = camera.Camera(np.dot(K,Rt))
print(K,Rt)
print(cam.factor())
