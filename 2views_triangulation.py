#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:01:49 2017

@author: bingfei
"""

import cv2
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random


def compute_P_from_fundamental(F):
    """ Computes the second camera matrix (assuming P1 = [I 0])
    from a fundamental matrix. """
    e = compute_epipole(F.T) # left epipole 
    Te = skew(e)
    return np.vstack((np.dot(Te,F.T).T,e)).T
    
def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """
    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

def compute_epipole(F):
    """ Computes the (right) epipole from a
    fundamental matrix F.
    (Use with F.T for left epipole.) """
    # return null space of F (Fx=0)
    U,S,V = np.linalg.svd(F) 
    e = V[-1]
    return e/e[2]

def triangulate_point(x1,x2,P1,P2): 
    """ Point pair triangulation from
    least squares solution. """
    
    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2
    U,S,V = np.linalg.svd(M)
    X = V[-1,:4]
    return X / X[3]

def triangulate(x1,x2,P1,P2):
    """ Two-view triangulation of points in
    x1,x2 (3*n homog. coordinates). """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don’t match.")
    X = [triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)] 
    return np.array(X).T

class Camera(object):
    """ Class for representing pin-hole cameras. """
    def __init__(self,P):
        """ Initialize P = K[R|t] camera model. """ 
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center
    
    def project(self,X):
        """ Project points in X (4*n array) and normalize coordinates. """
        x = np.dot(self.P,X) 
        for i in range(3):
            x[i] /= x[2] 
        return x


img1 = cv2.imread('/Users/bingfei/project/Triangulation/1.jpg',0)  #input image 1
img2 = cv2.imread('/Users/bingfei/project/Triangulation/2.jpg',0) #input image 2
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

pts1=pts1.T
pts2=pts2.T

pts1 = np.vstack((pts1,np.ones(pts1.shape[1])))
pts2 = np.vstack((pts2,np.ones(pts2.shape[1])))

P1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P2=compute_P_from_fundamental(F)

# triangulate inliers and remove points not in front of both cameras
X = triangulate(pts1,pts2,P1,P2) 

c=[]         
for cl in range(X.shape[1]):
    c.append([random.random(),random.random(),random.random()])
                   
fig = plt.figure()
ax = fig.gca(projection='3d')
for p3 in range(X.shape[1]):
    ax.scatter(-X[0][p3],X[1][p3],X[2][p3],color=tuple(c[p3]))
    ax.text
#plt.axis('off')

# project 3D points
cam1 = Camera(P1)
cam2 = Camera(P2)
x1p = cam1.project(X)
x2p = cam2.project(X)

plt.figure()
plt.imshow(img1)
plt.gray()
#plt.plot(x1p[0],x1p[1],'o')
for pl in range(X.shape[1]):
    plt.scatter(pts1[0][pl],pts1[1][pl],color=tuple(c[pl]))
    plt.text(str(pl),(pts1[0][pl],pts1[1][pl]))
    #plt.scatter(x1p[0][pl],x1p[1][pl],color=tuple(c[pl]))
plt.axis('off')

plt.figure()
plt.imshow(img2)
plt.gray()
#plt.plot(x2p[0],x2p[1],'o')
for pr in range(X.shape[1]):
    plt.scatter(pts2[0][pr],pts2[1][pr],color=tuple(c[pr]))
    #plt.scatter(x2p[0][pl],x2p[1][pl],color=tuple(c[pl]))
plt.axis('off')

plt.show()