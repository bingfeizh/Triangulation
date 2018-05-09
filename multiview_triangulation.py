#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:41:10 2017

@author: bingfei
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random

for number in range(1):
    
    img1 = cv2.imread('/Users/bingfei/project/Matching/Fairbank/image/1/'+str(number)+'.jpg',0)  #input image 1
    img2 = cv2.imread('/Users/bingfei/project/Matching/Fairbank/image/2/'+str(number)+'.jpg',0) #input image 2
    img3 = cv2.imread('/Users/bingfei/project/Matching/Fairbank/image/3/'+str(number)+'.jpg',0) #input image 3

    descripter = cv2.xfeatures2d.SIFT_create()
    #descripter = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = descripter.detectAndCompute(img1,None)
    kp2, des2 = descripter.detectAndCompute(img2,None)
    kp3, des3 = descripter.detectAndCompute(img3,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches1 = flann.knnMatch(des1,des2,k=2)
    matches2 = flann.knnMatch(des1,des3,k=2)


    good = []
    pts12 = []
    pts13 = []
    pts21 = []
    pts31 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches1):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts21.append(kp2[m.trainIdx].pt)
            pts12.append(kp1[m.queryIdx].pt)
            
    for ii,(mm,nn) in enumerate(matches2):
        if mm.distance < 0.8*nn.distance:
            good.append(m)
            pts31.append(kp3[mm.trainIdx].pt)
            pts13.append(kp1[mm.queryIdx].pt)
                    
    pts12=np.float32(pts12)
    pts13=np.float32(pts13)
    pts21=np.float32(pts21)
    pts31=np.float32(pts31)
                    
    pts1=[]
    pts2=[]
    pts3=[]
    for i in range(len(pts12)):
        for j in range(len(pts13)):
            if pts12[i].tolist()==pts13[j].tolist():
                pts1.append(pts12[i])
                pts2.append(pts21[i])
                pts3.append(pts31[j])


    pts1=np.array(pts1).T
    pts2=np.array(pts2).T
    pts3=np.array(pts3).T

    center1=np.mean(pts1,1)
    center2=np.mean(pts2,1)
    center3=np.mean(pts3,1)

    center1.resize(2,1)
    pts1c=pts1-center1
    center2.resize(2,1)
    pts2c=pts2-center2
    center3.resize(2,1)
    pts3c=pts3-center3
    
    pts1c = np.vstack((pts1c,np.ones(pts1c.shape[1])))
    pts2c = np.vstack((pts2c,np.ones(pts2c.shape[1])))
    pts3c = np.vstack((pts3c,np.ones(pts3c.shape[1])))
    
    M=np.zeros((9,pts1c.shape[1]))
    M[:3,:]=pts1c
    M[3:6,:]=pts2c
    M[6:,:]=pts3c
    
    U,S,V = np.linalg.svd(M)
    
    P1=U[:3,:3]
    P2=U[3:6,:3]
    P3=U[6:,:3]
    
    X=V.T[:,:3]
    
    c=[]         
    for cl in range(X.shape[0]):
        c.append([random.random(),random.random(),random.random()])
                       
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for p3 in range(X.shape[0]):
        ax.scatter(X[p3][0],X[p3][1],X[p3][2],color=tuple(c[p3]))
        ax.text(X[p3][0],X[p3][1],X[p3][2],str(p3))
        #plt.axis('off')
    plt.savefig('/Users/bingfei/project/Triangulation/3D/'+str(number)+'.png')
    plt.close()

    plt.figure()
    plt.imshow(img1)
    plt.gray()
    #plt.plot(x1p[0],x1p[1],'o')
    for pl in range(X.shape[0]):
        plt.scatter(pts1[0][pl],pts1[1][pl],color=tuple(c[pl]))
        plt.text(pts1[0][pl],pts1[1][pl],str(pl))
        #plt.scatter(x1p[0][pl],x1p[1][pl],color=tuple(c[pl]))
        plt.axis('off')
    plt.savefig('/Users/bingfei/project/Triangulation/1/'+str(number)+'.png')
    plt.close()

    plt.figure()
    plt.imshow(img2)
    plt.gray()
    #plt.plot(x2p[0],x2p[1],'o')
    for pr in range(X.shape[0]):
        plt.scatter(pts2[0][pr],pts2[1][pr],color=tuple(c[pr]))
        plt.text(pts2[0][pr],pts2[1][pr],str(pr))
        #plt.scatter(x2p[0][pl],x2p[1][pl],color=tuple(c[pl]))
        plt.axis('off')
    plt.savefig('/Users/bingfei/project/Triangulation/2/'+str(number)+'.png')
    plt.close()

    plt.figure()
    plt.imshow(img3)
    plt.gray()
    #plt.plot(x2p[0],x2p[1],'o')
    for pr in range(X.shape[0]):
        plt.scatter(pts3[0][pr],pts3[1][pr],color=tuple(c[pr]))
        plt.text(pts3[0][pr],pts3[1][pr],str(pr))
        #plt.scatter(x2p[0][pl],x2p[1][pl],color=tuple(c[pl]))
        plt.axis('off')
    plt.savefig('/Users/bingfei/project/Triangulation/3/'+str(number)+'.png')
    plt.close()
