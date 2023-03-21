import numpy as np
from sklearn.neighbors import KDTree


def LPM_cosF(neighborX, neighborY, lbd, vec, d2, tau, K):

    L = neighborX.shape[0]
    C = 0
    Km = np.array([K+2, K, K-2])
    M = len(Km)

    for KK in Km:
        neighborX = neighborX[:,1:KK+1]
        neighborY = neighborY[:,1:KK+1]
        ## This is a loop implementation for computing c1 and c2, much slower but more readable
        # ni = np.zeros((L,1))
        # c1 = np.zeros((L,1))
        # c2 = np.zeros((L,1))
        # for i in range(L):
        #     inters = np.intersect1d(neighborX[i,:], neighborY[i,:])  # 返回数组中相同的值 即邻域中相同的点
        #     ni[i] = len(inters)
        #     c1[i] = KK - ni[i]
        #     cos_sita = np.sum(vec[inters, :]*vec[i,:],axis=1)/np.sqrt(d2[inters]*d2[i]).reshape(ni[i].astype('int').item(), 1)
        #     ratio = np.minimum(d2[inters], d2[i])/np.maximum(d2[inters], d2[i])
        #     ratio = ratio.reshape(-1,1)
        #     label = cos_sita*ratio < tau
        #     c2[i] = np.sum(label.astype('float64'))

        neighborIndex = np.hstack((neighborX,neighborY))
        index = np.sort(neighborIndex,axis=1)
        # np.diff: a[n]-a[n-1]
        temp1 = np.hstack((np.diff(index,axis = 1),np.ones((L,1))))
        temp2 = (temp1==0).astype('int')
        ni = np.sum(temp2, axis=1)
        c1 = KK - ni
        # index.shape[1]=2KK
        temp3 = np.tile(vec.reshape((vec.shape[0],1,vec.shape[1])),(1,index.shape[1],1))*vec[index, :]
        temp4 = np.tile(d2.reshape((d2.shape[0],1)),(1,index.shape[1]))
        temp5 = d2[index]*temp4
        cos_sita = np.sum(temp3,axis=2).reshape((temp3.shape[0],temp3.shape[1]))/np.sqrt(temp5)
        ratio = np.minimum(d2[index], temp4)/np.maximum(d2[index], temp4)
        label = cos_sita*ratio < tau
        label = label.astype('int')
        c2 = np.sum(label*temp2,axis=1)

        C = C + (c1 + c2)/KK
    # C/M 和 C = C + (c1 + c2)/KK*M 等效
    idx = np.where((C/M) <= lbd)  # np.where返回满足条件的元素索引 此处对应公式14
    # idx对应的索引的pi为1 即内点集I的点集索引
    return idx[0], C

def LPM_filter(X, Y):
    lambda1 = 0.8  # 0.8
    lambda2 = 0.5  # 0.5
    numNeigh1 = 6
    numNeigh2 = 6
    tau1 = 0.2  # 0.2
    tau2 = 0.2  # 0.2

    vec = Y - X
    d2 = np.sum(vec**2, axis=1)

    treeX = KDTree(X)
    _, neighborX = treeX.query(X, k=numNeigh1+3)
    treeY = KDTree(Y)
    _, neighborY = treeY.query(Y, k=numNeigh1+3)

    idx, C = LPM_cosF(neighborX, neighborY, lambda1, vec, d2, tau1, numNeigh1)
    # 此处求的是I0，然后用I0代替初始匹配点集S
    # 然后重复上述步骤计算真正的内点集I*
    if len(idx) >= numNeigh2 + 4:
        treeX2 = KDTree(X[idx,:])  # X[idx,:]=I0
        _, neighborX2 = treeX2.query(X, k=numNeigh2+3)
        treeY2 = KDTree(Y[idx,:])
        _, neighborY2 = treeY2.query(Y, k=numNeigh2+3)
        neighborX2 = idx[neighborX2]
        neighborY2 = idx[neighborY2]
        idx, C = LPM_cosF(neighborX2, neighborY2, lambda2, vec, d2, tau2, numNeigh2)

    mask = np.zeros((X.shape[0],1))  # X.shape[0]=L
    mask[idx] = 1

    return mask.flatten().astype('bool')



