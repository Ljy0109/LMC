import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import F_Score
from read_H import read_H
from sklearn.neighbors import KDTree
from itertools import combinations
from LPM import LPM_filter


def SIFT(img1, img2, num_of_points):

    sift = cv2.SIFT_create(num_of_points)
    kp1 = sift.detect(img1)
    kp2 = sift.detect(img2)
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in raw_matches])
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in raw_matches])

    return src_pts, dst_pts


def construct_neighbor(m1, K, Neighbor):

    treem1 = KDTree(Neighbor)
    _, neighborm1 = treem1.query(m1, K + 1)
    return neighborm1[:, 1:K + 1]


def calculate_dist(m1, m2, K, index, tau):
    """
    calculation the reprojection error
    :param m1: feature points in Ix
    :param m2: fearure points in Iy
    :param K: the number of neighborhood points for each feature point
    :param index: the index of the neighborhood U in S (S={(m1, m2)})
    :param tau: threshold of the LMC method
    :return:
        best_index:  the index of the matching points obtained by LMC method in S.
        best_four_neigh_x, best_four_neigh_y: the index of the points of the reliable MHUs in S
    """
    index_X = construct_neighbor(m1, K, m1[index, :])
    index_Y = construct_neighbor(m2, K, m2[index, :])
    index_X = index[index_X]  # N_{x_i}^K = m1[index_X[i]]
    index_Y = index[index_Y]  # N_{y_i}^K = m2[index_Y[i]]
    N = m1.shape[0]

    # initialization
    best_four_neigh_x = np.zeros((m1.shape[0], 4), int)  # Store points in the reliable MHU
    best_four_neigh_y = np.zeros((m1.shape[0], 4), int)
    best_index = np.array([], int)  # index of inlier set in S
    for i in range(N):
        list_x = index_X[i, :].tolist()
        list_y = index_Y[i, :].tolist()
        same_index = set(list_x) & set(list_y)
        same_num = len(same_index)
        if same_num < 4:
            continue
        same_index_x = np.zeros(same_num, int)
        same_index_y = np.zeros(same_num, int)
        # construct Ri={(m1[index_X[i, same_index_x]], m2[index_Y[i, same_index_y]])}
        ti = -1
        for j in same_index:
            ti = ti + 1
            same_index_x[ti] = list_x.index(j)
            same_index_y[ti] = list_y.index(j)

        # construct MHUs
        index_std = np.arange(0, same_num, 1, int)
        four_neighbor_index = np.array(list(combinations(index_std, 4)))

        for j in range(four_neighbor_index.shape[0]):  # m <= Mi
            temp_index_x = index_X[i, [same_index_x[four_neighbor_index[j, :]]]]
            temp_index_y = index_Y[i, [same_index_y[four_neighbor_index[j, :]]]]
            four_neigh_x = np.float32(m1[temp_index_x, :]).reshape(4, 2)
            four_neigh_y = np.float32(m2[temp_index_y, :]).reshape(4, 2)

            # calculate the homography matrix using MHU_m
            h_x = cv2.getPerspectiveTransform(four_neigh_x, four_neigh_y)

            m1_re = np.float32(m1[i, :]).reshape(-1, 1, 2)
            trans_x = cv2.perspectiveTransform(m1_re, h_x).ravel()
            # calculate the reprojection error e_m(xi,yi)
            temp_dist = np.sqrt((trans_x[0] - m2[i, 0]) ** 2 + (trans_x[1] - m2[i, 1]) ** 2)
            if temp_dist < tau:
                if abs(h_x[0, 0]) < 0.000001 or abs(h_x[1, 1]) < 0.000001:
                    continue
                ##### The following four lines are not necessary #####
                x_set = set(map(tuple, four_neigh_x))
                y_set = set(map(tuple, four_neigh_y))
                if len(x_set) != 4 or len(y_set) != 4:
                    continue
                ##### If you want to sacrifice a little precision to shorten the run time, #####
                ##### you can comment out those four lines #####

                best_index = np.append(best_index, i)
                best_four_neigh_x[i, :] = temp_index_x
                best_four_neigh_y[i, :] = temp_index_y
                break

    return best_index, best_four_neigh_x, best_four_neigh_y


def LMC_LPM(m1, m2, K, tau):
    # the main function of LMC_LPM method

    # Build the neighborhood U using LPM
    # the parameters of LPM is the default
    # index is the index of the neighborhood U in S
    mask = LPM_filter(m1, m2)
    index = np.where(mask==True)[0]
    if index.size < K:
        print('error : There are not enough neighborhood points.\n Program terminated')
        return [], [], []

    # calculate the reprojection error
    best_index, best_four_neigh_x, best_four_neigh_y = calculate_dist(m1, m2, K, index, tau)
    if best_index.size == 0:
        print('error : There are not enough matches.\n Program terminated')
        return [], [], []

    return best_index, best_four_neigh_x, best_four_neigh_y


def LMC_RANSAC(m1, m2, K, tau, alpha):
    # the main function of LMC_LPM method

    # construct the neighborhood U using RANSAC
    # index is the index of the neighborhood U in S
    h_x, mask = cv2.findHomography(m1, m2, cv2.RANSAC, alpha)
    index = np.where(mask == 1)[0]
    if index.size < K + 1:
        print('error : There are not enough neighborhood points.\n Program terminated')
        return np.array([]), np.array([]), np.array([])

    # calculate the reprojection error
    best_index, best_four_neigh_x, best_four_neigh_y = calculate_dist(m1, m2, K, index, tau)
    if best_index.size == 0:
        print('error : There are not enough matches.\n Program terminated')
        return np.array([]), np.array([]), np.array([])

    return best_index, best_four_neigh_x, best_four_neigh_y


if __name__ == '__main__':
    img1 = cv2.imread('part_of_datasets/hpatches-sequences-release/v_graffiti/1.jpg')
    img2 = cv2.imread('part_of_datasets/hpatches-sequences-release/v_graffiti/3.jpg')
    H = np.mat(read_H('part_of_datasets/hpatches-sequences-release/v_graffiti/H_1_3'))
    m1, m2 = SIFT(img1, img2, 1000)
    # x = m1, y = m2, putative matching set S={(m1, m2)}

    # Remove duplicate matching point pairs detected by SIFT to avoid the situation that
    # multiple neighborhood points are the same during neighborhood construction ( one-to-many situation still occur).
    temp = np.hstack([m1, m2])
    temp = np.unique(temp, axis=0)
    m1 = temp[:, 0:2]
    m2 = temp[:, 2:4]

    time_start = time.time()
    # best_index, best_four_neigh_x, best_four_neigh_y = LMC_LPM(m1, m2, 8, 26)
    best_index, best_four_neigh_x, best_four_neigh_y = LMC_RANSAC(m1, m2, 8, 8, 3.4)
    time_end = time.time()
    time_c = time_end - time_start
    print("Runtime of LMC = ", time_c, "s")

    if len(best_index) == 0:
        exit(0)

    match1 = m1[best_index, :]
    match2 = m2[best_index, :]
    index_X = best_four_neigh_x[best_index, :]
    index_Y = best_four_neigh_y[best_index, :]
    F_s = F_Score.calculation(m1, m2, match1, match2, H, img1)

    m2[:, 0] = m2[:, 0] + img1.shape[1]
    match1 = m1[best_index, :]
    match2 = m2[best_index, :]

    b, g, r = cv2.split(img1)
    img1 = cv2.merge([r, g, b])
    b, g, r = cv2.split(img2)
    img2 = cv2.merge([r, g, b])

    imgs = np.hstack([img1, img2])

    plt.figure(1)
    plt.imshow(imgs)
    plt.plot(match1[:, 0], match1[:, 1], 'ro', markersize=0.55)
    plt.plot(match2[:, 0], match2[:, 1], 'ro', markersize=0.55)
    plt.plot([match1[:, 0], match2[:, 0]], [match1[:, 1], match2[:, 1]], 'b', linewidth=1)
    N = best_index.shape[0]
    L = index_X.shape[1]
    for i in range(N):
        for j in range(L):
            plt.plot([m1[index_X[i, j], 0], match1[i, 0]], [m1[index_X[i, j], 1], match1[i, 1]], 'y', linewidth=0.5)
            plt.plot([m2[index_Y[i, j], 0], match2[i, 0]], [m2[index_Y[i, j], 1], match2[i, 1]], 'y', linewidth=0.5)

    plt.show()

