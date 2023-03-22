import os
import cv2
import time
import F_Score
import numpy as np
from read_H import read_H
import matplotlib.pyplot as plt
import LMC


def read_files(path):
    t1 = -1
    file_mat = []
    for root, dirs, files in os.walk(path):
        if len(dirs) != 0:
            print('root = ', root, '\n')
            print('dirs = ', dirs, '\n')
            print('files = ', files, '\n')
            img1 = []
            img2 = []
            H = []
            continue
        t1 = t1 + 1
        t2 = -1
        img1_temp = []
        img2_temp = []
        H_temp = []
        for name in files:
            path = os.path.join(root, name)
            if os.path.splitext(name)[-1] == '.ppm' or os.path.splitext(name)[-1] == '.txt':
                continue
            print('path = ', os.path.join(root, name))
            t2 = t2 + 1
            if t2 == 0:
                img1_temp.append(path)
            elif t2 > 0 and t2 < 6:
                img2_temp.append(path)
            elif t2 >5:
                H_temp.append(path)
        img1.append(img1_temp)
        img2.append(img2_temp)
        H.append(H_temp)
    return img1, img2, H


def calculate(image1, image2, h):
    L = len(image1)
    inliers_num = np.ones((L, 5))
    F_s_LMC_RANSAC = np.ones((L, 5))
    F_s_LMC_LPM = np.ones((L, 5))
    time_LMC_RANSAC = np.ones((L, 5))
    time_LMC_LPM = np.ones((L, 5))
    recall_LMC_RANSAC = np.ones((L, 5))
    recall_LMC_LPM = np.ones((L, 5))
    precision_LMC_RANSAC = np.ones((L, 5))
    precision_LMC_LPM = np.ones((L, 5))
    for i in range(L):
        for j in range(5):
            img1 = cv2.imread(image1[i][0])
            img2 = cv2.imread(image2[i][j])
            H = np.mat(read_H(h[i][j]))
            X, Y = LMC.SIFT(img1, img2, 500)

            X_re = np.float32(X).reshape(-1, 1, 2)
            trans_x = cv2.perspectiveTransform(X_re, H).reshape(-1, 2)
            inliers = []
            for n in range(X.shape[0]):
                temp_dist = np.sqrt((trans_x[n, 0] - Y[n, 0]) ** 2 + (trans_x[n, 1] - Y[n, 1]) ** 2)
                if temp_dist < 5:
                    inliers.append(n)

            inliers_num[i, j] = len(inliers) / X.shape[0]

            # LMC_RANSAC
            time_start = time.time()
            best_index, best_four_neigh_x, best_four_neigh_y = LMC.LMC_RANSAC(X, Y, 8, 8, 3.4)
            time_end = time.time()
            if len(best_index) != 0:
                match1 = X[best_index, :]
                match2 = Y[best_index, :]
                recall_LMC_RANSAC[i, j], precision_LMC_RANSAC[i, j], F_s_LMC_RANSAC[i, j] = F_Score.calculation(X, Y, match1, match2, H, img1)
                time_LMC_RANSAC[i, j] = time_end - time_start
            else:
                time_LMC_RANSAC[i, j] = np.max(time_LMC_RANSAC)
                F_s_LMC_RANSAC[i, j] = 0
                recall_LMC_RANSAC[i, j] = 0
                precision_LMC_RANSAC[i, j] = 0

            # LMC_LPM
            try:
                time_start = time.time()
                best_index, best_four_neigh_x, best_four_neigh_y = LMC.LMC_LPM(X, Y, 8, 26)
                time_end = time.time()
                match1 = X[best_index, :]
                match2 = Y[best_index, :]
                recall_LMC_LPM[i, j], precision_LMC_LPM[i, j], F_s_LMC_LPM[i, j] = F_Score.calculation(X, Y, match1, match2, H, img1)
                time_LMC_LPM[i, j] = time_end - time_start
            except:
                time_LMC_LPM[i, j] = np.max(time_LMC_LPM)
                F_s_LMC_LPM[i, j] = 0
                recall_LMC_LPM[i, j] = 0
                precision_LMC_LPM[i, j] = 0

    return recall_LMC_RANSAC, recall_LMC_LPM,\
           precision_LMC_RANSAC, precision_LMC_LPM,\
           F_s_LMC_RANSAC, F_s_LMC_LPM, \
           time_LMC_RANSAC, time_LMC_LPM, inliers_num


if __name__ == '__main__':
    img1, img2, H = read_files('part_of_datasets/hpatches-sequences-release')
    recall_LMC_RANSAC, recall_LMC_LPM, \
    precision_LMC_RANSAC, precision_LMC_LPM, \
    F_s_LMC_RANSAC, F_s_LMC_LPM, \
    time_LMC_RANSAC, time_LMC_LPM, inliers_num\
        = calculate(img1, img2, H)

    r7 = np.sort(recall_LMC_RANSAC.ravel())
    r8 = np.sort(recall_LMC_LPM.ravel())

    p7 = np.sort(precision_LMC_RANSAC.ravel())
    p8 = np.sort(precision_LMC_LPM.ravel())

    f7 = np.sort(F_s_LMC_RANSAC.ravel())
    f8 = np.sort(F_s_LMC_LPM.ravel())

    t7 = np.sort(time_LMC_RANSAC.ravel()) * 1000
    t8 = np.sort(time_LMC_LPM.ravel()) * 1000

    x = np.arange(0, f7.shape[0], 1) / f7.shape[0]

    inliers = np.sort(inliers_num.ravel())

    plt.figure(1)

    plt.plot(x, f7, label='LMC_RANSAC,AF = ' + '{:.2%}'.format(np.mean(f7)), color='b', marker='o', markersize=5)
    plt.plot(x, f8, label='LMC_LPM,AF = ' + '{:.2%}'.format(np.mean(f8)), color='red', marker='*', markersize=5)
    plt.legend(loc='upper right')

    plt.figure(2)

    plt.plot(x, r7, label='LMC_RANSAC,AR = ' + '{:.2%}'.format(np.mean(r7)), color='b', marker='o', markersize=5)
    plt.plot(x, r8, label='LMC_LPM,AR = ' + '{:.2%}'.format(np.mean(r8)), color='red', marker='*', markersize=5)
    plt.legend(loc='upper right')

    plt.figure(3)

    plt.plot(x, p7, label='LMC_RANSAC,AP = ' + '{:.2%}'.format(np.mean(p7)), color='b', marker='o', markersize=5)
    plt.plot(x, p8, label='LMC_LPM,AP = ' + '{:.2%}'.format(np.mean(p8)), color='red', marker='*', markersize=5)
    plt.legend(loc='upper right')

    plt.figure(4)
    plt.yscale('log')
    plt.plot(x, t7, label='LMC_RANSAC,ART = ' + '{:.2e}'.format(np.mean(t7)), color='b', marker='o', markersize=5)
    plt.plot(x, t8, label='LMC_LPM,ART = ' + '{:.2e}'.format(np.mean(t8)), color='red', marker='*', markersize=5)
    plt.legend(loc='upper right')

    plt.figure(5)
    plt.plot(x, inliers, label='Average Inlier Ratio = ' + '{:.2%}'.format(np.mean(inliers)), color='r', marker='o', markersize=5)
    plt.legend(loc='upper right')

    plt.show()
