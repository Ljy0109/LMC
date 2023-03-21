import os
import time
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import LMC
import warnings
warnings.filterwarnings("ignore")


def read_files(path):
    t1 = -1
    file_mat = []
    for root, dirs, files in os.walk(path):
        print('root = ', root, '\n')
        print('dirs = ', dirs, '\n')
        print('files = ', files, '\n')
        t1 = t1 + 1
        t2 = -1
        for name in files:
            print('path = ', os.path.join(root, name))
            t2 = t2 + 1
            file_mat.append(os.path.join(root, name))

    return file_mat, t2 + 1


def calculate(file_mat, Gt, num):
    inliers_num = np.zeros(num)
    F_s_LMC_RANSAC = np.ones(num)
    F_s_LMC_LPM = np.ones(num)
    time_LMC_RANSAC = np.ones(num)
    time_LMC_LPM = np.ones(num)
    recall_LMC_RANSAC = np.ones(num)
    recall_LMC_LPM = np.ones(num)
    precision_LMC_RANSAC = np.ones(num)
    precision_LMC_LPM = np.ones(num)
    for i in range(num):
        data = io.loadmat(file_mat[i])
        gt = io.loadmat(Gt[i])
        mask = gt['GroundTruth']
        CorrectIndex = np.where(mask == 1)[0]
        X = data['point1']
        Y = data['point2']

        correct = set(CorrectIndex.tolist())
        print('length of correctIndex = ', len(correct), '\n')
        inliers_num[i] = len(correct)/len(X)

        time_start = time.time()
        best_index, best_four_neigh_x, best_four_neigh_y = LMC.LMC_RANSAC(X, Y, 8, 8, 3.4)
        time_end = time.time()
        match_correct = set(best_index.tolist())
        num = len(match_correct & correct)
        if len(match_correct) != 0 and num != 0:
            recall_LMC_RANSAC[i] = num / len(CorrectIndex)
            precision_LMC_RANSAC[i] = num / len(match_correct)
            time_LMC_RANSAC[i] = time_end - time_start
            F_s_LMC_RANSAC[i] = (2 * precision_LMC_RANSAC[i] * recall_LMC_RANSAC[i]) / (
                        precision_LMC_RANSAC[i] + recall_LMC_RANSAC[i])
        else:
            time_LMC_RANSAC[i] = np.max(time_LMC_RANSAC)
            F_s_LMC_RANSAC[i] = 0
            recall_LMC_RANSAC[i] = 0
            precision_LMC_RANSAC[i] = 0

        time_start = time.time()
        best_index, best_four_neigh_x, best_four_neigh_y = LMC.LMC_LPM(X, Y, 8, 26)
        time_end = time.time()
        match_correct = set(best_index.tolist())
        num = len(match_correct & correct)
        if len(match_correct) != 0 and num != 0:
            recall_LMC_LPM[i] = num / len(CorrectIndex)
            precision_LMC_LPM[i] = num / len(match_correct)
            time_LMC_LPM[i] = time_end - time_start
            F_s_LMC_LPM[i] = (2 * precision_LMC_LPM[i] * recall_LMC_LPM[i]) / (precision_LMC_LPM[i] + recall_LMC_LPM[i])
        else:
            time_LMC_LPM[i] = np.max(time_LMC_LPM)
            F_s_LMC_LPM[i] = 0
            recall_LMC_LPM[i] = 0
            precision_LMC_LPM[i] = 0

    return recall_LMC_RANSAC, recall_LMC_LPM, \
           precision_LMC_RANSAC, precision_LMC_LPM, \
           F_s_LMC_RANSAC, F_s_LMC_LPM, \
           time_LMC_RANSAC, time_LMC_LPM, inliers_num


if __name__ == '__main__':
    file_mat, num = read_files('part_of_datasets/SUIRD_v2.2/Mixture/points_mats')
    gt, _ = read_files('part_of_datasets/SUIRD_v2.2/Mixture/ground_truth_mats')
    recall_LMC_RANSAC, recall_LMC_LPM, \
    precision_LMC_RANSAC, precision_LMC_LPM, \
    F_s_LMC_RANSAC, F_s_LMC_LPM, \
    time_LMC_RANSAC, time_LMC_LPM, inliers_num\
        = calculate(file_mat, gt, num)

    r7 = np.sort(recall_LMC_RANSAC.ravel())
    r8 = np.sort(recall_LMC_LPM.ravel())

    p7 = np.sort(precision_LMC_RANSAC.ravel())
    p8 = np.sort(precision_LMC_LPM.ravel())

    f7 = np.sort(F_s_LMC_RANSAC.ravel())
    f8 = np.sort(F_s_LMC_LPM.ravel())

    t7 = np.sort(time_LMC_RANSAC.ravel()) * 1000
    t8 = np.sort(time_LMC_LPM.ravel()) * 1000

    x = np.arange(0, f7.shape[0], 1) / f7.shape[0]

    inliers = np.sort(inliers_num)

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