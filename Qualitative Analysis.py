import cv2
import matplotlib.pyplot as plt
import numpy as np
from read_H import read_H
import scipy.io as io
import random
import LMC


def byMat(path):
    data = io.loadmat(path)
    CorrectIndex = data['CorrectIndex'] - 1  # 原文件索引是matlab的，需要减一
    X = data['X']  # kp1/X
    Y = data['Y']
    if CorrectIndex.shape[0] == 1:
        CorrectIndex = CorrectIndex.T
    correct = set((CorrectIndex.reshape(CorrectIndex.shape[0])).tolist())

    index_std = np.arange(0, X.shape[0], 1, int)
    tf = np.delete(index_std, CorrectIndex)

    best_index, best_four_neigh_x, best_four_neigh_y = LMC.LMC_LPM(X, Y, 8, 26)
    tf1 = np.delete(index_std, best_index)

    pic = random.sample(range(0, index_std.shape[0]), 100)
    random_index = index_std[pic]
    tp = list((set(random_index.tolist()) & correct) & set(best_index.tolist()))
    tn = list(set(random_index.tolist()) & set(tf.tolist()) & set(tf1.tolist()))
    fn = list(set(random_index.tolist()) & set(tf.tolist()) & set(best_index.tolist()))
    fp = list((set(random_index.tolist()) & correct) & set(tf1.tolist()))
    TP1 = X[tp, :]
    TP2 = Y[tp, :]
    TN1 = X[tn, :]
    TN2 = Y[tn, :]
    FN1 = X[fn, :]
    FN2 = Y[fn, :]
    FP1 = X[fp, :]
    FP2 = Y[fp, :]
    nx = X[best_four_neigh_x[tp], :]
    ny = Y[best_four_neigh_y[tp], :]
    fx = X[best_four_neigh_x[fn], :]
    fy = Y[best_four_neigh_y[fn], :]

    return TP1, TP2, TN1, TN2, FN1, FN2, FP1, FP2, nx, ny, fx, fy


def byH(img1, img2, H):
    X, Y = LMC.SIFT(img1, img2, 1000)
    temp = np.hstack([X, Y])
    temp = np.unique(temp, axis=0)
    X = temp[:, 0:2]
    Y = temp[:, 2:4]
    m1_re = np.float32(X).reshape(-1, 1, 2)
    trans_x = cv2.perspectiveTransform(m1_re, H).reshape(X.shape)
    CorrectIndex = np.array([], int)
    for i in range(X.shape[0]):
        temp_dist = np.sqrt((trans_x[i, 0] - Y[i, 0]) ** 2 + (trans_x[i, 1] - Y[i, 1]) ** 2)
        if temp_dist <= 5:
            CorrectIndex = np.append(CorrectIndex, i)
    correct = set(CorrectIndex.tolist())
    index_std = np.arange(0, X.shape[0], 1, int)
    tf = np.delete(index_std, CorrectIndex)

    best_index, best_four_neigh_x, best_four_neigh_y = LMC.LMC_RANSAC(X, Y, 8, 8, 3.4)
    tf1 = np.delete(index_std, best_index)

    pic = random.sample(range(0, index_std.shape[0]), 100)
    random_index = index_std[pic]
    tp = list((set(random_index.tolist()) & correct) & set(best_index.tolist()))
    tn = list(set(random_index.tolist()) & set(tf.tolist()) & set(tf1.tolist()))
    fn = list(set(random_index.tolist()) & set(tf.tolist()) & set(best_index.tolist()))
    fp = list((set(random_index.tolist()) & correct) & set(tf1.tolist()))
    TP1 = X[tp, :]
    TP2 = Y[tp, :]
    TN1 = X[tn, :]
    TN2 = Y[tn, :]
    FN1 = X[fn, :]
    FN2 = Y[fn, :]
    FP1 = X[fp, :]
    FP2 = Y[fp, :]
    nx = X[best_four_neigh_x[tp], :]
    ny = Y[best_four_neigh_y[tp], :]
    fx = X[best_four_neigh_x[fn], :]
    fy = Y[best_four_neigh_y[fn], :]

    return TP1, TP2, TN1, TN2, FN1, FN2, FP1, FP2, nx, ny, fx, fy


def bySUIRD(X, Y, path5):
    gt = io.loadmat(path5)
    mask = gt['GroundTruth']
    CorrectIndex = np.where(mask == 1)[0]
    if CorrectIndex.shape[0] == 1:
        CorrectIndex = CorrectIndex.T
    correct = set((CorrectIndex.reshape(CorrectIndex.shape[0])).tolist())
    index_std = np.arange(0, X.shape[0], 1, int).astype(int)
    tf = np.delete(index_std, CorrectIndex)

    best_index, best_four_neigh_x, best_four_neigh_y = LMC.LMC_RANSAC(X, Y, 8, 8, 3.4)
    tf1 = np.delete(index_std, best_index)

    pic = random.sample(range(0, index_std.shape[0]), 100)
    random_index = index_std[pic]
    tp = list((set(random_index.tolist()) & correct) & set(best_index.tolist()))
    tn = list(set(random_index.tolist()) & set(tf.tolist()) & set(tf1.tolist()))
    fn = list(set(random_index.tolist()) & set(tf.tolist()) & set(best_index.tolist()))
    fp = list((set(random_index.tolist()) & correct) & set(tf1.tolist()))
    TP1 = X[tp, :]
    TP2 = Y[tp, :]
    TN1 = X[tn, :]
    TN2 = Y[tn, :]
    FN1 = X[fn, :]
    FN2 = Y[fn, :]
    FP1 = X[fp, :]
    FP2 = Y[fp, :]
    nx = X[best_four_neigh_x[tp], :]
    ny = Y[best_four_neigh_y[tp], :]
    fx = X[best_four_neigh_x[fn], :]
    fy = Y[best_four_neigh_y[fn], :]

    return TP1, TP2, TN1, TN2, FN1, FN2, FP1, FP2, nx, ny, fx, fy


if __name__ == '__main__':
    # options = 1 : test for DTU, Retina and RS
    # options = 2 : test for HPatches
    # options = 3 : test for SUIRD
    options = 3
    if options == 1:
        name = '1016'
        where = 'DTU'
        path1 = 'part_of_datasets/DTU/DTUCorrectIndex/house1016.mat'
        path2 = 'part_of_datasets/DTU/DTUdata/scan6/rect_010_max.png'
        path3 = 'part_of_datasets/DTU/DTUdata/scan6/rect_016_max.png'
        TP1, TP2, TN1, TN2, FN1, FN2, FP1, FP2, nx, ny, fx, fy = byMat(path1)
        img1 = cv2.imread(path2)
        img2 = cv2.imread(path3)
        b, g, r = cv2.split(img1)
        img1 = cv2.merge([r, g, b])
        b, g, r = cv2.split(img2)
        img2 = cv2.merge([r, g, b])
    elif options == 2:
        name = 'graffiti2'
        where = 'hp'
        path2 = 'part_of_datasets/hpatches-sequences-release/v_graffiti/1.jpg'
        path3 = 'part_of_datasets/hpatches-sequences-release/v_graffiti/2.jpg'
        path4 = 'part_of_datasets/hpatches-sequences-release/v_graffiti/H_1_2'
        img1 = cv2.imread(path2)
        img2 = cv2.imread(path3)
        H = read_H(path4)
        TP1, TP2, TN1, TN2, FN1, FN2, FP1, FP2, nx, ny, fx, fy = byH(img1, img2, H)
        b, g, r = cv2.split(img1)
        img1 = cv2.merge([r, g, b])
        b, g, r = cv2.split(img2)
        img2 = cv2.merge([r, g, b])
    elif options == 3:
        name = '46'
        where = 'Mixture'
        path1 = 'part_of_datasets/SUIRD_v2.2/' + where + '\points_mats/' + name + '.mat'
        path2 = 'part_of_datasets/SUIRD_v2.2/' + where + '\imgs/' + name + '-l.jpg'
        path3 = 'part_of_datasets/SUIRD_v2.2/' + where + '\imgs//' + name + '-r.jpg'
        path5 = 'part_of_datasets/SUIRD_v2.2/' + where + '\ground_truth_mats//' + name + '-Completed.mat'
        data = io.loadmat(path1)
        X = data['point1']
        Y = data['point2']
        img1 = data['image1']
        img2 = data['image2']
        TP1, TP2, TN1, TN2, FN1, FN2, FP1, FP2, nx, ny, fx, fy = bySUIRD(X, Y, path5)

    # 100 matches are randomly selected for display
    plt.figure(1)
    plt.gca().invert_yaxis()
    plt.quiver(TN1[:, 0], TN1[:, 1], TN2[:, 0] - TN1[:, 0], TN2[:, 1] - TN1[:, 1], color='black', angles='xy',
               scale_units='xy', scale=1, units='xy')
    plt.quiver(TP1[:, 0], TP1[:, 1], TP2[:, 0] - TP1[:, 0], TP2[:, 1] - TP1[:, 1], color='b', angles='xy',
               scale_units='xy', scale=1, units='xy')
    plt.quiver(FN1[:, 0], FN1[:, 1], FN2[:, 0] - FN1[:, 0], FN2[:, 1] - FN1[:, 1], color='r', angles='xy',
               scale_units='xy', scale=1, units='xy')
    plt.quiver(FP1[:, 0], FP1[:, 1], FP2[:, 0] - FP1[:, 0], FP2[:, 1] - FP1[:, 1], color='g', angles='xy',
               scale_units='xy', scale=1, units='xy')
    plt.xlim([0, img1.shape[1]])
    plt.ylim([img1.shape[0], 0])

    plt.figure(2)
    plt.gca().axis('off')
    temp = np.zeros((img1.shape[0], 20, 3)).astype(int) + 255
    imgs = np.hstack([np.hstack([img1, temp]), img2])
    plt.imshow(imgs)
    TP2[:, 0] = TP2[:, 0] + img1.shape[1] + temp.shape[1]
    FN2[:, 0] = FN2[:, 0] + img1.shape[1] + temp.shape[1]
    FP2[:, 0] = FP2[:, 0] + img1.shape[1] + temp.shape[1]
    ny[:, :, 0] = ny[:, :, 0] + img1.shape[1] + temp.shape[1]
    fy[:, :, 0] = fy[:, :, 0] + img1.shape[1] + temp.shape[1]
    plt.plot([TP1[:, 0], TP2[:, 0]], [TP1[:, 1], TP2[:, 1]], 'b', linewidth=1)
    plt.plot([FN1[:, 0], FN2[:, 0]], [FN1[:, 1], FN2[:, 1]], 'r', linewidth=1)
    plt.plot([FP1[:, 0], FP2[:, 0]], [FP1[:, 1], FP2[:, 1]], 'g', linewidth=1)
    for i in range(TP1.shape[0]):
        for j in range(4):
            plt.plot([nx[i, j, 0], TP1[i, 0]], [nx[i, j, 1], TP1[i, 1]], 'y', linewidth=1)
            plt.plot([ny[i, j, 0], TP2[i, 0]], [ny[i, j, 1], TP2[i, 1]], 'y', linewidth=1)
    for i in range(FN1.shape[0]):
        for j in range(4):
            plt.plot([fx[i, j, 0], FN1[i, 0]], [fx[i, j, 1], FN1[i, 1]], 'purple', linewidth=1)
            plt.plot([fy[i, j, 0], FN2[i, 0]], [fy[i, j, 1], FN2[i, 1]], 'purple', linewidth=1)

    plt.show()
