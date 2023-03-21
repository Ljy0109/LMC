"""
A program for calculating F-score, Recall, Precision
"""
import numpy as np
import cv2


def calculation(m1, m2, match1, match2, H, img):
    h, w, _ = img.shape
    eip = 5  # allowable error

    m1_re = np.float32(m1).reshape(-1, 1, 2)
    m1_H = cv2.perspectiveTransform(m1_re, H).reshape(m1_re.shape[0], 2)
    match1_re = np.float32(match1).reshape(-1, 1, 2)
    match1_H = cv2.perspectiveTransform(match1_re, H).reshape(match1_re.shape[0], 2)

    # Calculate the number of inliers in the putative matching set
    ti = -1
    index_m1_H = np.array(np.zeros(m1_H.shape[0]), int)
    for i in range(m1.shape[0]):
        x1 = m1_H[i, 0]
        y1 = m1_H[i, 1]
        x2 = m2[i, 0]
        y2 = m2[i, 1]
        dist = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)
        if dist > eip:
            ti = ti + 1
            index_m1_H[ti] = i
    if ti != -1:
        index_m1_H = index_m1_H[0:ti]
        repeat_point = np.delete(m1_H, index_m1_H, axis=0)
    else:
        repeat_point = m1_H
    print('the number of inliers in the putative matching set = ', repeat_point.shape[0])

    # Calculate the number of inliers in the result of LMC method
    ti = -1
    index_match1H = np.array(np.zeros(match1_H.shape[0]), int)
    for i in range(match1_H.shape[0]):
        x1 = match1_H[i, 0]
        y1 = match1_H[i, 1]
        x2 = match2[i, 0]
        y2 = match2[i, 1]
        dist = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)
        if dist > eip:
            ti = ti + 1
            index_match1H[ti] = i
    if ti != -1:
        index_match1H = index_match1H[0:ti]
        repeat_point_match = np.delete(match1_H, index_match1H, axis=0)  # 以m2形式表示重复点
    else:
        repeat_point_match = match1_H
    print('the number of inliers in the result of LMC method = ', repeat_point_match.shape[0])
    print('matches in the result of LMC method =  = ', match2.shape[0])
    false_matches = match2.shape[0] - repeat_point_match.shape[0]

    if repeat_point_match.shape[0] != 0 and repeat_point.shape[0] != 0:
        Recall = repeat_point_match.shape[0] / repeat_point.shape[0]
        Precision = repeat_point_match.shape[0] / (repeat_point_match.shape[0] + false_matches)
        F_s = (2 * Precision * Recall) / (Precision + Recall)
    else:
        Recall = 0
        Precision = 0
        F_s = 0

    print('Recall = ', Recall)
    print('Precision = ', Precision)
    print('F_score = ', F_s)
    return Recall, Precision, F_s





