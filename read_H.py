# coding=utf-8
# read homography matrix in HPatches
import numpy as np


def read_H(filename):
    """
    :param filename: filename = 'hpatches-sequences-release/v_azzola/H_1_2'  # path
    :return: H
    """
    lines = ''
    H = np.array([])
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            else:
                this_lines = lines.split()
                for this_line in this_lines:
                    H = np.append(H, np.float32(this_line))

    return H.reshape(3, 3)


if __name__ == '__main__':
    H = read_H('hpatches-sequences-release/v_azzola/H_1_2')
    print(H)
