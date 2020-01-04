from scipy.io import loadmat as lm
import numpy as np


# Initializer for matrix A by converting .mat data to numpy array

def ainit(mdata):
    dictr = {}
    mat0 = lm(mdata, mdict=dictr)
    ain = np.array(dictr['D'])

    # Returning transpose so that the matrix is correctly interpreted as A = (a(1), ... , a(n))
    # where every column vector a(i) has length 28x28=784 and thus represents one
    # training image

    ain = ain.transpose()

    # With digits only the shape is to be considered and so transforming arrays to binary form

    ain = ain > 0

    return ain


# Initializing filename array used by training function

def fnames():
    return ["digit0.mat", "digit1.mat", "digit2.mat", "digit3.mat", "digit4.mat",
            "digit5.mat", "digit6.mat", "digit7.mat", "digit8.mat", "digit9.mat"]


# Function to transfer data of handmade test images to arrays

def testimgs():
    imgs = [np.zeros(784) for i in range(4)]
    names = ['zero_1', 'zero_2', 'one_1', 'one_2']

    # Arrays in text files are made to contain only digits 0-9 as entries and therefore
    # the data can be read character by character

    for j in range(4):
        k = 0
        with open(names[j]) as f:
            for line in f:
                for ch in line:
                    if ch != "\n":
                        imgs[j][k] = int(ch)
                        k += 1

        # Changing also test images to binary form for better shape recognition

        imgs[j] = imgs[j] > 0
    return np.array(imgs)