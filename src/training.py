from initialize import ainit
from plotting import residualplot, timeplot
from scipy.linalg import svd
import numpy as np
import math as m
import time


# Tester to find reasonable value for k-cutoff

def kcutoff(uarr, iniarr):
    maxk = len(uarr[0][0]) - 1
    digits = len(uarr)

    # For every digit and every k saving residual of first image of the respective U-array
    res = np.zeros([digits, len(uarr[0][0])])

    # Taking time of looping through k-values to analyze importance of cutoff value
    t = np.zeros(maxk)

    # Loop through SVD U-matrices for different digits
    for uval in range(digits):

        # Loop through possible values of k excluding k = 1
        for kval in range(maxk):
            start = time.time()  # Starting timing for current iteration

            # Calculating residual
            ucal = uarr[uval][:, :kval + 1]
            mtx = np.identity(784) - np.matmul(ucal, np.transpose(ucal))
            res[uval, kval] = np.sum(np.matmul(mtx, iniarr[uval][:, 0]) ** 2)

            end = time.time()  # Ending timing for current iteration
            t[kval] = end - start

    # Plotting residuals against k
    residualplot(res)
    timeplot(t)

    # Choosing k-cutoff to be at k for which correct residual of every digit is
    # 1/3 of the highest one

    cutoff = np.zeros(digits)
    for j in range(digits):
        for kval in range(maxk):
            if res[j, kval] < res[j, 0]/3:
                cutoff[j] = kval
                break

    kcut = m.floor(max(cutoff))
    print("")
    print("k-cutoff was chosen to: ", kcut)

    return kcut


# Tester to find best value for k

def kchoose(uarr, ini, kmax):

    # Calculating for every k the difference between correct digit residual and lowest incorrect digit
    # residual where correct digit is one that gives the highest result for the given k value

    # (Optimal value for k is then the one for which this difference is largest)

    diffs = np.zeros(kmax)  # Storing the differences

    # Loop through k-values before cutoff
    for kval in range(kmax):
        l = len(uarr)
        kdiffs = np.zeros(l)  # Storing differences of residuals for current k

        # Storing relevant residuals before calculating differences
        low = -1
        high = -1

        # Loop through digits for every digit
        for digit1 in range(l):

            for digit2 in range(l):

                # Computing residual of digit1 against training set of digit2
                ucal = uarr[digit1][:, :kval + 1]
                mtx = np.identity(784) - np.matmul(ucal, np.transpose(ucal))
                res = np.sum(np.matmul(mtx, ini[digit2][:, 0]) ** 2)

                # Storing as lower value of difference if residual is against digits own training set
                if digit2 == digit1:
                    low = res

                # Storing as higher value of difference if residual is at the moment lowest of
                # ones with training set of incorrect digit
                elif high == -1 or res < high:
                    high = res

            # Computing the difference of correct digit residual and lowest incorrect digit
            # residual for current k
            kdiffs[digit1] = high - low
            low, high, = -1, -1

        # Choosing the difference for current k
        diffs[kval] = np.max(kdiffs)

    # Choosing the k as one with largest of worst difference results
    bestk = np.argmax(diffs) + 1
    print("")
    print("Best value for k is ", bestk)

    return bestk


# Training for digits in data files given by trainimg

def train(trainimg):
    digits = 10

    # Using list for storing in beginning to allow arrays of arbitrary length

    ini = [0] * digits  # Storing training data for each digit in 1D-array form
    u = [0] * digits  # Storing the output U-arrays of singular value decomposition

    # Training of all digits 0-9
    for z in range(digits):
        print("Digit ", z, " in training")
        ini[z] = ainit(trainimg[z])
        u[z], s, v = svd(ini[z])
    uarr = np.array(u)

    kmax = kcutoff(uarr, ini)  # Finding reasonable cutoff value for k
    k = kchoose(uarr, ini, kmax)  # Finding the best k to be used in recognizing
    return u, k

