from matplotlib import pyplot as plt
import numpy as np


# Plots test images and first few eigenpatterns from training sets of digits 0 and 1

def plot(uarr0, uarr1, tarr):

    # Plotting the test images
    fig, axs = plt.subplots(2, 2)
    i = 0
    for ax in axs.flatten():
        ax.imshow(tarr[i].reshape((28, 28)))
        i += 1

    # Plotting the eigenpatterns of digits 0 and 1
    fig2, axs2 = plt.subplots(2, 2)
    fig3, axs3 = plt.subplots(2, 2)
    i, j = 0, 0
    for ax2 in axs2.flatten():
        ax2.imshow(uarr0[:, i].reshape((28, 28)))
        i += 1
    for ax3 in axs3.flatten():
        ax3.imshow(uarr1[:, j].reshape((28, 28)))
        j += 1

    plt.tight_layout()
    plt.show()


# Plots time used for execution of matrix calculation as a function of k

def timeplot(times):
    x = np.arange(0, len(times), 1)
    plt.plot(x, times, 'b-')
    plt.xlabel("Value of k")
    plt.ylabel("Time (seconds)")
    plt.title("Time used in calculation of k:th residual")
    plt.show()


# Plots residuals against k (Used for choosing k-cutoff)

def residualplot(res):

    # Loop through digits
    for i in range(len(res[:, 0])):
        x = np.arange(0, len(res[0]), 1)
        plt.plot(x, res[i], 'r-')
        plt.xlabel("Value of k")
        plt.ylabel("Residual")
        plt.title("Value of k:th residual")
    plt.show()


# Plots statistics of correctly recognized images

def statplot(percentages):
    x = np.arange(0, len(percentages), 1)
    plt.plot(x, percentages, 'ro')
    plt.axis([-1, 11, 0, 105])
    plt.title("Percentages of correctly recognized images")
    plt.xlabel("Digit")
    plt.ylabel("Amount of correct recognitions (%)")
    plt.grid()
    plt.show()
    return 1