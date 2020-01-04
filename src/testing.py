import numpy as np
from initialize import ainit
from plotting import statplot


# Tester for image recognition. If printstatistics = True, prints how many of the digits in digitarr
# are correctly recognized to be digit of correctdigit

def tester(digitarr, correctdigit, printstatistics=True):
    correct = 0  # Storing amount of correctly recognized digits

    # Loop through result digits of recognition
    for i in range(len(digitarr)):
        if digitarr[i] == correctdigit:
            correct += 1

    # Printing statistics if that is wanted
    if printstatistics:
        print("")
        print(correct, "/", len(digitarr), "of digits were correctly recognized as ", correctdigit)
    return correct


# Using SVD to recognize digits in testimgs. If printresults = True the function
# prints result of recognition for every image in testimgs

def recognize(u, k, testimgs, printresults=True):
    imgnum = len(testimgs[:, 0])
    results = np.zeros(imgnum)  # Storing result digits of recognition

    # Loop through all images to be tested
    for img in range(imgnum):

        minres = 1000  # Storing smallest residual given for current img
        digitnow = 0  # Storing best guess for digit in current img

        # Loop to test every image against U matrix of SVD for every possible digit
        for digit in range(len(u)):

            # Computing the residual for current img against U-matrix of current digit
            ucal = u[digit][:, :k]
            mtx = np.identity(784) - np.matmul(ucal, np.transpose(ucal))
            res = np.sum(np.matmul(mtx, testimgs[img]) ** 2)

            # Changing best guess if results are better than against formerly tested digits
            if res < minres:
                minres = res
                digitnow = digit
        if printresults:
            print("")
            print("Digit number ", img + 1, " of tested images is "
                                            "recognized as: ", digitnow)
        results[img] = digitnow

    return results


# Function for running test in tester function for every digit and then plotting statistics

def runtests(u, k, fnames, plotstatistics=True):
    percentages = np.zeros(len(u))  # Storing % of correctly recognized digits for every image set

    # Loop through all digits
    for digit in range(len(u)):
        imgs = np.transpose(ainit(fnames[digit]))[:50, :]  # Initializing test images for current digit

        # Recognizing the digits in test mages
        results = recognize(u, k, imgs, printresults=False)

        # Saving amount of correct results
        percentages[digit] = tester(results, digit) * 2

    if plotstatistics:
        statplot(percentages)  # Plotting the statistics
    return percentages
