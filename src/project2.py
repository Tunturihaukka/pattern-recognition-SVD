from initialize import fnames, testimgs
from training import train
from testing import recognize, runtests
from plotting import plot

# Training all digits 0-9

a = train(fnames())

# First, recognizing 2 handmade images of digit 0

recognize(a[0], a[1], testimgs()[0:2])

# Then same for 2 handmade images of digit 1

recognize(a[0], a[1], testimgs()[2:4])

# Using 50 images of all digits 0-9 to get statistics of how robust this method is
# for pattern recognition. Plotting statistcs of the result

runtests(a[0], a[1], fnames())

# Plotting tested images and a few eigenpatterns from training of digits of choice, 0 and 1

plot(a[0][0], a[0][1], testimgs())
