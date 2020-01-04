# pattern-recognition-SVD
Recognition of centered hand written digits using singular value decomposition

The program is ran by makefile in /src folder. 
The input files "digit0.mat", ... , "digit9.mat" should be in the same folder as the program files when running it.

The digits are first trained one by one using large sets of images of handwritten digits. The program then analyzes recognizing accuracy and time complexity of the computations in terms of how far the SVD computations are conducted. A curoff value is then chosen accordingly to avoid innecessary computations in recognition while still getting moderately accurate results. 

Recognition is then applied to a few self made digits and larger sets of other hand written digits to get data of the accuracy of the recognition. 

The program plots data from recognizing accuracy and time complexity, first few eigenstates from recognition of 0 and 1, the self made digits and in the end the data of the accuracy of recognition.

Running the program may take a couple of minutes.
