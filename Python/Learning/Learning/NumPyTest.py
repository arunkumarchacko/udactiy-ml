#import numpy as np

#print "NumpyTest"

#a = np.arange(15).reshape(3, 5)
#print a
#print a.shape
#print a.ndim

#print np.array([4,2,35,23])
#print np.ones((2,3,4), dtype=np.int32)

#import numpy as np
#import matplotlib.pyplot as plt
## Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
#mu, sigma = 2, 5
#v = np.random.normal(mu,sigma,10000)
## Plot a normalized histogram with 50 bins
#plt.hist(v, bins=50, normed=1)       # matplotlib version (plot)
#plt.show()

#(n, bins) = np.histogram(v, bins=50, normed=True)  # NumPy version (no plot)
#plt.plot(.5*(bins[1:]+bins[:-1]), n)
#plt.show()


#n = [1,2,3,4,5,6,7]

#print np.mean(n)
#print np.std(n)
#print np.median(n)


import numpy as np

'''
The following code is to help you play with Numpy, which is a library 
that provides functions that are especially useful when you have to
work with large arrays and matrices of numeric data, like doing 
matrix matrix multiplications. Also, Numpy is battle tested and 
optimized so that it runs fast, much faster than if you were working
with Python lists directly.
'''

'''
The array object class is the foundation of Numpy, and Numpy arrays are like
lists in Python, except that every thing inside an array must be of the
same type, like int or float.
'''
# Change False to True to see Numpy arrays in action
if False:
    array = np.array([1, 4, 5, 8], float)
    print array
    print ""
    array = np.array([[1, 2, 3], [4, 5, 6]], float)  # a 2D array/Matrix
    print array

'''
You can index, slice, and manipulate a Numpy array much like you would with a
a Python list.
'''
# Change False to True to see array indexing and slicing in action
if False:
    array = np.array([1, 4, 5, 8], float)
    print array
    print ""
    print array[1]
    print ""
    print array[:2]
    print ""
    array[1] = 5.0
    print array[1]

# Change False to True to see Matrix indexing and slicing in action
if False:
    two_D_array = np.array([[1, 2, 3], [4, 5, 6]], float)
    print two_D_array
    print ""
    print two_D_array[1][1]
    print ""
    print two_D_array[1, :]
    print ""
    print two_D_array[:, 2]

'''
Here are some arithmetic operations that you can do with Numpy arrays
'''
# Change False to True to see Array arithmetics in action
if False:
    array_1 = np.array([1, 2, 3], float)
    array_2 = np.array([5, 2, 6], float)
    print array_1 + array_2
    print ""
    print array_1 - array_2
    print ""
    print array_1 * array_2

# Change False to True to see Matrix arithmetics in action
if False:
    array_1 = np.array([[1, 2], [3, 4]], float)
    array_2 = np.array([[5, 6], [7, 8]], float)
    print array_1 + array_2
    print ""
    print array_1 - array_2
    print ""
    print array_1 * array_2

'''
In addition to the standard arthimetic operations, Numpy also has a range of
other mathematical operations that you can apply to Numpy arrays, such as
mean and dot product.

Both of these functions will be useful in later programming quizzes.
'''
if True:
    array_1 = np.array([1, 2, 3], float)
    array_2 = np.array([[6], [7], [8]], float)
    print np.mean(array_1)
    print np.mean(array_2)
    print ""
    print np.dot(array_1, array_2)
