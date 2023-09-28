#data.py is a python function where we create out data form these algorithm

# Get N from data.py

# Get wave from data.py

# define W:
import numpy as np

def dft(wave):
    """ This is discrete fourier transform. Wave is a finite sequence of
    equally-spaced samples of a function. This function convert wave into
    a complex-valued function of frequency
    Run time complexity is O(N^2).
    """
    N = len(wave)
    W = np.exp(-2j*np.pi/N)
    trans = []
    for i in range(N):
       temp = 0
       for k in range(N):
          temp += wave[k]*W**(i*k)
       trans.append(temp)
    return np.array(trans)

def cooley_tukey(wave):
    """This is one algorithm of FFT, which is called Cooley Turkey FFT algorithm.
    Run time complexity is O(NlogN) beacuase it split wave into even and odd, and
    deal the wave recursively.
    """
    N = len(wave)
    if N <= 1:
       return dft(wave)

    even = cooley_tukey(wave[::2])
    odd = cooley_tukey(wave[1::2])
    temp = []
    for i in range(N):
        factor = np.exp(-2j * np.pi * i / N)
        if i < N//2:
            temp.append(even[i] + factor * odd[i])
        else:
            temp.append(even[i-N//2] + factor*odd[i-N//2])
    return np.array(temp)