import Noisy_data
import matplotlib.pyplot as plt
import numpy as np
import functions
from numpy.linalg import norm as norm

def abs_err(real, fil):
    total = 0
    n = len(real)
    for i in range(n):
       total += norm(real[i] - fil[i])
    return total/n
   

def rel_error(real, fil,abs_err):
    """return the average relative error of entire data
    """
    total = 0
    n = len(real)
    #sum_f = 0
    sum_a = 0
    for i in range(n):
       #sum_f += abs(fil[i])
       sum_a += abs(real[i])
    return norm(abs_err*n/sum_a)
#get data

y1 = Noisy_data.y1
y2 = Noisy_data.y2
y3 = Noisy_data.y3
time1 = Noisy_data.time1
time2 = Noisy_data.time2
time3 = Noisy_data.time3

width = 3
#fre1=128
n1 = 128
peak11 = 10    # ideal value is approximately N/T1
peak12 = 20
peak13 = 30
peak1 = [peak11, peak12, peak13]
#freq2=1024
n2 = 1024
peak21 = 100    # ideal value is approximately N/T1
peak22 = 200
peak23 = 300
peak2 = [peak21,peak22,peak23]
#freq3 = 8192
n3 = 8192
peak31 = 1000    # ideal value is approximately N/T1
peak32 = 2000
peak33 = 3000
peak3 = [peak31,peak32,peak33]

def get_data(n, y, peak):
    """Return filter data from three method
    """
    # change original data to the frequency basis
    z = np.fft.fft(y)
    z_dft = functions.dft(y)
    z_fft = functions.cooley_tukey(y)
    # freq = n
    freq = np.arange(n)
    # define the Gaussian filtration function
    filter_function1 = (np.exp(-(freq-peak[0])**2/width)+np.exp(-(freq+peak[0]-n)**2/width))
    filter_function2 = (np.exp(-(freq-peak[1])**2/width)+np.exp(-(freq+peak[1]-n)**2/width))
    filter_function3 = (np.exp(-(freq-peak[2])**2/width)+np.exp(-(freq+peak[2]-n)**2/width))
    filter_function = filter_function1+filter_function2+filter_function3
    z_filtered = z*filter_function
    z_dft_filtered = z_dft*filter_function
    z_fft_filtered = z_fft*filter_function
    #get filtered data
    y_filtered_1 = np.fft.ifft(z_filtered)#buildin_fft
    y_filtered_2 = np.fft.ifft(z_dft_filtered) #dft
    y_filtered_3 = np.fft.ifft(z_fft_filtered) #fft
    return y_filtered_1, y_filtered_2, y_filtered_3

actual1 = Noisy_data.y11+Noisy_data.y12+Noisy_data.y13 #n=128
actual2 = Noisy_data.y21+Noisy_data.y22+Noisy_data.y23 #n=1024
actual3 = Noisy_data.y31+Noisy_data.y32+Noisy_data.y33 #n=8192

def error_solver(n,y, peak, actual):
    output = get_data(n,y,peak)
    built_in = output[0]
    dft = output[1]
    fft = output[2]
    #abs_err
    err1= abs_err(actual, built_in)
    err2= abs_err(actual, dft)
    err3= abs_err(actual, fft)

    #rel_err
    rel_err1 = rel_error(actual, built_in, err1) #built_in fft
    rel_err2 = rel_error(actual, dft, err2) #dft
    rel_err3 = rel_error(actual, fft, err3) #fft
    return rel_err1, rel_err2, rel_err3, err1, err2, err3

#error dealer
ns = [n1, n2, n3]
data1 = error_solver(n1,y1,peak1,actual1)#n1
data2 = error_solver(n2,y2,peak2,actual2)#n2
data3 = error_solver(n3,y3,peak3,actual3)#n3
#print err table
line = "-"*122
space1 = " "*21
space = " "
print("{}error anaylsis{}".format(space*54, space*54))
print(line)
print("  n ||{}absolute error{}||{}relative error{}||".format(space1,space1,space1, space1))
print("    ||   builtin_fft    |{}dft{}|{}fft{}||   builtin_fft    |{}dft{}|{}fft{}||".format(space*7,
                                                                                              space*8,
                                                                                              space*7,
                                                                                              space*8,
                                                                                              space*7,
                                                                                              space*8,
                                                                                              space*7,
                                                                                              space*8))
print(line)
print("{:4d}||{:16.16f}|{:16.16f}|{:16.16f}||{:16.16f}|{:16.16f}|{:16.16f}||".format(ns[0],data1[3],
                                                                                     data1[4],
                                                                                     data1[5],
                                                                                     data1[0],
                                                                                     data1[1],
                                                                                     data1[2]))
print(line)
print("{:4d}||{:16.16f}|{:16.16f}|{:16.16f}||{:16.16f}|{:16.16f}|{:16.16f}||".format(ns[1],data2[3],
                                                                                     data2[4],
                                                                                     data2[5],
                                                                                     data2[0],
                                                                                     data2[1],
                                                                                     data2[2]))
print(line)
print("{:4d}||{:16.16f}|{:16.16f}|{:16.16f}||{:16.16f}|{:16.16f}|{:16.16f}||".format(ns[2],data3[3],
                                                                                     data3[4],
                                                                                     data3[5],
                                                                                     data3[0],
                                                                                     data3[1],
                                                                                     data3[2]))
