import numpy as np
import random

pi = np.pi
#0, 128 ,w = pi*x/64
N1 = 128
time1 = np.arange(N1)

A11=4        # wave amplitude
x11 = 10     # frequency index
y11 = A11*np.sin(np.pi*x11/64*time1)


A12=7   
x12=20  
y12=A12*np.sin(np.pi*x12/64*time1)

A13=10
x13=30 
y13=A13*np.sin(np.pi*x13/64*time1)

y1 = y11+y12+y13

for x in range(1,128):
    if x != 10 and x != 20 and x!= 30:
        amplitude = random.random()
        y_temp = amplitude*np.sin(np.pi*x/64*time1)
        y1 += y_temp

#0,1024, w=pi*x/512
N2 =1024
time2 = np.arange(N2)

A21=4
x21 = 100
y21=A21*np.sin(np.pi*x21/512*time2)


A22=7   
x22=200  
y22=A22*np.sin(np.pi*x22/512*time2)

A23=10   
x23=300  
y23=A23*np.sin(np.pi*x23/512*time2)

y2 = y21+y22+y23

for x in range(1,1024):
    if x != 100 and x != 200 and x!= 300:
        amplitude = random.random()
        y_temp = amplitude*np.sin(np.pi*x/512*time2)
        y2 += y_temp

#0,8192, w= pi*x/4096
N3 =8192
time3 = np.arange(N3)

A31=4
x31 = 1000
y31=A31*np.sin(np.pi*x31/4096*time3)


A32=7
x32=2000
y32=A32*np.sin(np.pi*x32/4096*time3)

A33=10
x33=3000
y33=A33*np.sin(np.pi*x33/4096*time3)

y3 = y31+y32+y33

for x in range(1,8192):
    if x != 1000 and x != 2000 and x!= 3000:
        amplitude = random.random()
        y_temp = amplitude*np.sin(np.pi*x/4096*time3)
        y3 += y_temp