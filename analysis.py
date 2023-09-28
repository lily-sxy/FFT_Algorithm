import functions
import Noisy_data
import matplotlib.pyplot as plt
import numpy as np

y1 = Noisy_data.y1
y2 = Noisy_data.y2
y3 = Noisy_data.y3
time1 = Noisy_data.time1
time2 = Noisy_data.time2
time3 = Noisy_data.time3

width = 3
peak11 = 10
peak12 = 20
peak13 = 30

#128 data points
# show the original data with noise
plt.plot(time1,y1)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("figure 1: Noisy Data, 128 discrete freq")
plt.show()

# change original data to the frequency basis
z1 = np.fft.fft(y1)
z1_dft = functions.dft(y1)
z1_fft = functions.cooley_tukey(y1)

plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("figure 2: 3 FFT methods: frequency basis and corresponding amplitude")
plt.plot(np.abs(z1),label = "built in FFT")
plt.plot(np.abs(z1_dft),label = "DFT")
plt.plot(np.abs(z1_fft),label = "Cooley-Tukey FFt")
plt.legend()
plt.show()

# define the Gaussian filtration function
freq1 = np.arange(128)

filter_function11 = (np.exp(-(freq1-peak11)**2/width)+np.exp(-(freq1+peak11-128)**2/width))
filter_function12 = (np.exp(-(freq1-peak12)**2/width)+np.exp(-(freq1+peak12-128)**2/width))
filter_function13 = (np.exp(-(freq1-peak13)**2/width)+np.exp(-(freq1+peak13-128)**2/width))
filter_function1 = filter_function11+filter_function12+filter_function13
plt.plot(freq1,filter_function1)
plt.xlabel("Frequency")
plt.ylabel("Probability")
plt.title("figure 3: Gaussian (probability) filter function")
plt.show()

# get the filtered frequency basis
z1_filtered = z1*filter_function1
z1_dft_filtered = z1_dft*filter_function1
z1_fft_filtered = z1_fft*filter_function1
plt.plot(freq1,abs(z1_filtered),label = "filtered built in FFT")
plt.plot(freq1,abs(z1_dft_filtered), label = "filtered DFT")
plt.plot(freq1,abs(z1_fft_filtered), label = "filtered FFT")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("figure 4: Filtered Frequency Basis")
plt.legend()
plt.show()

# use inverse Fourier transform to get the filtered data
y1_filtered_1 = np.fft.ifft(z1_filtered)
y1_filtered_2 = np.fft.ifft(z1_dft_filtered)
y1_filtered_3 = np.fft.ifft(z1_fft_filtered)
plt.plot(time1,y1_filtered_1,label = "filtered built in FFT")
plt.plot(time1,y1_filtered_2, label = "filtered DFT")
plt.plot(time1,y1_filtered_3, label = "filtered FFT")

plt.plot(time1,Noisy_data.y11+Noisy_data.y12+Noisy_data.y13,label="actual")
plt.xlim(0,100)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("figure 5: Filtered Data")
plt.legend()
plt.show()



#1024 data points
plt.plot(time2,y2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("figure 6: Noisy Data, 1024 discrete freq")
plt.xlim(0,100)
plt.show()

z2 = np.fft.fft(y2)
z2_dft = functions.dft(y2)
z2_fft = functions.cooley_tukey(y2)

freq2 = np.arange(1024)
peak21 = 100
peak22 = 200
peak23 = 300

filter_function21 = (np.exp(-(freq2-peak21)**2/width)+np.exp(-(freq2+peak21-1024)**2/width))
filter_function22 = (np.exp(-(freq2-peak22)**2/width)+np.exp(-(freq2+peak22-1024)**2/width))
filter_function23 = (np.exp(-(freq2-peak23)**2/width)+np.exp(-(freq2+peak23-1024)**2/width))
filter_function2 = filter_function21+filter_function22+filter_function23
z2_filtered = z2*filter_function2
z2_dft_filtered = z2_dft*filter_function2
z2_fft_filtered = z2_fft*filter_function2

y2_filtered_1 = np.fft.ifft(z2_filtered)
y2_filtered_2 = np.fft.ifft(z2_dft_filtered)
y2_filtered_3 = np.fft.ifft(z2_fft_filtered)


plt.plot(time2,y2_filtered_1,label = "filtered built in FFT")
plt.plot(time2,y2_filtered_2, label = "filtered DFT")
plt.plot(time2,y2_filtered_3, label = "filtered FFT")
plt.plot(time2,Noisy_data.y21+Noisy_data.y22+Noisy_data.y23,label="actual")
plt.xlim(0,100)
plt.title("figure 7: Filtered Data")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.show()


#8192 data points
plt.plot(time3,y3)
plt.xlabel("Time")
plt.ylabel("Sum of 3 sine waves")
plt.title("figure 8: Noisy Data, 8192 discrete freq")
plt.xlim(0,80)
plt.show()

z3 = np.fft.fft(y3)
z3_dft = functions.dft(y3)
z3_fft = functions.cooley_tukey(y3)
freq3 = np.arange(8192)
peak31 = 1000
peak32 = 2000
peak33 = 3000

filter_function31 = (np.exp(-(freq3-peak31)**2/width)+np.exp(-(freq3+peak31-8192)**2/width))
filter_function32 = (np.exp(-(freq3-peak32)**2/width)+np.exp(-(freq3+peak32-8192)**2/width))
filter_function33 = (np.exp(-(freq3-peak33)**2/width)+np.exp(-(freq3+peak33-8192)**2/width))
filter_function3 = filter_function31+filter_function32+filter_function33
z3_filtered = z3*filter_function3
z3_dft_filtered = z3_dft*filter_function3
z3_fft_filtered = z3_fft*filter_function3
y3_filtered_1 = np.fft.ifft(z3_filtered)
y3_filtered_2 = np.fft.ifft(z3_dft_filtered)
y3_filtered_3 = np.fft.ifft(z3_fft_filtered)
plt.plot(time3,y3_filtered_1,label = "filtered built in FFT")
plt.plot(time3,y3_filtered_2, label = "filtered DFT")
plt.plot(time3,y3_filtered_3, label = "filtered FFT")
plt.plot(time3,Noisy_data.y31+Noisy_data.y32+Noisy_data.y33,label="actual")
plt.xlim(0,80)
plt.title("figure 9: Filtered Data")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
