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
#fre1=128
n1 = 128
peak1 = [10, 20, 30]
#freq2=1024
n2 = 1024
peak2 = [100,200,300]
#freq3 = 8192
n3 = 8192
peak3 = [1000,2000,3000]

true = [Noisy_data.y11+Noisy_data.y12+Noisy_data.y13,
        Noisy_data.y21+Noisy_data.y22+Noisy_data.y23,
        Noisy_data.y31+Noisy_data.y32+Noisy_data.y33]



def get_data(n, y, peak):
    """Return filter data from three method
    """
    # change original data to the frequency basis
    z = np.fft.fft(y)
    z_dft = functions.dft(y)
    z_fft = functions.cooley_tukey(y)
    
    zs = [z, z_dft, z_fft]
    
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
    
    zfs = [z_filtered,z_dft_filtered,z_fft_filtered]
    
    #get filtered data
    y_filtered_1 = np.fft.ifft(z_filtered)#buildin_fft
    y_filtered_2 = np.fft.ifft(z_dft_filtered) #dft
    y_filtered_3 = np.fft.ifft(z_fft_filtered) #fft
    
    ys = [y_filtered_1, y_filtered_2, y_filtered_3]
    
    return zs, filter_function, zfs, ys


def plot_original(time, y, n):
    """ show the original data with noise
    """
    plt.plot(time,y)
    if n > 200:
        plt.xlim(0,100)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("figure1: Noisy Data, {} discrete freq".format(n))
    plt.show()


def plot_fft(z, z_dft, z_fft):
    """
    """
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("figure 2: 3 FFT methods: frequency basis and corresponding amplitude")
    plt.plot(np.abs(z),label = "built in FFT")
    plt.plot(np.abs(z_dft),label = "DFT")
    plt.plot(np.abs(z_fft),label = "Cooley-Tukey FFt")
    plt.legend()
    plt.show()

def plot_gaussion(freq, filter_function):
    plt.plot(np.arange(freq),filter_function)
    plt.xlabel("Frequency")
    plt.ylabel("Probability")
    plt.title("figure 3: Gaussian (probability) filter function")
    plt.show()

def plot_fitered_freq_basis(n, z_filtered, z_dft_filtered, z_fft_filtered):
    """plot the filtered frequency basis
    """
    freq = np.arange(n)
    plt.plot(freq, abs(z_filtered),label = "filtered built in FFT")
    plt.plot(freq, abs(z_dft_filtered), label = "filtered DFT")
    plt.plot(freq, abs(z_fft_filtered), label = "filtered FFT")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("figure 4: Filtered Frequency Basis")
    plt.legend()
    plt.show()


def plot_filtered_data(y_filtered_1, y_filtered_2, y_filtered_3, time, index):
    """plot the filtered data
    """
    plt.plot(time,y_filtered_1,label = "filtered built in FFT")
    plt.plot(time,y_filtered_2, label = "filtered DFT")
    plt.plot(time,y_filtered_3, label = "filtered FFT")
    plt.plot(time,true[index-1],label="actual")
    plt.legend()
    plt.xlim(0,100)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("figure 5: Filtered Data")
    plt.show()
    
    

data1 = get_data(n1, y1, peak1)#zs, filter_function, zfs, ys
data2 = get_data(n2, y2, peak2)
data3 = get_data(n3, y3, peak3)
#n1 = 128
plot_original(time1, y1, n1)
plot_fft(data1[0][0], data1[0][1], data1[0][2])
plot_gaussion(n1, data1[1])
plot_fitered_freq_basis(n1, data1[2][0], data1[2][1], data1[2][2])
plot_filtered_data(data1[3][0], data1[3][1], data1[3][2], time1, 1)


#n1 = 1024
plot_original(time2, y2, n2)
plot_fft(data2[0][0], data2[0][1], data2[0][2])
plot_filtered_data(data2[3][0], data2[3][1], data2[3][2], time2, 2)


#n1 = 8192
plot_original(time3, y3, n3)
plot_fft(data3[0][0], data3[0][1], data3[0][2])
plot_filtered_data(data3[3][0], data3[3][1], data3[3][2], time3, 3)


