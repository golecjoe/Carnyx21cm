import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def make_data_arrays(resispaths,skypaths):

	data1 = np.loadtxt(resispaths[0],delimiter=',',skiprows=2)

	freq_array = data1[:,1]

	resisaud = np.zeros(data1[:,2].size)
	skyaud = np.zeros(data1[:,2].size)

	for i in resispaths:
		tmpdata = np.loadtxt(i,delimiter=',',skiprows=2)
		resisaud += tmpdata[:,2]
	resisaud /= len(resisfiles)

	for i in skypaths:
		tmpdata = np.loadtxt(i,delimiter=',',skiprows=2)
		skyaud += tmpdata[:,2]
	skyaud /= len(skyfiles)
	n = 30

	smooth_sky = np.convolve(skyaud,np.ones(n)/n,mode='valid') 
	smooth_resis = np.convolve(resisaud,np.ones(n)/n,mode='valid') 
	smooth_freq = np.convolve(freq_array,np.ones(n)/n,mode='valid')

	return smooth_freq,smooth_resis,smooth_sky


resisfiles = glob.glob('take_50ohm_data_4/*.csv')
skyfiles = glob.glob('take_21cm_data_4/*.csv')
smooth_freq_4,smooth_resis_4,smooth_sky_4 = make_data_arrays(resisfiles,skyfiles)
ratio_4 = smooth_sky_4/smooth_resis_4

resisfiles = glob.glob('take_50ohm_data_3/*.csv')
skyfiles = glob.glob('take_21cm_data_3/*.csv')
smooth_freq_3,smooth_resis_3,smooth_sky_3 = make_data_arrays(resisfiles,skyfiles)
ratio_3 = smooth_sky_3/smooth_resis_3

resisfiles = glob.glob('take_50ohm_data/*.csv')
skyfiles = glob.glob('take_21cm_data/*.csv')
smooth_freq_1,smooth_resis_1,smooth_sky_1 = make_data_arrays(resisfiles,skyfiles)
ratio_1 = smooth_sky_1/smooth_resis_1

resisfiles = glob.glob('take_50ohm_data_5/*.csv')
skyfiles = glob.glob('take_21cm_data_5/*.csv')
smooth_freq_5,smooth_resis_5,smooth_sky_5 = make_data_arrays(resisfiles,skyfiles)
ratio_5 = smooth_sky_5/smooth_resis_5

resisfiles = glob.glob('take_50ohm_data_6/*.csv')
skyfiles = glob.glob('take_21cm_data_6/*.csv')
smooth_freq_6,smooth_resis_6,smooth_sky_6 = make_data_arrays(resisfiles,skyfiles)
ratio_6 = smooth_sky_6/smooth_resis_6

resisfiles = glob.glob('take_50ohm_data_7/*.csv')
skyfiles = glob.glob('take_21cm_data_7/*.csv')
smooth_freq_7,smooth_resis_7,smooth_sky_7 = make_data_arrays(resisfiles,skyfiles)
ratio_7 = smooth_sky_7/smooth_resis_7

plt.figure()
plt.plot(smooth_freq_1,ratio_1-np.mean(ratio_1))
plt.plot(smooth_freq_3,ratio_3-np.mean(ratio_3))
# plt.plot(smooth_freq_4,ratio_4-np.mean(ratio_4))
# plt.plot(smooth_freq_5,ratio_5-np.mean(ratio_5))
# plt.plot(smooth_freq_6,ratio_6-np.mean(ratio_6))
# plt.plot(smooth_freq_7,ratio_7-np.mean(ratio_7))

plt.show()