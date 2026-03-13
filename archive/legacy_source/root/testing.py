import numpy as np
import mne

filePath = 'Data/sourcedata/rawdata/S001/'
fileName = 'S001R01'
edf = mne.io.read_raw_edf(filePath+fileName+'.edf')
channelNames = [[j.strip('.,') for j in i] for i in [edf.ch_names]]
header = ','.join(channelNames[0])
np.savetxt(fileName+'.csv', edf.get_data().T, delimiter=',', header=header, comments='')

# filePath = 'S001R01.csv'