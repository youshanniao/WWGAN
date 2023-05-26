# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def slice_concat(seed, slice_index):
    slicedata = np.empty([0,2])
    cyclenum = 0
    for i in slice_index:
        iternum = np.loadtxt('data' + '/' + str(seed) + '_' + 'slice' + '_' + str(int(i)) + '.csv')
        iternum = iternum.reshape(np.size(iternum), 1)
        print('mean=' + str(np.mean(iternum)))
        print('std=' + str(np.std(iternum)))
        cyclenum += np.size(iternum)
        slicenum = np.full((np.size(iternum),1), cyclenum)
        slicedata_i = np.concatenate((iternum, slicenum), axis=1)
        slicedata = np.append(slicedata, slicedata_i, axis=0)
    sliced = pd.DataFrame(data=slicedata, columns=['Value', 'Time'])
    print(sliced) 
    return slice