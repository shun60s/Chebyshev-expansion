#coding:utf-8

# A trial numpy chebyshev polynomial expansion
# to the signal after apply impulse response

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.16.3
#  scipy 1.4.1
#  matplotlib  2.1.1

import sys
import argparse
import numpy as np
import numpy.polynomial.chebyshev as chev
from scipy import signal
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from matplotlib import pyplot as plt




def plot_gen(x, title0):
    # general waveform draw
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title( title0 )
    plt.xlabel('step')
    plt.ylabel('value')
    ax.plot(x)
    plt.grid()
    plt.axis('tight')
    plt.show()

def plot_compare(xw0s,xw1s):
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Comparison In(red) Out(Blue)')
    plt.xlabel('step')
    plt.ylabel('value')
    ax.plot(xw0s,'r')
    ax.plot(xw1s,'b')
    plt.grid()
    plt.axis('tight')
    plt.show()

def get_portion(xw0,xw1,sel0,max_corr_index, flag_plot=False):
    #
    if sel0 == 0:
        xw0s= xw0[max_corr_index: max_corr_index+len(xw1)].copy()
        xw1s= xw1.copy()
    else:
        xw0s= xw0.copy()
        xw1s= xw1[-1*max_corr_index: -1*max_corr_index+len(xw0)].copy()
        
    if flag_plot :
        plot_compare(xw0s,xw1s)
    
    return xw0s, xw1s


def make_same_length( xw0, xw1):
    len0 = len( xw0)
    len1 = len( xw1)
    if len0 > len1 :
        xw1b = np.pad( xw1, [0, len0-len1], 'constant')
        xw0b = xw0
        lenb = len0
        sel0 = 0
        print(' len(xw0) > len(w1) ')
    elif len1 > len0 :
        xw0b = np.pad( xw0, [0, len1-len0], 'constant')
        xw1b = xw1
        lenb = len1
        sel0 = 1
        print(' len(xw1) > len(w0) ')
    else :
        xw0b = xw0
        xw1b = xw1
        lenb = len0
        sel0 = 0
        print(' len(xw0) = len(w1) ')
    return xw0b, xw1b, lenb, sel0

def load_wav( path0):
    # return 
    #        yg: wav data (mono) 
    #        sr: sampling rate
    try:
        sr, y = wavread(path0)
    except:
        print ('error: wavread ', path0)
        sys.exit()
    else:
        yg= y / (2 ** 15)
        if yg.ndim == 2:  # if stereo
            yg= np.average(yg, axis=1)
    
    print ('file ', path0)
    print ('sampling rate ', sr)
    print ('length ', len(yg))
    return yg,sr,len(yg)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='trial numpy chebyshev polynomial expansion')
    parser.add_argument('--wav_file0', '-i', default='wav/short_overlapadd_out.wav', help='input wav file name(16bit) mono')
    parser.add_argument('--wav_file1', '-o', default='wav/short-output-TwoTube-rtwdf_part1.wav', help='output wav file name(16bit) mono')
    args = parser.parse_args()
    path0= args.wav_file0
    path1= args.wav_file1
    
    # overwrite
    if 1:
        # path0= 'wav/short.wav'                     # input: the signal before apply impulse response
        # path0= 'wav/short_overlapadd_out.wav'      # input: the signal after  apply impulse response
        
        # path1= 'wav/short-output-TwoTube-rtwdf_part2.wav'  # sample fragment 2nd
        # path1= 'wav/short-output-TwoTube-rtwdf_part3.wav'  # sample fragment 3rd
        pass
        
    # load compared two wav files
    xw0,sr0,len0=load_wav(path0)
    xw1,sr1,len1=load_wav(path1)
    
    xw0b,xw1b,lenb,sel0 = make_same_length( xw0, xw1)
    
    corr= signal.correlate(xw0b, xw1b, mode='same')
    max_corr_index=np.argmax(corr) - int(len(corr)/2)  # due to be centered 
    print(' max corr index', max_corr_index)
    # plot_gen(corr, 'corr')
    
    xwIn, xwOut= get_portion(xw0,xw1,sel0,max_corr_index, flag_plot=True)
    
    
    #
    deg0=20  # if warning, try small deg value.
    t_s = chev.chebfit( xwIn, xwOut, deg0)
    y_c = chev.chebval( xwIn, t_s)
    print ( 't_s', t_s)
    plot_gen( t_s, 't_s')
    plot_compare(xwOut,y_c)