#coding:utf-8


# A trial ARMA Spectral density power estimation
# by using SPECTRUM : Spectral Analysis in Python <https://github.com/cokelaer/spectrum>
# 
#----------------------------------------------------------
# This trial sees output something wrong (doubtful). 
# It's unclear if it's correct or due to any false use.
#----------------------------------------------------------

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.16.3
#  scipy 1.4.1
#  matplotlib  2.1.1
#  spectrum 0.7.6

import sys
import argparse
import numpy as np
import numpy.polynomial.chebyshev as chev
from scipy import signal
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from matplotlib import pyplot as plt
import spectrum  # SPECTRUM : Spectral Analysis in Python <https://github.com/cokelaer/spectrum>


class Class_ARMA(object):
    # The result changes by max_lag value, also, each dim size.
    def __init__(self, xw, dim_a=20, dim_b=20, max_lag=100, NFFT=4096, sr=44100, Flag_Show=True):
        #
        self.xw= xw.copy()
        self.sr= sr # sampling rate
        self.NFFT= NFFT
        self.Flag_Show= Flag_Show
        # ARMA model, a and b
        self.a, self.b, self.rho = spectrum.arma_estimate(np.array(self.xw), dim_a, dim_b, max_lag)
        if self.Flag_Show:
            print ('a=', self.a)
            print ('b=', self.b)
            print ('White noise variance estimate (rho) ', self.rho)
        
        # rho_1=1.0  # if force to use 1 as rho
        self.psd = spectrum.arma2psd(A=self.a, B=self.b, rho=self.rho, NFFT=self.NFFT, norm=False)
        self.psd2 = self.psd[0:int(len(self.psd)/2)]  # get half side
        self.get_peak()
        
        # other ways
        self.psd2_ar= None
        self.psd_welch= None
        
    def get_peak(self, ):
        # get peak frequency
        self.max_ids= signal.argrelmax( self.psd2, order=1)
        self.peak_freqs= np.array(self.max_ids[0]) * (self.sr / len(self.psd))
        print ('peak frequency=', self.peak_freqs)
        
        
    def ar_model(self,): # another way
        # AR model using the Yule-Walker (autocorrelation) method
        self.ar, self.var, self.reflec = spectrum.aryule( np.array(self.xw), 30, allow_singularity=True)
        if self.Flag_Show:
            print ('a=', self.ar)
            
        self.psd_ar = spectrum.arma2psd(A=self.ar,  NFFT=self.NFFT, norm=False)
        self.psd2_ar = self.psd_ar[0:int(len(self.psd_ar)/2)]  # get half side
        
    def welch_psd(self,): # another way
        # power spectral density, Welch’s method
        self.freqList_welch, self.psd_welch = signal.welch( np.array(self.xw), self.sr,  window='hanning',  nperseg=self.NFFT)
        
        

def plot_arma( arma1, arma2):
    #
    freqs = np.linspace(0, arma1.sr /2 , len( arma1.psd2) )
    fig = plt.figure()
    plt.plot(freqs, 10 * np.log10( arma1.psd2), label='ARMA(In)')  # semilogx
    plt.plot(freqs, 10 * np.log10( arma2.psd2), label='ARMA(Out)') # semilogx
    
    if arma1.psd2_ar is not None:
        freqs_ar = np.linspace(0, arma1.sr /2 , len( arma1.psd2_ar) )
        plt.plot(freqs_ar, 10 * np.log10( arma1.psd2_ar), label='AR(In)') # semilogx
        
    if arma1.psd_welch is not None:
        plt.plot(arma1.freqList_welch, 10 * np.log10( arma1.psd_welch), label='Welch PSD(In)') # semilogx
    
    plt.plot(freqs[ arma1.max_ids],10 * np.log10( arma1.psd2[arma1.max_ids]) ,'ro',label='ARMA(In) Peaks')
    plt.plot(freqs[ arma2.max_ids],10 * np.log10( arma2.psd2[arma2.max_ids]) ,'bo',label='ARMA(Out) Peaks')
    
    plt.ylabel('Spectral density power [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.legend()
    plt.grid()
    plt.axis('tight')
    plt.show()

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
    
    parser = argparse.ArgumentParser(description='trial ARMA Spectral density power estimation')
    parser.add_argument('--wav_file0', '-i', default='wav/short.wav', help='input wav file name(16bit) mono')
    parser.add_argument('--wav_file1', '-o', default='wav/short-output-TwoTube-rtwdf_part1.wav', help='output wav file name(16bit) mono')
    args = parser.parse_args()
    path0= args.wav_file0
    path1= args.wav_file1
    
    # overwrite
    if 1:
        # path0= 'wav/short.wav'                     # input: the signal before apply impulse response
        # path0= 'wav/short_overlapadd_out.wav'      # input: the signal after  apply impulse response
        
        #path1= 'wav/short-output-TwoTube-rtwdf_part2.wav'  # sample fragment 2nd
        #path1= 'wav/short-output-TwoTube-rtwdf_part3.wav'  # sample fragment 3rd
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
    
    # ARMA model
    arma1= Class_ARMA(xwIn,  sr=sr0)
    arma2= Class_ARMA(xwOut, sr=sr1)
    
    # to comarison with other ways
    arma1.ar_model()  # compute AR model
    arma1.welch_psd() # compute Welch PSD
    
    plot_arma( arma1, arma2)
    
    
