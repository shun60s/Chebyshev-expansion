#coding:utf-8

# overlap-add convolve with impulse response waveform


import sys
import os
import argparse
import numpy as np
from scipy import signal
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.16.3
#  scipy 1.4.1

def load_wav( path0, force_mono=False):
    # return 
    #        yg: wav data
    #        sr: sampling rate
    try:
        sr, y = wavread(path0)
    except:
        print ('error: wavread ', path0)
        sys.exit()
    else:
        yg= y / (2 ** 15)
        if force_mono :
            if yg.ndim == 2:  # if stereo
                yg= np.average(yg, axis=1)
    
    print ('file ', path0)
    print ('sampling rate ', sr)
    print ('length ', yg.shape)
    print ('yg.max', np.amax( np.abs(yg)))
    return yg,sr
    
def load_wav32( path0, wave_len, yg_in):
    #
    #        wave_len: impluse effective length time [sec]
    # return 
    #        yg: wav data (stereo)
    #        sr: sampling rate
    try:
        sr, y = wavread(path0)
    except:
        print ('error: wavread ', path0)
        sys.exit()
    else:
        len0= int(wave_len * sr)
        yg= y[sr : sr+len0] # / (2 ** 31)
        
        if yg_in.ndim == 2:
            yg2=np.hstack((yg,yg)).reshape( 2, len(yg) ).T
        else:
            yg2=yg.copy()
    
    print ('file ', path0)
    print ('sampling rate ', sr)
    print ('yg2.shape', yg2.shape)
    print ('yg.max', np.amax( np.abs(yg)), yg[0],yg[-1])
    return yg2, sr

def save_wav( path0, data, sr=44100, normalize=False):
    #
    print ('file ', path0)
    
    amplitude = np.iinfo(np.int16).max
    max_data = np.amax(np.abs(data))  # normalize, max level is 16bit full bit
    if max_data <  (1.0 / amplitude):
        max_data=1.0
    
    try:
        if normalize :
            wavwrite(path0, sr, np.array( (amplitude / max_data) * data , dtype=np.int16))
        else:
            wavwrite(path0, sr, np.array( amplitude  * data , dtype=np.int16))
    except:
        print ('error: wavwrite ', path0)
        sys.exit()



if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='overlap-add convolve with impulse response waveform')
    parser.add_argument('--wav_file', '-w', default='wav/short.wav', help='wav file name(16bit)')
    parser.add_argument('--wav_32_file', '-i', default='wav/impulse_1sec_100_1sec_44100-TwoTube-output-rtwdf.wav', help='Two Tube Amplifier impulse response wav file name (mono 32bit)')
    args = parser.parse_args()
    
    
    path0= args.wav_file
    yg,sr= load_wav(path0)
    
    path2= args.wav_32_file
    yg2,sr2= load_wav32(path2, 0.150, yg)
    
    # overlap-add convolve with impulse response waveform
    out1= signal.oaconvolve( yg, yg2, axes=0)  # need scipy > 1.4.1
    # set output file name
    #path_out0= os.path.splitext(os.path.basename(path0))[0] + '_overlapadd_out.wav'
    path_out0= os.path.splitext(path0)[0] + '_overlapadd_out.wav'
    save_wav( path_out0, out1, sr, normalize=True)
