import torch
from scipy import signal
import scipy.io
import scipy.io.wavfile as wav
import numpy as np
import h5py
import librosa
import sys
import os

# check gpu status
use_gpu = torch.cuda.is_available()

def make_spectrum_phase(y, FRAMESIZE, OVERLAP, FFTSIZE):
    D=librosa.stft(y,n_fft=FRAMESIZE,hop_length=OVERLAP,win_length=FFTSIZE,window=scipy.signal.hamming)
    Sxx = np.log10(abs(D)**2) 
    phase = np.exp(1j * np.angle(D))
    mean = np.mean(Sxx, axis=1).reshape((257,1))
    std = np.std(Sxx, axis=1).reshape((257,1))+1e-12
    Sxx = (Sxx-mean)/std  
    return Sxx, phase, mean, std

def recons_spec_phase(Sxx_r, phase):
    Sxx_r = np.sqrt(10**Sxx_r)
    R = np.multiply(Sxx_r , phase)
    result = librosa.istft(R,
                     hop_length=256,
                     win_length=512,
                     window=scipy.signal.hamming)
    return result

if len(sys.argv) < 3:
    print ("Usage: python test_gen_spec.py model.hdf5 list_noisy")
    sys.exit(1) 

# load model
model_path = "src/model/" + sys.argv[1]
model=torch.load(model_path) #"weights/DNN_spec_20160425v2.hdf5"
model.eval()

FRAMESIZE = 512
OVERLAP = 256
FFTSIZE = 512
RATE = 16000
FRAMEWIDTH = 2
FBIN = FRAMESIZE//2+1
noisylistpath = "src/dataset/" + sys.argv[2]

with open(noisylistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print (filename)
        y,sr=librosa.load(line[:-1],sr=RATE)
        training_data = np.empty((10000, FBIN, FRAMEWIDTH*2+1)) # For Noisy data

        Sxx, phase, mean, std = make_spectrum_phase(y, FRAMESIZE, OVERLAP, FFTSIZE)
        idx = 0     
        for i in range(FRAMEWIDTH, Sxx.shape[1]-FRAMEWIDTH): # 5 Frmae
            training_data[idx,:,:] = Sxx[:,i-FRAMEWIDTH:i+FRAMEWIDTH+1] # For Noisy data
            idx = idx + 1

        X_train = training_data[:idx]
        X_train = np.reshape(X_train,(idx,-1))

        # change to pytorch tensor
        X_train = X_train.astype(np.float32)
        if use_gpu:
            x = torch.tensor(X_train).cuda()
        else:
            x = torch.tensor(X_train)
        y_pred = model(x)

        predict = y_pred.detach().numpy()

        count=0
        for i in range(FRAMEWIDTH, Sxx.shape[1]-FRAMEWIDTH):
            Sxx[:,i] = predict[count]
            count+=1
        # # The un-enhanced part of spec should be un-normalized
        Sxx[:, :FRAMEWIDTH] = (Sxx[:, :FRAMEWIDTH] * std) + mean
        Sxx[:, -FRAMEWIDTH:] = (Sxx[:, -FRAMEWIDTH:] * std) + mean    

        recons_y = recons_spec_phase(Sxx, phase)
        output = librosa.util.fix_length(recons_y, y.shape[0])
        wav.write(os.path.join("src/enhanced_voice/",filename),RATE,np.int16(output*32767))
