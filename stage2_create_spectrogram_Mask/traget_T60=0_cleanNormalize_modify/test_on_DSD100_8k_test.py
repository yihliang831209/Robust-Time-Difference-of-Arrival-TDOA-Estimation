# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:05:49 2020
test on DSD100_test 
@author: psp
"""

from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import scipy.io as sio
import torch 
from torch import distributed, nn
import numpy as np
from openunmix import data
from openunmix import model as MODEL
from openunmix import utils
from openunmix import transforms
from utils import human_seconds, load_model, save_model, sizeof_fmt
import copy
import sklearn.preprocessing
import tqdm
import scipy.io as sio
import scipy
import scipy.signal as signal
import scipy.io.wavfile as wav
from openunmix.model import Decoder_TorchISTFT, Encoder_TorchSTFT
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## parameter setting #########
nfft = 512
hop = 128
hidden_size = 1024
batch_size = 1
epochs_size = 30
def compute_measures(se,s):
    Rss=s.transpose().dot(s)
    this_s=s

    a=this_s.transpose().dot(se)/Rss
    e_true=a*this_s
    e_res=se-a*this_s
    Sss=np.sum((e_true)**2)
    Snn=np.sum((e_res)**2)

    SDR=10*np.log10(Sss/Snn)

    # Rsr= s.transpose().dot(e_res)
    # b=np.linalg.inv(Rss).dot(Rsr)

    # e_interf = s.dot(b)
    # e_artif= e_res-e_interf

    # SIR=10*np.log10(Sss/np.sum((e_interf)**2))
    # SAR=10*np.log10(Sss/np.sum((e_artif)**2))
    return SDR#, SIR, SAR
def wavwrite(fn, data, fs):
    data = scipy.array(scipy.around(data * 2**(15)), dtype = "int16")
    wav.write(fn, fs, data)
class Wavedata(Dataset):
    def __init__(self,mix,vocal_music):
        self.mix = mix
        self.vocal_music = vocal_music
    def __len__(self):

        return len(self.mix[:,0,0])

    def __getitem__(self,idx):
        data = self.mix[idx,:,:]
        target = self.vocal_music[idx,:,:]
        
        return torch.tensor(data).float(), torch.tensor(target).float()

#%% =============create testing data signal==========================
#____input testing data____#
data1=sio.loadmat('../DATA/ADSP專題_Data/test_data_1.mat')
data2=sio.loadmat('../DATA/ADSP專題_Data/test_data_2.mat')




x_test1 = data1['x'][:,:,0]
y_test1 = data1['y'][:,:]
x_test1_2 = data1['x'][:,:,1]
y_test1_2 = data1['y'][:,:]

x_test2 = data2['x'][:,:,0]
y_test2 = data2['y'][:,:]
x_test2_2 = data2['x'][:,:,1]
y_test2_2 = data2['y'][:,:]

x_test = np.concatenate([x_test1,x_test1_2,x_test2,x_test2_2], axis = 0)
x_test = np.expand_dims(x_test, 1)
y_test = np.concatenate([y_test1,y_test1_2,y_test2,y_test2_2],axis = 0)

# model save
def save_checkpoint(checkpoint_path, model, optimizer):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizer’s states and hyperparameters used.
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

num_epoch = 99
checkpoint_path=r'.\checkpoint/tasnet_'+str(num_epoch)+'epoch.pth'
encoder = Encoder_TorchSTFT(n_fft_short = nfft, n_hop=hop,center=True).to(device)
decoder = Decoder_TorchISTFT(n_fft_short = nfft, n_hop=hop,center=True).to(device)
# scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset)
model = MODEL.OpenUnmix(
        nb_bins=nfft // 2 + 1,
        nb_channels=1,
        nb_output_channels=1,
        hidden_size=hidden_size,
        unidirectional=False
    ).to(device)
print(model)
## compute model size ================================
size = sizeof_fmt(4 * sum(p.numel() for p in model.parameters()))
size_2 = np.sum([p.numel() for p in model.parameters()]).item()
print(f"Model size {size}")

# (temp_in,_) = next(iter(train_loader))
# macs, params = profile(model, inputs=(temp_in.to(device), ))
# macs, params = clever_format([macs, params], "%.3f")
## compute model size END ================================


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.L1Loss()
Checkpoint = torch.load(checkpoint_path)
model.load_state_dict(Checkpoint['state_dict'])
optimizer.load_state_dict(Checkpoint['optimizer'])
model.eval()
print('strat testing....')
# for name, param in model.named_parameters():
#     print(name)
# params = model.state_dict()
# encoder_weight_long = params['encoder.conv1d_long.weight']
# encoder_weight_short = params['encoder.conv1d_short.weight']
# decoder_weight_long = params['decoder.decode_long.weight']
# decoder_weight_short = params['decoder.decode_short.weight']


# sio.savemat('8k_encoder_MF480_32_8_noVAD_enRelu_maskRelu.mat',{'encoder_long':(encoder_weight_long.cpu()).detach().numpy(),'encoder_short':(encoder_weight_short.cpu()).detach().numpy(),'decoder_long':(decoder_weight_long.cpu()).detach().numpy(),'decoder_short':(decoder_weight_short.cpu()).detach().numpy()})


vocal_SDR = []
music_SDR = []
vocal_SDRi = []
music_SDRi = []
mix_vocal_SDR = []
mix_music_SDR = []


mix_cat = np.zeros([0])
vocal_cat = np.zeros([0])
music_cat = np.zeros([0])
estimate_vocal_cat = np.zeros([0])
estimate_music_cat = np.zeros([0])
h0 = torch.zeros([model.lstm.num_layers*2,batch_size, model.lstm.hidden_size]).to(device)
c0 = torch.zeros([model.lstm.num_layers*2,batch_size, model.lstm.hidden_size]).to(device)
count = 0
dir_name = 'signal/test_2sec_opbased_allsamples_'+str(num_epoch)+'epoch'
if os.path.isdir(dir_name):
    print('dir exist!')
else:
    os.mkdir(dir_name)
    print('create dir!')
    
# data shapeing
op_based = 2  # op_based=0 means whole song as one basic
x_test = np.reshape(x_test,[-1,op_based*16000])
y_test = np.reshape(y_test,[-1,op_based*16000])
    
    
for i in range(len(x_test)):
    # Forward pass
    x_temp = x_test[i,:]
    x_temp = np.reshape(x_temp,[1,-1])
    y_temp = y_test[i,:]
    y_temp = np.reshape(y_temp,[1,-1])
    print(str(i)+' out of '+str(len(x_test)))
    data_in = np.expand_dims(x_temp,0)
    data_in = torch.tensor(data_in).float()
    y = np.expand_dims(y_temp,0)
    y = torch.tensor(y).float()
    with torch.no_grad():
        # Forward pass 
        X = encoder(data_in.to(device)) #[batch,channel,f_bin,t_frame,real/imag]
        mag_X = torch.sqrt(torch.pow(X[:,:,:,:,0],2)+torch.pow(X[:,:,:,:,1],2)) #[batch,channel,f_bin,t_frame]
        pow_X = torch.pow(X[:,:,:,:,0],2)+torch.pow(X[:,:,:,:,1],2)
        in_ = torch.log(pow_X+(torch.ones(mag_X.shape)*0.000001).to(device))
        mask_esti,(hn,cn) = model(in_,h0,c0)
        mask_esti = torch.squeeze(mask_esti)
        pow_Y_estimate = torch.squeeze(pow_X)*mask_esti
        mag_Y_estimate = torch.sqrt(pow_Y_estimate)
        ## creat upper bound #############################
        Y = encoder(y.to(device))
        mag_Y = torch.sqrt(torch.pow(Y[:,:,:,:,0],2)+torch.pow(Y[:,:,:,:,1],2)) #[batch,channel,f_bin,t_frame]
        pow_Y = torch.pow(Y[:,:,:,:,0],2)+torch.pow(Y[:,:,:,:,1],2)
        IRM = torch.squeeze(pow_Y)/(torch.squeeze(pow_X)+(torch.ones(torch.squeeze(mag_X).shape)*0.001).to(device))
        pow_Y_estimate_up = torch.squeeze(pow_X)*IRM
        mag_Y_estimate_up = torch.sqrt(pow_Y_estimate_up)
        loss = criterion(mask_esti,IRM.to(device))
        train_loss=loss.item()
        print(train_loss)
        X_phase = torch.angle(torch.view_as_complex(X))
        X_phase = torch.squeeze(X_phase)
        Y_estimate = torch.polar(mag_Y_estimate, X_phase)
        Y_estimate =torch.view_as_real(Y_estimate)
        y_estimate = decoder(Y_estimate[:,:,:],16000*op_based).cpu().detach().numpy()
        
        Y_estimate_up = torch.polar(mag_Y_estimate_up, X_phase)
        Y_estimate_up =torch.view_as_real(Y_estimate_up)
        y_estimate_up = decoder(Y_estimate_up[:,:,:],16000*op_based).cpu().detach().numpy()
        
        Y_estimate_up = torch.polar(torch.squeeze(mag_Y), X_phase)
        Y_estimate_up =torch.view_as_real(Y_estimate_up)
        cleanMag_noisyPhase = decoder(Y_estimate_up[:,:,:],16000*op_based).cpu().detach().numpy()
        
        Y_phase = torch.angle(torch.view_as_complex(Y))
        Y_phase = torch.squeeze(Y_phase)
        Y_estimate_up = torch.polar(torch.squeeze(mag_Y), Y_phase)
        Y_estimate_up =torch.view_as_real(Y_estimate_up)
        cleanMag_cleanPhase = decoder(Y_estimate_up[:,:,:],16000*op_based).cpu().detach().numpy()

    
    wavwrite(dir_name+'/enhanced_'+str(i)+'.wav', y_estimate, 16000)
    # wavwrite(dir_name+'/enhanced_up_'+str(i)+'.wav', y_estimate_up, 16000)
    wavwrite(dir_name+'/clean_'+str(i)+'.wav', y_temp[0,:], 16000)
    wavwrite(dir_name+'/noisy_'+str(i)+'.wav', x_temp[0,:], 16000)
    # wavwrite(dir_name+'/cleanMag_noisyPhase_'+str(i)+'.wav', cleanMag_noisyPhase, 16000)
    # wavwrite(dir_name+'/cleanMag_cleanPhase_'+str(i)+'.wav', cleanMag_cleanPhase, 16000)

   
    