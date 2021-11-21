# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:05:49 2020
test on DSD100_test 
@author: psp
"""


import scipy.io as sio
import torch 
from torch import nn
import numpy as np
from openunmix import model as MODEL
from utils import load_model, save_model, sizeof_fmt
import scipy
import scipy.io.wavfile as wav
import scipy.io as sio
from openunmix.model import Decoder_TorchISTFT, Encoder_TorchSTFT
import matplotlib.pyplot as plt
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## parameter setting #########
nfft = 512
hop = 128
hidden_size = 1024
batch_size = 1
def wavwrite(fn, data, fs):
    data = scipy.array(scipy.around(data * 2**(15)), dtype = "int16")
    wav.write(fn, fs, data)
# ============== initialize model =================
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

Checkpoint = torch.load(checkpoint_path)
model.load_state_dict(Checkpoint['state_dict'])
model.eval()
h0 = torch.zeros([model.lstm.num_layers*2,batch_size, model.lstm.hidden_size]).to(device)
c0 = torch.zeros([model.lstm.num_layers*2,batch_size, model.lstm.hidden_size]).to(device)
#%% =============create testing data signal==========================
#____input testing data____#
T60 = 0
for T60 in range(11):
    print('creating T60=0.'+str(T60)+'...')
    if T60==1:
        continue
    test_folder = '../../../DATA\ADSP_T60=0_target_normalizedClean\ADSP專題_Data/'
    data1=sio.loadmat(test_folder+'T60_0.'+str(T60)+'_test_data.mat')

    x_test1 = data1['x']
    y_test1 = data1['y']
    theta1 = data1['theta']

# tempx = x_test1[:,:,0]
# tempx = np.reshape(tempx,[-1,32000])
# tempy = y_test1[:,:,0]
# tempy = np.reshape(tempy,[-1,32000])
# wavwrite('noisy1.wav', np.transpose(tempx[0,:]), 16000)
# wavwrite('clean.wav', tempy[0,:], 16000)
# tempx = x_test1[:,:,1]
# tempx = np.reshape(tempx,[-1,32000])
# wavwrite('noisy2.wav', np.transpose(tempx[0,:]), 16000)

    x_test = np.concatenate([x_test1], axis = 0)
    x_test = np.transpose(x_test,[2,0,1])
    x_test = np.reshape(x_test,[2,-1,32000])
    x_test = np.transpose(x_test,[1,0,2])
    y_test = np.concatenate([y_test1],axis = 0)
    y_test = np.transpose(y_test,[2,0,1])
    y_test = np.reshape(y_test,[2,-1,32000])
    y_test = np.transpose(y_test,[1,0,2])
    theta = np.concatenate([theta1],axis = 0)
    theta = np.reshape(theta,[-1,2])


    
    print('strat testing....')
    dir_name = 'created_mat/T60=0.'+str(T60)
    if os.path.isdir(dir_name):
        print('dir exist!')
    else:
        os.mkdir(dir_name)
        print('create dir!')
    with torch.no_grad():
        for i in range(x_test.shape[0]):
        
            X_C1 = encoder(torch.tensor(x_test[i,0,:]).float().to(device))
            X_C2 = encoder(torch.tensor(x_test[i,1,:]).float().to(device))
            mag_X1 = torch.sqrt(torch.pow(X_C1[:,:,0],2)+torch.pow(X_C1[:,:,1],2)) #[batch,channel,f_bin,t_frame]
            mag_X2 = torch.sqrt(torch.pow(X_C2[:,:,0],2)+torch.pow(X_C2[:,:,1],2)) #[batch,channel,f_bin,t_frame]
            pow_X1 = torch.pow(X_C1[:,:,0],2)+torch.pow(X_C1[:,:,1],2)
            pow_X2 = torch.pow(X_C2[:,:,0],2)+torch.pow(X_C2[:,:,1],2)
        
        
            Y_C1 = encoder(torch.tensor(y_test[i,0,:]).float().to(device))
            Y_C2 = encoder(torch.tensor(y_test[i,1,:]).float().to(device))
            mag_Y1 = torch.sqrt(torch.pow(Y_C1[:,:,0],2)+torch.pow(Y_C1[:,:,1],2)) #[batch,channel,f_bin,t_frame]
            mag_Y2 = torch.sqrt(torch.pow(Y_C2[:,:,0],2)+torch.pow(Y_C2[:,:,1],2)) #[batch,channel,f_bin,t_frame]
            pow_Y1 = torch.pow(Y_C1[:,:,0],2)+torch.pow(Y_C1[:,:,1],2)
            pow_Y2 = torch.pow(Y_C2[:,:,0],2)+torch.pow(Y_C2[:,:,1],2)

            
            IRM_temp = (pow_Y1)/(pow_X1)
            one = torch.ones([1],dtype=torch.float32,device=device)
            IRM1 = torch.where(IRM_temp>one,one,IRM_temp)
            IRM_temp = (pow_Y2)/(pow_X2)
            one = torch.ones([1],dtype=torch.float32,device=device)
            IRM2 = torch.where(IRM_temp>one,one,IRM_temp)
        
            pow_X_temp = torch.where(pow_X1==0,torch.min(pow_X1),pow_X1)
            in_1 = torch.log(pow_X_temp.to(device))
            in_1 = torch.reshape(in_1,[1,1,257,-1])
            pow_X_temp = torch.where(pow_X2==0,torch.min(pow_X2),pow_X2)
            in_2 = torch.log(pow_X_temp.to(device))
            in_2 = torch.reshape(in_2,[1,1,257,-1])

            mask_esti_1,(hn,cn) = model(in_1,h0,c0) #[batch,channel,f_bin,frame]
            mask_esti_2,(hn,cn) = model(in_2,h0,c0) #[batch,channel,f_bin,frame]
            mask_esti_1 = torch.squeeze(mask_esti_1)
            mask_esti_2 = torch.squeeze(mask_esti_2)
    
        
            pow_Y1_estimate = torch.squeeze(pow_X1)*mask_esti_1
            mag_Y1_estimate = torch.sqrt(pow_Y1_estimate)
            pow_Y2_estimate = torch.squeeze(pow_X2)*mask_esti_2
            mag_Y2_estimate = torch.sqrt(pow_Y2_estimate)
        
        
            X1_phase = torch.angle(torch.view_as_complex(X_C1))
            X2_phase = torch.angle(torch.view_as_complex(X_C2))
            Y1_estimate = torch.polar(mag_Y1_estimate, X1_phase)
            Y2_estimate = torch.polar(mag_Y2_estimate, X2_phase)
            Y1_estimate =torch.view_as_real(Y1_estimate)
            Y2_estimate =torch.view_as_real(Y2_estimate)
            y1_estimate = decoder(Y1_estimate[:,:,:]).cpu().detach().numpy()
            y1_estimate = np.expand_dims(y1_estimate, 1)
            y2_estimate = decoder(Y2_estimate[:,:,:]).cpu().detach().numpy()
            y2_estimate = np.expand_dims(y2_estimate, 1)

            wavwrite('signal/T60=0.'+str(T60)+'_enhanced_'+str(i)+'.wav', np.concatenate([y1_estimate,y2_estimate],1), 16000)
            wavwrite('signal/T60=0.'+str(T60)+'_noisy_'+str(i)+'.wav', np.transpose(x_test[i,:,:],[1,0]), 16000)
            wavwrite('signal/T60=0.'+str(T60)+'_clean_'+str(i)+'.wav', np.transpose(y_test[i,:,:],[1,0]), 16000)

            spec1 = torch.unsqueeze(torch.view_as_complex(X_C1),0)
            spec2 = torch.unsqueeze(torch.view_as_complex(X_C2),0)
            spec = torch.cat([spec1,spec2],0)
            spec =spec.detach().cpu().numpy()
        
            mask1 = torch.unsqueeze(mask_esti_1,0)
            mask2 = torch.unsqueeze(mask_esti_2,0)
            mask = torch.cat([mask1,mask2],0)
            mask =mask.detach().cpu().numpy()
            
            IRM1 = torch.unsqueeze(IRM1,0)
            IRM2 = torch.unsqueeze(IRM2,0)
            IRM = torch.cat([IRM1,IRM2],0)
            IRM = IRM.detach().cpu().numpy()
            sio.savemat(dir_name+'/sample_'+str(i)+'.mat', {"spectrogram": spec, "mask": mask,"theta":theta[i,0],'IRM':IRM})
    

    