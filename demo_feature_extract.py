import numpy as np
from scipy import signal
import emg_functions
import os

main_path='./'
data_path = main_path + '/dataset_demo'
feature_path = main_path + '/features_demo'

nTree=100
fs=2000
window_len_time=0.2
window_step_time=0.1
window_len=int(np.floor(window_len_time*fs))
window_step=int(np.floor(window_step_time*fs))
signal_duration=2
reaction_duration=1
Nsample=int((signal_duration)*fs)
reaction_len=int(np.ceil(reaction_duration*fs))
Nwindow=int(np.round((Nsample-window_len)/window_step+1))
Nwindow_start=int(np.round((reaction_len-window_len)/window_step+1))

Nchannel=8
ssc_thresh=0
zc_thresh=0
subjects = np.array(range(1))
class_num=6
label_order=['power','lateral','tripod','pointer','open','rest']

for subject in subjects:
    folders = sorted(os.listdir(data_path + '/' + str(subject)))
    for folder in folders:
        print('extracting features of subject' + str(subject) + ' in ' + folder)

        data=np.load(data_path + '/' + str(subject) + '/' + folder + '/data.npy')

        feature_mav = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')
        feature_wl = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')
        feature_ssc = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')
        feature_zc = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')
        feature_rms = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')
        feature_vcf = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')
        feature_mdf = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')
        feature_mnf = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')
        feature_pkf = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')
        feature_skw = np.zeros([np.size(data,axis=0), Nwindow-Nwindow_start,Nchannel], 'float')

        for trial in range(np.size(data,axis=0)):

            data_raw=data[trial,:,:]
            data_filter=list()
            for idx_ch in range(np.size(data_raw,axis=0)):
                data_channel=data_raw[idx_ch,:]
                b, a = signal.butter(4, (10 * 2 / fs, 500 * 2 / fs), btype='bandpass')
                data_channel_filter=signal.lfilter(b,a,data_channel,axis=-1,zi=None)
                data_filter.append(data_channel_filter)

            data_filter = np.transpose(np.vstack(data_filter))

            wl = np.reshape(emg_functions.get_wl(data_filter, window_len, window_step),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]
            ssc = np.reshape(emg_functions.get_ssc(data_filter, window_len, window_step, ssc_thresh),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]
            zc = np.reshape(emg_functions.get_zc(data_filter, window_len, window_step, zc_thresh),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]
            rms = np.reshape(emg_functions.get_rms(data_filter, window_len, window_step),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]
            mav = np.reshape(emg_functions.get_mav(data_filter, window_len, window_step),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]
            skw = np.reshape(emg_functions.get_skw(data_filter, window_len, window_step),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]
            vcf = np.reshape(emg_functions.get_vcf(data_filter, window_len, window_step, fs),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]
            mdf = np.reshape(emg_functions.get_mdf(data_filter, window_len, window_step, fs),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]
            mnf = np.reshape(emg_functions.get_mnf(data_filter, window_len, window_step, fs),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]
            pkf = np.reshape(emg_functions.get_pkf(data_filter, window_len, window_step, fs),(-1,Nchannel),'C')[Nwindow_start:Nwindow,:]

            feature_ssc[trial, :, :] = ssc
            feature_zc[trial, :, :] = zc
            feature_rms[trial, :, :] = rms
            feature_mav[trial, :, :] = mav
            feature_wl[trial, :, :] = wl
            feature_vcf[trial, :, :] = vcf
            feature_mdf[trial, :, :] = mdf
            feature_mnf[trial, :, :] = mnf
            feature_pkf[trial, :, :] = pkf
            feature_skw[trial, :, :] = skw

        feature_save_path = feature_path +'/' + str(subject) + '/' + folder
        if not os.path.exists(feature_save_path):
            os.makedirs(feature_save_path)
        np.save(feature_save_path + '/feature_ssc.npy', feature_ssc)
        np.save(feature_save_path + '/feature_zc.npy', feature_zc)
        np.save(feature_save_path + '/feature_rms.npy', feature_rms)
        np.save(feature_save_path + '/feature_mav.npy', feature_mav)
        np.save(feature_save_path + '/feature_wl.npy', feature_wl)
        np.save(feature_save_path + '/feature_vcf.npy', feature_vcf)
        np.save(feature_save_path + '/feature_mdf.npy', feature_mdf)
        np.save(feature_save_path + '/feature_mnf.npy', feature_mnf)
        np.save(feature_save_path + '/feature_pkf.npy', feature_pkf)
        np.save(feature_save_path + '/feature_skw.npy', feature_skw)
print('feature extraction completed')