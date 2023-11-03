#############################################################
# ECG Data Iris Spectrogram UI using Streamlit in python    #
# Author: Dong woo Lee                                      #
# "Generated with the assistance of OpenAI's ChatGPT."      #
#############################################################

import streamlit as st
import pandas as pd
import os
import pickle
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time as current_time
import wfdb
import scipy.signal as sp_signal

from Pan_tompkins_algorithm import Pan_tompkins  # https://github.com/Pramod07Ch/Pan-Tompkins-algorithm-python/blob/main/Pan_tompkins_algorithm.py
from physionet_processing import spectrogram  # https://github.com/awerdich/physionet/blob/master/physionet_processing.py
from cartesian_to_polar import *

def data_read(dir):
    # 데이터 로드
    record = wfdb.rdrecord(dir)
    ts = record.p_signal
    fs = record.fs
    return ts, fs

def zero_pad(signal, target_length, sampling_rate):
    padded_signal = np.zeros(target_length)
    original_length = min(len(signal), target_length)
    padded_signal[:original_length] = signal[:original_length]
    return padded_signal, np.linspace(0, (target_length-1)/sampling_rate, target_length)

def main(file_path):
    data = pd.read_csv(file_path, header=None, names=['ID', 'Label'])
    
    st.title('Iris Spectrogram Data Visualization')
    selected_label = st.selectbox('Choose a Label for visualization:', data['Label'].unique())
    selected_ids = data[data['Label'] == selected_label]['ID'].tolist()
    selected_id = st.selectbox('Choose an ID for visualization:', selected_ids)
    selected_ecg_preprocessing = st.checkbox('Enable ECG Preprocessing', value=True)
    col_zero_padding = st.checkbox('Column Zero Padding', value=True)
    row_zero_padding = st.checkbox('Row Zero Padding', value=True)
    selected_method = st.selectbox('Choose an Interpolation Method:', ['linear', 'cubic', 'nearest'])
    scatter_size = st.slider('Select Scatter Point Size:', min_value=1, max_value=20, value=5)
    grid_resolution = st.slider('Select Grid Resolution:', min_value=1, max_value=10, value=5)
    fname =  f"training2017/{selected_id}"

    if os.path.exists(f"{fname}.hea") :
        signal, fs = data_read(fname)
        ts = signal[:,0]
        time = np.arange(0, len(ts))/fs

        if selected_ecg_preprocessing:
            pan_tompkins = Pan_tompkins(ts, fs)
            ts = pan_tompkins.fit()
            min_length = min(len(time), len(ts))
            time = time[:min_length]
            ts = ts[:min_length]
        else :
            pass

        if row_zero_padding:
            max_length = 61
            ts, time = zero_pad(ts, max_length * fs, fs)

        f1, PSD = sp_signal.periodogram(ts, fs, 'flattop', scaling='density')
        Sx = spectrogram(np.expand_dims(ts, axis = 0), log_spectrogram = False)[2]
        Sx_log = spectrogram(np.expand_dims(ts, axis = 0), log_spectrogram = True)[2]
        log_spectrogram_image_data = np.transpose(Sx_log[0])

        if col_zero_padding:
            min_value = log_spectrogram_image_data.min()
            padding_value = min_value
            padding_size = log_spectrogram_image_data.shape[0]
            log_spectrogram_image_data_padded = np.pad(log_spectrogram_image_data, ((0, padding_size), (0, 0)), 'constant', constant_values=padding_value)

            log_spectrogram_image_data_padded[log_spectrogram_image_data_padded == padding_value] = np.nan # zero padding to mapping NaN
            img = np.flipud(log_spectrogram_image_data_padded)

        else :
            img = np.flipud(log_spectrogram_image_data)

        d = img
        nx, ny = d.shape

        k = create_kspace_locations(nx, ny)
        kx = np.real(k)
        ky = np.imag(k)

        method = selected_method
        grid_d, grid_x, grid_y = grid2(d, kx, ky, 2**grid_resolution, method=method)

        fig, axs = plt.subplots(3, 3, figsize=(24, 24))
        for ax in axs.flat:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        axs[0, 0:3] = fig.add_subplot(3, 3, (1, 3)) # Set the 1st subplot to take up the space of the 2nd and 3rd subplots
        axs[0, 0].plot(time, ts, 'b')
        axs[0, 0].set(xlabel='Time [s]', xlim=[0, time[-1]], xticks=np.arange(0, time[-1]+5, 10))
        axs[0, 0].set(ylabel='Potential [mV]')
        axs[0, 0].set_title('ECG Time Series', fontsize=15)

        axs[1, 0].plot(f1, PSD, 'b')
        axs[1, 0].set(xlabel='Frequency [Hz]', ylabel='PSD')
        axs[1, 0].set_title('Power Spectral Density (PSD)', fontsize=15)

        axs[1, 1].imshow(np.transpose(Sx[0]), aspect='auto', cmap='jet', origin='lower')
        axs[1, 1].set_title('Spectrogram without log transform', fontsize=15)
        axs[1, 1].set(xlabel='Time [s]', ylabel='Frequency [Hz]')

        axs[1, 2].imshow(np.transpose(Sx_log[0]), aspect='auto', cmap='jet', origin='lower')
        axs[1, 2].set_title('Spectrogram with log transform', fontsize=15)
        axs[1, 2].set(xlabel='Time [s]', ylabel='Frequency [Hz]')

        axs[2, 0].scatter(kx, ky, s=scatter_size, c=d.flatten(), cmap='jet')
        axs[2, 0].set_xlim(min(kx), max(kx))
        axs[2, 0].set_ylim(min(ky), max(ky))
        axs[2, 0].axis('off')
        axs[2, 0].set_title('Scatter Plot')

        axs[2, 1].imshow(grid_d.T, extent=(min(kx), max(kx), min(ky), max(ky)), origin='lower', aspect='auto', cmap='jet')
        axs[2, 1].set_xlabel('kx')
        axs[2, 1].set_ylabel('ky')
        axs[2, 1].set_title('Interpolated k-space data')
        axs[2, 1].axis('off')

        axs[2, 2] = fig.add_subplot(3, 3, 9, projection='3d')
        axs[2, 2].set_box_aspect([8,7,6])
        surf = axs[2, 2].plot_surface(grid_x, grid_y, grid_d, cmap='jet', alpha=0.5)
        # fig.colorbar(surf, ax=axs[0, 2])
        axs[2, 2].set_xlabel('kx')
        axs[2, 2].set_ylabel('ky')
        axs[2, 2].set_zlabel('Intensity')
        axs[2, 2].view_init(elev=20, azim=120)
        axs[2, 2].set_title('3D Plot')
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.write("The selected file does not exist.")
        st.write(f"Current Directory: {os.getcwd()}")

if __name__ == "__main__":
    file_path = "./training2017/REFERENCE-original.csv"
    main(file_path)