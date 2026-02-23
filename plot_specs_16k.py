import os
import glob
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def plot_melspec(audio_file, save_file, n_mels=256, hop_length=128, n_fft=1024, figsize=(16, 6), dpi=300):
        y, sr = sf.read(audio_file, dtype='float32')
            
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            fmax=sr//2,
            power=2.0
        )

        S_db = librosa.power_to_db(S, ref=np.max)
        
        plt.figure(figsize=figsize, dpi=dpi)
        
        img = librosa.display.specshow(
            S_db, 
            sr=sr,
            hop_length=hop_length,
            cmap='magma'  # or 'viridis', 'plasma', 'inferno'
        )
            
        plt.axis('off')
        
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        plt.savefig(
            save_file, 
            bbox_inches='tight', 
            pad_inches=0, 
            dpi=dpi,
            facecolor='black',
            transparent=True
        )
        
        plt.close()
    

def plot_linearspec(audio_file, save_file, n_mels=256, hop_length=128, n_fft=1024, figsize=(16, 6), dpi=300):
        y, sr = sf.read(audio_file, dtype='float32')
            
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=hop_length*2)
    
        S_db = librosa.power_to_db(np.abs(S)**2, ref=np.max)
        
        plt.figure(figsize=figsize, dpi=dpi)
        
        img = librosa.display.specshow(
            S_db, 
            sr=sr,
            hop_length=hop_length,
            cmap='magma'  # or 'viridis', 'plasma', 'inferno'
        )
            
        plt.axis('off')

        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        plt.savefig(
            save_file, 
            bbox_inches='tight', 
            pad_inches=0, 
            dpi=dpi,
            facecolor='black',
            transparent=True
        )
        
        plt.close()
    

def process_audio_files(input_folder, output_folder, plot_func, ext='wav', **kwargs):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    audio_files = librosa.util.find_files(input_folder, ext=ext)
    print("Find {} audio files in {}".format(len(audio_files), input_folder))
    
    for audio_file in tqdm(audio_files):
        save_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(audio_file))[0]}.png")
        # print(f"Processing {audio_file} -> {save_file}")
        plot_func(audio_file, save_file, **kwargs)
        

if __name__ == "__main__":
    input_folder = 'wav'
    output_folder = 'fig'
    
    process_audio_files(input_folder, output_folder, plot_melspec, ext='wav', figsize=(8, 4), dpi=100)