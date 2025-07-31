import os, random
import torch 
import torchaudio
from torchaudio import transforms
import numpy as np
import config

class AudioUtil():
    @staticmethod
    def open(filename):
        signal = torch.load(filename)
        return(signal)
    
    @staticmethod
    def rechannel(audio):
        signal, sr = audio
        if (signal.shape[0] == 2):
            audio = (signal[:1], sr)
        elif (len(signal.shape) == 1):
            audio = (signal.unsqueeze(0), sr)
        return audio
        
    @staticmethod
    def resample(audio, new_sr):
        sample, sr = audio
        if (sr == new_sr):
            return audio
        resample = transforms.Resample(sr, new_sr)(sample[:1])
        return (resample, new_sr)
    
    @staticmethod
    def pad_trunc(aud, max_len):
        sig, sr = aud
        num_rows, sig_len = sig.shape

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        
        return (sig, sr)
    
    @staticmethod
    def MFCC(audio, n_mfcc=32, n_mels=64):
        signal, sr = audio

        return transforms.MFCC(sr, n_mfcc=n_mfcc, melkwargs={"n_mels": n_mels})(signal)
        
    @staticmethod
    def preprocess(audio):
        rechannel = AudioUtil.rechannel(audio)
        resample = AudioUtil.resample(rechannel, config.SAMPLE_RATE)
        pad = AudioUtil.pad_trunc(resample, config.SAMPLE_LENGTH)
        mel = AudioUtil.MFCC(pad)
        return torch.flatten(mel)
    

def load_data(root): 
    # Find all class folders and map them to integer labels
    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    global idx_to_class
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    samples = []
    # Find all audio files and their corresponding class index
    for class_name in classes:
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(root, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith('.pt'):
                path = os.path.join(class_dir, filename)
                audio = AudioUtil.open(path)
                print(audio)
                print(audio.shape)
                data = AudioUtil.preprocess((audio, config.SAMPLE_RATE))
                print(data)
                print(data.shape)
                samples.append((data, class_idx))

    data, class_idx = zip(*samples)
    return np.array(data), np.array(class_idx)
    