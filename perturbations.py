import torch
import wave
import librosa
import torchaudio.transforms as T
import numpy as np
import io
import random
from pydub import AudioSegment
from IPython.display import Audio
from datasets import load_dataset, Dataset
from scipy.signal import butter, lfilter

# Need to test
def pitch_scale(audio, pitch_factor=0):
    waveform = np.array(audio["audio"]["array"])
    sampling_rate = audio["audio"]["sampling_rate"]
    waveform_pitched = librosa.effects.pitch_shift(waveform.astype(np.float32), sr=sampling_rate, n_steps=pitch_factor)
    return {
        "audio": {
            "array": waveform_pitched,
            "sampling_rate": sampling_rate
        }
    }
    
def add_reverb(audio, reverb_factor=0.3):
    waveform = np.array(audio["audio"]["array"], dtype=np.float32)
    reverb_waveform = librosa.effects.preemphasis(waveform, coef=reverb_factor)
    return {
        "audio": {
            "array": reverb_waveform,
            "sampling_rate": audio["audio"]["sampling_rate"]
        }
    }
    
def change_volume(audio, volume_factor_range=(0.5, 1.5)):
    waveform = np.array(audio["audio"]["array"], dtype=np.float32)
    volume_factor = np.random.uniform(*volume_factor_range)
    volume_changed_waveform = waveform * volume_factor
    return {
        "audio": {
            "array": np.clip(volume_changed_waveform, -1.0, 1.0),
            "sampling_rate": audio["audio"]["sampling_rate"]
        }
    }

def low_pass_filter(audio, cutoff=1000):
    waveform = np.array(audio["audio"]["array"], dtype=np.float32)
    sampling_rate = audio["audio"]["sampling_rate"]
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(6, normal_cutoff, btype='low', analog=False)
    filtered_waveform = lfilter(b, a, waveform)
    return {
        "audio": {
            "array": filtered_waveform,
            "sampling_rate": sampling_rate
        }
    }


def time_stretch(audio, stretch_factor = 1):
    waveform = np.array(audio["audio"]["array"])
    waveform_stretched = librosa.effects.time_stretch(waveform.astype(np.float32), rate=stretch_factor)
    return {
        "audio": {
            "array": waveform_stretched,
            "sampling_rate": audio["audio"]["sampling_rate"]
        }
    }