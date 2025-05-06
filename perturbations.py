import torch
import wave
import librosa
import torchaudio.transforms as T
import numpy as np
import io
import random
from IPython.display import Audio
from datasets import load_dataset, Dataset
from scipy.signal import butter, lfilter

# Need to test
def add_gaussian_noise(data, noise_level=0.01):
    waveform = np.array(data["audio"]["array"], dtype=np.float32)
    noise = np.random.normal(0, noise_level, size=waveform.shape).astype(np.float32)
    noisy_waveform = waveform + noise
    data["audio"]["array"] = noisy_waveform
    return data

  
def pitch_scale(data, pitch_factor=1):
    waveform = np.array(data["audio"]["array"])
    sampling_rate = data["audio"]["sampling_rate"]
    waveform_pitched = librosa.effects.pitch_shift(waveform.astype(np.float32), sr=sampling_rate, n_steps=pitch_factor)
    data["audio"]["array"] = waveform_pitched
    return data
    
def add_reverb(data, reverb_factor=0.3):
    waveform = np.array(data["audio"]["array"], dtype=np.float32)
    reverb_waveform = librosa.effects.preemphasis(waveform, coef=reverb_factor)
    data["audio"]["array"] = reverb_waveform
    return data
    
def change_volume(data, volume_factor_range=(0.5, 1.5)):
    waveform = np.array(data["audio"]["array"], dtype=np.float32)
    volume_factor = np.random.uniform(*volume_factor_range)
    volume_changed_waveform = waveform * volume_factor
    volume_changed_waveform = np.clip(volume_changed_waveform, -1.0, 1.0)
    
    data["audio"]["array"] = volume_changed_waveform
    return data

def low_pass_filter(data, cutoff=1000):
    waveform = np.array(data["audio"]["array"], dtype=np.float32)
    sampling_rate = data["audio"]["sampling_rate"]
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(6, normal_cutoff, btype='low', analog=False)
    filtered_waveform = lfilter(b, a, waveform)
    
    data["audio"]["array"] = filtered_waveform
    return data

def time_stretch(data, stretch_factor = 1.5):
    waveform = np.array(data["audio"]["array"])
    waveform_stretched = librosa.effects.time_stretch(waveform.astype(np.float32), rate=stretch_factor)
    
    data["audio"]["array"] = waveform_stretched
    return data
    
def overlay_audio(speech_sample, sound_sample, mixing_ratio=1, start_time=0):
    """
    Overlays `sound_sample` onto `speech_sample` starting at `start_time` seconds.

    Args:
        speech_sample: Dict with keys 'audio' -> {'array', 'sampling_rate'}.
        sound_sample: Same format as speech_sample.
        mixing_ratio: Float multiplier for sound_sample volume.
        start_time: Time in seconds to start overlaying sound_sample.

    Returns:
        Modified speech_sample with overlaid audio.
    """
    speech_audio = np.array(speech_sample['audio']['array'], dtype=np.float32)
    speech_sr = speech_sample['audio']['sampling_rate']
    
    sound_audio = np.array(sound_sample['audio']['array'], dtype=np.float32)
    sound_sr = sound_sample['audio']['sampling_rate']
    
    # Resample if necessary
    if sound_sr != speech_sr:
        sound_audio = librosa.resample(sound_audio, orig_sr=sound_sr, target_sr=speech_sr)

    # Scale by mixing ratio
    sound_audio *= mixing_ratio

    # Determine start sample index
    start_sample = int(start_time * speech_sr)
    
    # Create a copy to modify
    mixed_audio = speech_audio.copy()

    # Ensure there's enough room to overlay sound
    end_sample = min(start_sample + len(sound_audio), len(mixed_audio))
    overlay_length = end_sample - start_sample
    if overlay_length > 0:
        mixed_audio[start_sample:end_sample] += sound_audio[:overlay_length]

    # Normalize if clipping
    if np.max(np.abs(mixed_audio)) > 1.0:
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
    
    # Return modified data
    output = dict(speech_sample)  # shallow copy is enough here
    output['audio'] = dict(speech_sample['audio'])  # avoid mutating input
    output['audio']['array'] = mixed_audio
    output['audio']['sampling_rate'] = speech_sr
    
    return output
  