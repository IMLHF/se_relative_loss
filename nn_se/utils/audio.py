from numpy import linalg
import numpy as np
import soundfile as sf
import librosa

from ..FLAGS import PARAM

'''
soundfile.info(file, verbose=False)
soundfile.available_formats()
soundfile.available_subtypes(format=None)
soundfile.read(file, frames=-1, start=0, stop=None, dtype='float64', always_2d=False, fill_value=None, out=None, samplerate=None, channels=None, format=None, subtype=None, endian=None, closefd=True)
soundfile.write(file, data, samplerate, subtype=None, endian=None, format=None, closefd=True)
'''

def read_audio(file):
  data, sr = sf.read(file)
  if sr != PARAM.sampling_rate:
    data = librosa.resample(data, sr, PARAM.sampling_rate, res_type='kaiser_fast')
    print('resample wav(%d to %d) :' % (sr, PARAM.sampling_rate), file)
    # librosa.output.write_wav(file, data, PARAM.sampling_rate)
  return data, PARAM.sampling_rate

def write_audio(file, data, sr):
  return sf.write(file, data, sr)

def repeat_to_len(wave, repeat_len, random_trunc_long_wav=False):
  wave_len = len(wave)
  if random_trunc_long_wav and wave_len > repeat_len:
    random_s = np.random.randint(wave_len-repeat_len+1)
    wave = wave[random_s:random_s+repeat_len]
    return wave

  while len(wave) < repeat_len:
    wave = np.tile(wave, 2)
  wave = wave[0:repeat_len]
  return wave


def mix_wav_by_SNR(waveData, noise, snr):
  As = linalg.norm(waveData)
  An = linalg.norm(noise)

  alpha = As/(An*(10**(snr/20))) if An != 0 else 0
  waveMix = (waveData+alpha*noise)/(1.0+alpha)
  # return mixed, speech_weight, noise_weight
  return waveMix, 1.0/(1.0+alpha), alpha/(1.0+alpha)
