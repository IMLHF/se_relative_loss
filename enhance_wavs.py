from nn_se import inference
from nn_se.utils import audio
from nn_se.utils import misc_utils
from nn_se import FLAGS
from matplotlib import pyplot as plt
import os
from pathlib import Path
import numpy as np
import librosa
import scipy

smg = None

noisy_and_ref_list = [
  ('exp/paper_sample/0db_mix_ref/p265_015_nfree_0678.wav', 'exp/paper_sample/0db_mix_ref/p265_015.wav'),
  ('exp/paper_sample/0db_mix_ref/p265_026_nfree_0571.wav', 'exp/paper_sample/0db_mix_ref/p265_026.wav'),
  ('exp/paper_sample/0db_mix_ref/p265_038_nfree_0758.wav', 'exp/paper_sample/0db_mix_ref/p265_038.wav'),
  ('exp/paper_sample/0db_mix_ref/p274_110_nfree_0663.wav', 'exp/paper_sample/0db_mix_ref/p274_110.wav'),
]

def magnitude_spectrum_librosa_stft(signal, NFFT, overlap):
  signal = np.array(signal, dtype=np.float)
  tmp = librosa.core.stft(signal,
                          n_fft=NFFT,
                          hop_length=NFFT-overlap,
                          window=scipy.signal.windows.hann)
  tmp = np.absolute(tmp)
  return tmp.T

def enhance_and_calcMetrics(noisy_and_ref):
  noisy_wav_dir, ref_wav_dir = noisy_and_ref
  noisy_wav, sr = audio.read_audio(noisy_wav_dir)
  ref_wav, sr = audio.read_audio(ref_wav_dir) if ref_wav_dir else (None, sr)
  config_name = FLAGS.PARAM().config_name()
  save_dir = 'exp/paper_sample'
  noisy_stem = Path(noisy_wav_dir).stem
  global smg
  if smg is None:
    smg = inference.build_SMG(ckpt_name=os.path.join('exp', config_name, 'ckpt'))
  enhanced_out = inference.enhance_one_wav(smg, noisy_wav)
  enhanced_wav = enhanced_out.enhanced_wav
  # enhanced_mag = enhanced_out.enhanced_mag
  mask = enhanced_out.mask

  # plot mask, enhanced_mag; save enhanced_wav
  # plt.imshow(mask.T[:,::-1], cmap='hot', origin='low')
  plt.figure(figsize=(7, 4))
  plt.pcolormesh(mask.T, cmap='hot', vmin=-0.5, vmax=2)
  plt.subplots_adjust(top=0.98, right=1)
  # plt.title('STFT Magnitude')
  plt.xlabel('Frame')
  plt.ylabel('Frequency')
  plt.colorbar(shrink=.83)
  plt.savefig(os.path.join(save_dir,"%s_%s_%s" % (config_name, noisy_stem, "mask.jpg")))
  # plt.show()
  plt.close()

  plt.figure(figsize=(7, 4))
  enhanced_mag = magnitude_spectrum_librosa_stft(enhanced_wav, 256, 128)
  enhanced_mag = np.log(enhanced_mag*50+1e-8)
  print(np.max(enhanced_mag), np.min(enhanced_mag))
  plt.pcolormesh(enhanced_mag.T, cmap='hot', vmin=-2)
  plt.subplots_adjust(top=0.98, right=1)
  # plt.title('STFT Magnitude')
  plt.xlabel('Frame')
  plt.ylabel('Frequency')
  plt.colorbar(shrink=.83)
  plt.savefig(os.path.join(save_dir,"%s_%s_%s" % (config_name, noisy_stem, "mag.jpg")))
  # plt.show()
  plt.close()





enhance_and_calcMetrics(noisy_and_ref_list[0])
