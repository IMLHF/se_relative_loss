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
import tqdm
from nn_se.utils.assess.core import calc_pesq, calc_stoi, calc_sdr, calc_SegSNR

smg = None

noisy_and_ref_list = [
  # ('exp/paper_sample/0db_mix_ref/p265_015_nfree_0678.wav', 'exp/paper_sample/0db_mix_ref/p265_015.wav'),
  # ('exp/paper_sample/0db_mix_ref/p265_026_nfree_0571.wav', 'exp/paper_sample/0db_mix_ref/p265_026.wav'),
  # ('exp/paper_sample/0db_mix_ref/p265_038_nfree_0758.wav', 'exp/paper_sample/0db_mix_ref/p265_038.wav'),
  # ('exp/paper_sample/0db_mix_ref/p267_087_nfree_0663.wav', 'exp/paper_sample/0db_mix_ref/p267_087.wav'),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10001_1zcIwhmdeo4_00001.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10001_1zcIwhmdeo4_00002.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10001_1zcIwhmdeo4_00003.wav', None),
  ('exp/paper_sample/voxceleb1_sample/mixed/id10001_utrA-v8pPm4_00001.wav', None),
  ('exp/paper_sample/voxceleb1_sample/mixed/id10006_3RybHF5mX78_00001.wav', None),
  # ('exp/paper_sample/voxceleb1_sample/mixed/id10006_3RybHF5mX78_00002.wav', None),
  ('exp/paper_sample/voxceleb1_sample/mixed/id10006_5tGaUGO_z50_00001.wav', None),
  ('exp/paper_sample/voxceleb1_sample/mixed/id10006_zQROl4ZsMVA_00002.wav', None),
]
save_dir = 'exp/paper_sample/voxceleb1_sample'

def magnitude_spectrum_librosa_stft(signal, NFFT, overlap):
  signal = np.array(signal, dtype=np.float)
  tmp = librosa.core.stft(signal,
                          n_fft=NFFT,
                          hop_length=NFFT-overlap,
                          window=scipy.signal.windows.hann)
  tmp = np.absolute(tmp)
  return tmp.T

def enhance_and_calcMetrics(noisy_and_ref):
  figsize = (2.8, 1.6)
  noisy_wav_dir, ref_wav_dir = noisy_and_ref
  noisy_wav, sr = audio.read_audio(noisy_wav_dir)
  ref_wav, sr = audio.read_audio(ref_wav_dir) if ref_wav_dir else (None, sr)
  config_name = FLAGS.PARAM().config_name()
  noisy_stem = Path(noisy_wav_dir).stem
  global smg
  if smg is None:
    smg = inference.build_SMG(ckpt_name=os.path.join('exp', config_name, 'ckpt'))
  enhanced_out = inference.enhance_one_wav(smg, noisy_wav)
  enhanced_wav = enhanced_out.enhanced_wav
  # enhanced_mag = enhanced_out.enhanced_mag
  mask = enhanced_out.mask

  ## plot mask, enhanced_mag; save enhanced_wav; calc metrics [pesq]
  name_prefix = "%s_%s" % (config_name, noisy_stem)

  # calc metrics [pesq_noisy->pesq_enhanced | pesqi, stoi, sdr, ssnr]
  if ref_wav is not None:
    pesq_noisy = calc_pesq(ref_wav, noisy_wav, sr)
    pesq_enhanced = calc_pesq(ref_wav, enhanced_wav, sr)
    pesqi = pesq_enhanced - pesq_noisy
    stoi_noisy = calc_stoi(ref_wav, noisy_wav, sr)
    stoi_enhanced = calc_stoi(ref_wav, enhanced_wav, sr)
    stoii = stoi_enhanced - stoi_noisy
    sdr_noisy = calc_sdr(ref_wav, noisy_wav, sr)
    sdr_enhanced = calc_sdr(ref_wav, enhanced_wav, sr)
    sdri = sdr_enhanced - sdr_noisy
    ssnr_noisy = calc_SegSNR(ref_wav/np.max(ref_wav), noisy_wav/np.max(noisy_wav), 256, 256)
    ssnr_enhanced = calc_SegSNR(ref_wav/np.max(ref_wav), enhanced_wav/np.max(enhanced_wav), 256, 256)
    ssnri = ssnr_enhanced - ssnr_noisy

    metrics_eval_ans_f = "%s_metrics_eval_ans.log" % config_name
    with open(os.path.join(save_dir, metrics_eval_ans_f), 'a') as f:
      f.write(name_prefix+":\n")
      f.write("    pesq: %.3f->%.3f, pesqi: %.3f.\n" % (pesq_noisy, pesq_enhanced, pesqi))
      f.write("    stoi: %.3f->%.3f, stoii: %.3f.\n" % (stoi_noisy, stoi_enhanced, stoii))
      f.write("    sdr : %.3f->%.3f, sdri : %.3f.\n" % (sdr_noisy, sdr_enhanced, sdri))
      f.write("    ssnr: %.3f->%.3f, ssnri: %.3f.\n" % (ssnr_noisy, ssnr_enhanced, ssnri))

  # get x_ticks
  n_frame = np.shape(mask)[0]
  n=0
  i=0
  x1 = []
  x2 = []
  while(n<n_frame):
    x1.append(n)
    x2.append(i)
    n = n+125
    i = i+1

  # mask
  plt.figure(figsize=figsize)
  plt.pcolormesh(mask.T, vmin=-0.2, vmax=1.5)
  plt.subplots_adjust(top=0.97, right=0.96, left=0.17, bottom=0.27)
  plt.xlabel('Time(S)')
  plt.ylabel('Frequency(Hz)')
  plt.xticks(x1, x2)
  plt.yticks((0,33,64,97,129), ("0","1k","2k","3k","4k"))
  plt.colorbar()
  plt.savefig(os.path.join(save_dir,"%s_%s" % (name_prefix, "mask.jpg")))
  # plt.show()
  plt.close()

  # enhanced_mag
  enhanced_mag = magnitude_spectrum_librosa_stft(enhanced_wav, 256, 64*3)
  enhanced_mag = np.log(enhanced_mag*3+1e-2)
  plt.figure(figsize=figsize)
  # print(np.max(enhanced_mag), np.min(enhanced_mag))
  plt.pcolormesh(enhanced_mag.T, cmap='hot', vmin=-4.5, vmax=2.5)
  plt.subplots_adjust(top=0.97, right=0.96, left=0.17, bottom=0.27)
  # plt.title('STFT Magnitude')
  plt.xlabel('Time(S)')
  plt.ylabel('Frequency(Hz)')
  plt.xticks(x1, x2)
  plt.yticks((0,33,64,97,129), ("0","1k","2k","3k","4k"))
  plt.colorbar()
  plt.savefig(os.path.join(save_dir,"%s_%s" % (name_prefix, "enhanced_mag.jpg")))
  # plt.show()
  plt.close()

  # enhanced_wav
  audio.write_audio(os.path.join(save_dir, "%s_%s" % (name_prefix, "enhanced.wav")), enhanced_wav, sr)

  # noisy_mag
  noisy_mag_file = os.path.join(save_dir, "%s_%s" % (noisy_stem, "noisy_mag.jpg"))
  noisy_mag = magnitude_spectrum_librosa_stft(noisy_wav, 256, 64*3)
  noisy_mag = np.log(noisy_mag*3+1e-2)
  plt.figure(figsize=figsize)
  plt.pcolormesh(noisy_mag.T, cmap='hot', vmin=-4.5, vmax=2.5)
  plt.subplots_adjust(top=0.97, right=0.96, left=0.17, bottom=0.27)
  # plt.title('STFT Magnitude')
  plt.xlabel('Time(S)')
  plt.ylabel('Frequency(Hz)')
  plt.xticks(x1, x2)
  plt.yticks((0,33,64,97,129), ("0","1k","2k","3k","4k"))
  plt.colorbar()
  plt.savefig(noisy_mag_file)
  # plt.show()
  plt.close()

  # clean_mag
  clean_mag_file = os.path.join(save_dir, "%s_%s" % (noisy_stem, "clean_mag.jpg"))
  if ref_wav is not None:
    clean_mag = magnitude_spectrum_librosa_stft(ref_wav, 256, 64*3)
    clean_mag = np.log(clean_mag*3+1e-2)
    plt.figure(figsize=figsize)
    plt.pcolormesh(clean_mag.T, cmap='hot', vmin=-4.5, vmax=2.5)
    plt.subplots_adjust(top=0.97, right=0.96, left=0.17, bottom=0.27)
    # plt.title('STFT Magnitude')
    plt.xlabel('Time(S)')
    plt.ylabel('Frequency(Hz)')
    plt.xticks(x1, x2)
    plt.yticks((0,33,64,97,129), ("0","1k","2k","3k","4k"))
    plt.colorbar()
    plt.savefig(clean_mag_file)
    # plt.show()
    plt.close()

if __name__ == "__main__":
  # enhance_and_calcMetrics(noisy_and_ref_list[0])
  for noisy_and_ref in tqdm.tqdm(noisy_and_ref_list, ncols=100):
    enhance_and_calcMetrics(noisy_and_ref)
