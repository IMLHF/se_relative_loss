import os
# os.environ["OMP_NUM_THREADS"] = "1"
import tensorflow as tf
import collections
from pathlib import Path
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import sys
import json

from .utils import misc_utils
from .utils import audio
from .inference import build_SMG
from .inference import enhance_for_test
from .inference import SMG
from .utils.assess.core import calc_pesq, calc_stoi, calc_sdr, calc_SegSNR
from .FLAGS import PARAM

test_processor = 1
smg = None

class JSONEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(JSONEncoder, self).default(obj)

class TestsetEvalAns(
    collections.namedtuple("TestsetEvalAns",
                           ("pesq_noisy_mean", "pesq_noisy_var",
                            "stoi_noisy_mean", "stoi_noisy_var",
                            "sdr_noisy_mean", "sdr_noisy_var",
                            "ssnr_noisy_mean", "ssnr_noisy_var",
                            "pesq_enhanced_mean", "pesq_enhanced_var",
                            "stoi_enhanced_mean", "stoi_enhanced_var",
                            "sdr_enhanced_mean", "sdr_enhanced_var",
                            "ssnr_enhanced_mean", "ssnr_enhanced_var",
                            "mag_v_range_reMAE_mean", "mag_v_range_reMAE_var"))):
  pass


class RecordEvalAns(
    collections.namedtuple("TestsetEvalAns",
                           ("clean_wav_name", "noise_wav_name",
                            "pesq_noisy",
                            "stoi_noisy",
                            "sdr_noisy",
                            "ssnr_noisy",
                            "pesq_enhanced",
                            "stoi_enhanced",
                            "sdr_enhanced",
                            "ssnr_enhanced",
                            "mag_v_range_reMAE"))):
  pass


def avg_mag_mae(ref_mag, enc_mag):
  if np.sum(np.shape(ref_mag)) == 0:
    return 0.0
  abs_sum = np.abs(ref_mag)+np.abs(enc_mag)
  ans = np.mean(np.abs(ref_mag - enc_mag)/abs_sum)
  return ans


def get_mag_v_range_reMAE(ref_mag, enc_mag, mag_reMAE_v_range):
  mag_reMAE_v_range.sort()
  mag_v_range_reMAE_lst = []
  idx = (ref_mag > 1e-12) & (ref_mag <= mag_reMAE_v_range[0])
  mag_v_range_reMAE_lst.append(avg_mag_mae(ref_mag[idx], enc_mag[idx]))

  if len(mag_reMAE_v_range) > 1:
    for i, v in enumerate(mag_reMAE_v_range[1:]):
      s = mag_reMAE_v_range[i]
      e = v
      idx = (ref_mag > s) & (ref_mag <= e)
      mag_v_range_reMAE_lst.append(avg_mag_mae(ref_mag[idx], enc_mag[idx]))

  idx = (ref_mag > mag_reMAE_v_range[-1])
  mag_v_range_reMAE_lst.append(avg_mag_mae(ref_mag[idx], enc_mag[idx]))

  mag_v_range_reMAE_arr = np.array(mag_v_range_reMAE_lst, dtype=np.float32)
  return mag_v_range_reMAE_arr


def eval_one_record(clean_dir_and_noise_dir, mix_snr, save_dir=None):
  """
  if save_dir is not None: save clean, noise and mixed in save_dir.
  """
  global smg
  if smg is None:
    smg = build_SMG(finalizeG=True)
  clean_dir, noise_dir = clean_dir_and_noise_dir
  assert Path(clean_dir).exists(), 'clean_dir not exist.'
  assert Path(noise_dir).exists(), 'noise_dir not exist.'
  clean_wav, c_sr = audio.read_audio(clean_dir)
  noise_wav, n_sr = audio.read_audio(noise_dir)
  assert c_sr == n_sr and c_sr == PARAM.sampling_rate, 'Sample_rate error.'
  assert len(clean_wav) > 0 and len(noise_wav) > 0, 'clean or noise length is 0.'

  len_clean = len(clean_wav)
  noise_wav = audio.repeat_to_len(noise_wav, len_clean, False)
  mixed_wav, w_clean, w_noise = audio.mix_wav_by_SNR(clean_wav, noise_wav, mix_snr)
  clean_wav = clean_wav * w_clean
  noise_wav = noise_wav * w_noise
  enhanced_wav, enhanced_mag, clean_mag = enhance_for_test(smg, mixed_wav, clean_wav)

  clean_dir_name = Path(clean_dir).stem
  noise_dir_name = Path(noise_dir).stem
  if save_dir is not None:
    audio.write_audio(os.path.join(save_dir, clean_dir_name+'.wav'), clean_wav, PARAM.sampling_rate)
    audio.write_audio(os.path.join(save_dir, clean_dir_name+'_'+noise_dir_name+'.wav'), mixed_wav, PARAM.sampling_rate)
    audio.write_audio(os.path.join(save_dir, clean_dir_name+'_'+noise_dir_name+'_enhanced.wav'), enhanced_wav, PARAM.sampling_rate)

  pesq_noisy = calc_pesq(clean_wav, mixed_wav, PARAM.sampling_rate)
  stoi_noisy = calc_stoi(clean_wav, mixed_wav, PARAM.sampling_rate)
  sdr_noisy = calc_sdr(clean_wav, mixed_wav, PARAM.sampling_rate)
  ssnr_noisy = calc_SegSNR(clean_wav, mixed_wav, PARAM.frame_length, PARAM.frame_length//2)
  pesq_enhanced = calc_pesq(clean_wav, enhanced_wav, PARAM.sampling_rate)
  stoi_enhanced = calc_stoi(clean_wav, enhanced_wav, PARAM.sampling_rate)
  ssnr_enhanced = calc_SegSNR(clean_wav, enhanced_wav, PARAM.frame_length, PARAM.frame_length//2)
  sdr_enhanced = calc_sdr(clean_wav, enhanced_wav, PARAM.sampling_rate)

  mag_v_range_reMAE = get_mag_v_range_reMAE(clean_mag, enhanced_mag, PARAM.mag_reMAE_v_range)

  return RecordEvalAns(clean_wav_name=Path(clean_dir).stem, noise_wav_name=Path(noise_dir).stem,
                       pesq_noisy=pesq_noisy, stoi_noisy=stoi_noisy, sdr_noisy=sdr_noisy, ssnr_noisy=ssnr_noisy,
                       pesq_enhanced=pesq_enhanced, stoi_enhanced=stoi_enhanced, sdr_enhanced=sdr_enhanced, ssnr_enhanced=ssnr_enhanced,
                       mag_v_range_reMAE=mag_v_range_reMAE)


def eval_testSet_by_list(clean_noise_pair_list, mix_snr, save_dir=None):
  """
  if save_dir is not None: save clean, noise and mixed in save_dir.
  """

  func = partial(eval_one_record, mix_snr=mix_snr, save_dir=save_dir)
  pool = Pool(test_processor)
  job = pool.imap(func, clean_noise_pair_list)
  eval_ans_list = list(tqdm(job, "Testing SNR(%d)" % mix_snr, len(clean_noise_pair_list), unit="test record", ncols=60))
  pool.close()
  # eval_ans_list = []
  # # for clean_dir_and_noise_dir in clean_noise_pair_list:
  # for clean_dir_and_noise_dir in tqdm(clean_noise_pair_list, ncols=100, unit="test record"):
  #   eval_ans = eval_one_record(clean_dir_and_noise_dir, mix_snr, save_dir)
  #   eval_ans_list.append(eval_ans)
  #   # print(eval_ans)
  #   # print("________________________________________________________________________________________________________________")

  # write log
  test_log_file = misc_utils.test_log_file_dir(mix_snr)
  misc_utils.print_log("write log\n", str(test_log_file), no_prt=True)
  for eval_ans in eval_ans_list:
    msg = ""
    msg += eval_ans.clean_wav_name+"_"+eval_ans.noise_wav_name + (" |  PESQi: %.2f >>>\n" % (eval_ans.pesq_enhanced-eval_ans.pesq_noisy))
    msg += ("       pesq: %.2f -> %.2f | stoi: %.2f -> %.2f | sdr: %.2f -> %.2f\n\n" % (eval_ans.pesq_noisy, eval_ans.pesq_enhanced,
                                                                                        eval_ans.stoi_noisy, eval_ans.stoi_enhanced,
                                                                                        eval_ans.sdr_noisy, eval_ans.sdr_enhanced))
    misc_utils.print_log(msg, str(test_log_file), no_prt=True, no_time=True)

  clean_wavs_lst = [eval_ans_.clean_wav_name for eval_ans_ in eval_ans_list]
  noise_wavs_lst = [eval_ans_.noise_wav_name for eval_ans_ in eval_ans_list]
  pesq_noisy_vec = np.array([eval_ans_.pesq_noisy for eval_ans_ in eval_ans_list], dtype=np.float32)
  stoi_noisy_vec = np.array([eval_ans_.stoi_noisy for eval_ans_ in eval_ans_list], dtype=np.float32)
  sdr_noisy_vec = np.array([eval_ans_.sdr_noisy for eval_ans_ in eval_ans_list], dtype=np.float32)
  ssnr_noisy_vec = np.array([eval_ans_.ssnr_noisy for eval_ans_ in eval_ans_list], dtype=np.float32)
  pesq_enhanced_vec = np.array([eval_ans_.pesq_enhanced for eval_ans_ in eval_ans_list], dtype=np.float32)
  stoi_enhanced_vec = np.array([eval_ans_.stoi_enhanced for eval_ans_ in eval_ans_list], dtype=np.float32)
  sdr_enhanced_vec = np.array([eval_ans_.sdr_enhanced for eval_ans_ in eval_ans_list], dtype=np.float32)
  ssnr_enhanced_vec = np.array([eval_ans_.ssnr_enhanced for eval_ans_ in eval_ans_list], dtype=np.float32)
  mag_v_range_reMAE_vec = np.stack([eval_ans_.mag_v_range_reMAE for eval_ans_ in eval_ans_list], axis=0)
  mag_v_range_reMAE_mean = np.mean(mag_v_range_reMAE_vec, axis=0)
  mag_v_range_reMAE_var = np.var(mag_v_range_reMAE_vec, axis=0)

  testAns_dict = {
    'clean_wav_lst': clean_wavs_lst,
    'noise_wav_lst': noise_wavs_lst,
    'pesq_noisy_lst': list(pesq_noisy_vec),
    'stoi_noisy_lst': list(stoi_noisy_vec),
    'sdr_noisy_lst': list(sdr_noisy_vec),
    'ssnr_noisy_lst': list(ssnr_noisy_vec),
    'pesq_enhanced_lst': list(pesq_enhanced_vec),
    'stoi_enhanced_lst': list(stoi_enhanced_vec),
    'sdr_enhanced_lst': list(sdr_enhanced_vec),
    'ssnr_enhanced_lst': list(ssnr_enhanced_vec),
  }
  test_json_f = misc_utils.test_json_file_dir(mix_snr).open('w')
  json.dump(testAns_dict, test_json_f, cls=JSONEncoder)
  test_json_f.close()

  testAns = TestsetEvalAns(pesq_noisy_mean=np.mean(pesq_noisy_vec), pesq_noisy_var=np.var(pesq_noisy_vec),
                           stoi_noisy_mean=np.mean(stoi_noisy_vec), stoi_noisy_var=np.var(stoi_noisy_vec),
                           sdr_noisy_mean=np.mean(sdr_noisy_vec), sdr_noisy_var=np.var(sdr_noisy_vec),
                           ssnr_noisy_mean=np.mean(ssnr_noisy_vec), ssnr_noisy_var=np.var(ssnr_noisy_vec),
                           pesq_enhanced_mean=np.mean(pesq_enhanced_vec), pesq_enhanced_var=np.var(pesq_enhanced_vec),
                           stoi_enhanced_mean=np.mean(stoi_enhanced_vec), stoi_enhanced_var=np.var(stoi_enhanced_vec),
                           sdr_enhanced_mean=np.mean(sdr_enhanced_vec), sdr_enhanced_var=np.var(sdr_enhanced_vec),
                           ssnr_enhanced_mean=np.mean(ssnr_enhanced_vec), ssnr_enhanced_var=np.var(ssnr_enhanced_vec),
                           mag_v_range_reMAE_mean=mag_v_range_reMAE_mean, mag_v_range_reMAE_var=mag_v_range_reMAE_var)

  misc_utils.print_log("SNR(%d) test over.\n\n" % mix_snr, test_log_file)
  # misc_utils.print_log(str(testAns)+"\n", test_log_file)
  msg = ("SNR(%d) test result >\n"
         " pesq: %.3f ± %.3f -> %.3f ± %.3f\n"
         " stoi: %.3f ± %.3f -> %.3f ± %.3f\n"
         " sdr: %.3f ± %.3f -> %.3f ± %.3f\n"
         " ssnr: %.3f ± %.3f -> %.3f ± %.3f\n"
         " magVRangeMse_mean %s\n"
         " magVRangeMse_var  %s\n" % (
             mix_snr,
             testAns.pesq_noisy_mean, testAns.pesq_noisy_var, testAns.pesq_enhanced_mean, testAns.pesq_enhanced_var,
             testAns.stoi_noisy_mean, testAns.stoi_noisy_var, testAns.stoi_enhanced_mean, testAns.stoi_enhanced_var,
             testAns.sdr_noisy_mean, testAns.sdr_noisy_var, testAns.sdr_enhanced_mean, testAns.sdr_enhanced_var,
             testAns.ssnr_noisy_mean, testAns.ssnr_noisy_var, testAns.ssnr_enhanced_mean, testAns.ssnr_enhanced_var,
             list(np.around(testAns.mag_v_range_reMAE_mean, decimals=5)), list(np.around(testAns.mag_v_range_reMAE_var, decimals=5))))
  misc_utils.print_log(msg, test_log_file, no_time=True)
  return testAns, msg


def eval_testSet_by_meta(mix_SNR, save_test_records=False):
  """
  save_test_records: if save test answer (clean, mixed, enhanced)
  """

  set_root = misc_utils.datasets_dir().joinpath(PARAM.test_name) # "/xx/$datasets_name/train"
  meta_dir = set_root.joinpath(PARAM.test_name+".meta")
  test_log_file = str(misc_utils.test_log_file_dir(mix_SNR))
  misc_utils.print_log("Using meta '%s'\n" % str(meta_dir), test_log_file)

  metaf = meta_dir.open("r")
  meta_list = list(metaf.readlines())
  meta_list.sort()
  meta_list = [meta.strip().split("|") for meta in meta_list] # [ (clean, noise), ...]

  if not save_test_records:
    test_records_save_dir = None
  else:
    _dir = misc_utils.test_records_save_dir()
    test_records_save_dir = str(_dir)
    if _dir.exists():
      import shutil
      shutil.rmtree(str(_dir))
    _dir.mkdir()
  eval_testSet_by_list(meta_list, mix_SNR, test_records_save_dir)


def main():
  eval_testSet_by_meta(-5)
  eval_testSet_by_meta(0, True)
  eval_testSet_by_meta(5)
  eval_testSet_by_meta(10)
  eval_testSet_by_meta(15)

if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])

  if len(sys.argv) > 1:
    test_processor = int(sys.argv[1])
  main()
  """
  run cmd:
  `OMP_NUM_THREADS=1 python -m xx._3_eval_metrics 3`
  """
