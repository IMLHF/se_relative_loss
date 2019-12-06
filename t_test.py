from nn_se.utils import misc_utils

import sys
import os
import json
import numpy as np
from scipy.stats import norm, t
from scipy import stats

snr_lst = ["-05", "+00", "+05", "+10", "+15"]

# class JSONDncoder(json.JSONDncoder):
#   def default(self, obj):
#     if isinstance(obj, np.integer):
#       return int(obj)
#     elif isinstance(obj, np.floating):
#       return float(obj)
#     elif isinstance(obj, np.ndarray):
#       return obj.tolist()
#     else:
#       return super(JSONDncoder, self).default(obj)

def z_test(exp_X_name, exp_Y_name, snr):
  X_all_dir = os.path.join('exp', exp_X_name, 'log', 'test_snr(%s).json' % snr)
  X_json_f = open(X_all_dir, 'r')
  X_all = json.load(X_json_f)
  X_json_f.close()

  Y_all_dir = os.path.join('exp', exp_Y_name, 'log', 'test_snr(%s).json' % snr)
  Y_json_f = open(Y_all_dir, 'r')
  Y_all = json.load(Y_json_f)
  Y_json_f.close()

  X_pesq_arr = np.array(X_all['pesq_enhanced_lst'])
  X_stoi_arr = np.array(X_all['stoi_enhanced_lst'])
  X_sdr_arr = np.array(X_all['sdr_enhanced_lst'])
  X_ssnr_arr = np.array(X_all['ssnr_enhanced_lst'])
  Y_pesq_arr = np.array(Y_all['pesq_enhanced_lst'])
  Y_stoi_arr = np.array(Y_all['stoi_enhanced_lst'])
  Y_sdr_arr = np.array(Y_all['sdr_enhanced_lst'])
  Y_ssnr_arr = np.array(Y_all['ssnr_enhanced_lst'])

  D_pesq = Y_pesq_arr - X_pesq_arr
  D_stoi = Y_stoi_arr - X_stoi_arr
  D_sdr = Y_sdr_arr - X_sdr_arr
  D_ssnr = Y_ssnr_arr - X_ssnr_arr

  n_sample = 30
  delta_u_pesq = np.mean(D_pesq)
  u0_pesq = 0.0
  mean_D_pesq = np.mean(D_pesq[:n_sample])
  std_D_pesq = np.std(D_pesq[:n_sample])
  t0_pesq = (mean_D_pesq-u0_pesq)/std_D_pesq*np.sqrt(n_sample)
  p_pesq = 1.0-t.cdf(t0_pesq, n_sample-1)
  # t0_pesq, p_pesq = stats.ttest_rel(X_pesq_arr[:30], Y_pesq_arr[:30])
  # if t0_pesq > 0:
  #   p_pesq = 1 - p_pesq / 2
  # else:
  #   p_pesq = p_pesq / 2

  delta_u_stoi = np.mean(D_stoi)
  u0_stoi = 0.0
  mean_D_stoi = np.mean(D_stoi[:n_sample])
  std_D_stoi = np.std(D_stoi[:n_sample])
  t0_stoi = (mean_D_stoi-u0_stoi)/std_D_stoi*np.sqrt(n_sample)
  p_stoi = 1.0-t.cdf(t0_stoi, n_sample-1)

  delta_u_sdr = np.mean(D_sdr)
  u0_sdr = 0.0
  mean_D_sdr = np.mean(D_sdr[:n_sample])
  std_D_sdr = np.std(D_sdr[:n_sample])
  t0_sdr = (mean_D_sdr-u0_sdr)/std_D_sdr*np.sqrt(n_sample)
  p_sdr = 1.0 - t.cdf(t0_sdr, n_sample-1)

  delta_u_ssnr = np.mean(D_ssnr)
  u0_ssnr = 0.0
  mean_D_ssnr = np.mean(D_ssnr[:n_sample])
  std_D_ssnr = np.std(D_ssnr[:n_sample])
  t0_ssnr = (mean_D_ssnr-u0_ssnr)/std_D_ssnr*np.sqrt(n_sample)
  p_ssnr = 1.0 - t.cdf(t0_ssnr, n_sample-1)

  print("SNR(%s): \n"
        "  t0_pesq %.4f, p_pesq %.2e, uy-ux %.3f, u0_pesq %.3f.\n"
        "  t0_stoi %.4f, p_stoi %.2e, uy-ux %.3f, u0_stoi %.3f.\n"
        "  t0_sdr  %.3f, p_sdr  %.2e, uy-ux %.3f, u0_sdr  %.3f.\n"
        "  t0_ssnr %.3f, p_ssnr %.2e, uy-ux %.3f, u0_ssnr %.3f." % (
            snr,
            t0_pesq, p_pesq, delta_u_pesq, u0_pesq,
            t0_stoi, p_stoi, delta_u_stoi, u0_stoi,
            t0_sdr, p_sdr, delta_u_sdr, u0_sdr,
            t0_ssnr, p_ssnr, delta_u_ssnr, u0_ssnr), flush=True)

def main(exp_X_name, exp_Y_name):
  for snr in snr_lst:
    z_test(exp_X_name, exp_Y_name, snr)

if __name__ == "__main__":

  assert len(sys.argv) == 3
  exp_X_name = str(sys.argv[1])
  exp_Y_name = str(sys.argv[2])
  # D = Y-X
  # H0: u_D>=u0, H1: u_D<u0


  main(exp_X_name, exp_Y_name)

  """
  run cmd:
  `OMP_NUM_THREADS=1 python -m xx._3_eval_metrics 3`
  """
