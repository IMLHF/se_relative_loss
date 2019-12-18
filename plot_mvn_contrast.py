import matplotlib.pyplot as plt

def plot_compare(metrics, y0, y1, y2, y3, y4,
                 exp1_name, exp2_name, exp3_name, exp4_name):
  x=["-5db","0db","5db","10db","15db"]
  plt.figure(figsize=(4,3))
  plt.plot(x, y0, label='Noisy', linewidth=1, color='black', marker='s',
           markerfacecolor='black', markersize=8)
  plt.plot(x, y1, label=exp1_name, linewidth=1, color='blue', marker='o',
           markerfacecolor='blue', markersize=8)
  plt.plot(x, y2, label=exp2_name, linewidth=1, color='red', marker='>',
           markerfacecolor='red', markersize=8)
  plt.plot(x, y3, label=exp3_name, linewidth=1, color='green', marker='*',
           markerfacecolor='green', markersize=8)
  plt.plot(x, y4, label=exp4_name, linewidth=1, color='orange', marker='+',
           markerfacecolor='orange', markersize=8)
  # fig.autofmt_xdate()
  plt.xlabel('SNR Condition')
  plt.ylabel('%s Scores' % metrics)
  plt.legend()
  left = 0.14
  if metrics == 'SSNR':
    left = 0.17
  if metrics == 'STOI':
    left = 0.16
  plt.subplots_adjust(top=0.99, right=0.99, left=left, bottom=0.15)
  plt.savefig("exp/paper_sample/_mvn_contrast_%s.jpg" % metrics)
  plt.show()
  plt.close()

if __name__ == "__main__":

  mse_irm_real_pesq = [2.411, 2.673, 2.885, 3.068, 3.233]
  mse_irm_real_stoi = [0.687, 0.735, 0.771, 0.801, 0.826]
  mse_irm_real_sdr  = [8.724, 12.277, 15.374, 18.355, 21.365]
  mse_irm_real_ssnr = [4.520, 7.241, 9.802, 12.330, 14.794]

  mse_irm_real_mvn_pesq = [2.259, 2.533, 2.768, 2.982, 3.177]
  mse_irm_real_mvn_stoi = [0.668, 0.723, 0.764, 0.798, 0.826]
  mse_irm_real_mvn_sdr  = [7.020, 10.792, 14.168, 17.421, 20.651]
  mse_irm_real_mvn_ssnr = [3.247, 6.013, 8.746, 11.456, 14.054]

  mse_psm_real_pesq = [2.355, 2.616, 2.835, 3.027, 3.193]
  mse_psm_real_stoi = [0.672, 0.723, 0.761, 0.792, 0.819]
  mse_psm_real_sdr  = [8.958, 12.449, 15.569, 18.458, 21.279]
  mse_psm_real_ssnr = [4.579, 7.161, 9.553, 11.787, 13.801]

  mse_psm_real_mvn_pesq = [2.159, 2.409, 2.624, 2.808, 2.968]
  mse_psm_real_mvn_stoi = [0.630, 0.689, 0.735, 0.769, 0.796]
  mse_psm_real_mvn_sdr  = [3.738, 7.558, 10.863, 13.738, 16.248]
  mse_psm_real_mvn_ssnr = [-1.482, 0.598, 2.723, 4.866, 6.995]

  noisy_pesq = [1.751, 1.991, 2.263, 2.538, 2.814]
  noisy_stoi = [0.597, 0.659, 0.715, 0.764, 0.806]
  noisy_sdr  = [-4.647, 0.175, 5.117, 10.098, 15.092]
  noisy_ssnr = [-3.262, -1.426, 1.274, 5.138, 9.686]

  plot_compare('PESQ', noisy_pesq,
               mse_irm_real_pesq, mse_irm_real_mvn_pesq, mse_psm_real_pesq, mse_psm_real_mvn_pesq,
               'IRM+SMAN', "IRM+MVN", "PSM+SMAN", "PSM+MVN")

  plot_compare('STOI', noisy_stoi,
               mse_irm_real_stoi, mse_irm_real_mvn_stoi, mse_psm_real_stoi, mse_psm_real_mvn_stoi,
               'IRM+SMAN', "IRM+MVN", "PSM+SMAN", "PSM+MVN")

  plot_compare('SDR', noisy_sdr,
               mse_irm_real_sdr, mse_irm_real_mvn_sdr, mse_psm_real_sdr, mse_psm_real_mvn_sdr,
               'IRM+SMAN', "IRM+MVN", "PSM+SMAN", "PSM+MVN")

  plot_compare('SSNR', noisy_ssnr,
               mse_irm_real_ssnr, mse_irm_real_mvn_ssnr, mse_psm_real_ssnr, mse_psm_real_mvn_ssnr,
               'IRM+SMAN', "IRM+MVN", "PSM+SMAN", "PSM+MVN")

