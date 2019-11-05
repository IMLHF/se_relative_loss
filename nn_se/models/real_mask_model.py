import tensorflow as tf

from .modules import Module
from ..FLAGS import PARAM
from ..utils import losses
from .modules import RealVariables


class CNN_RNN_FC_REAL_MASK_MODEL(Module):
  def __init__(self,
               mode,
               variables: RealVariables,
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    super(CNN_RNN_FC_REAL_MASK_MODEL, self).__init__(
        mode,
        variables,
        mixed_wav_batch,
        clean_wav_batch,
        noise_wav_batch)

  def forward(self, mixed_wav_batch):
    outputs = self.real_networks_forward(mixed_wav_batch)
    est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch = outputs
    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch

  def get_loss(self, forward_outputs):
    est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch = forward_outputs

    # region losses

    ## assist loss
    f, b = tf.split(self.blstm_outputs[-1], 2, axis=-1) # f,b: [batch, frame, RNN_SIZE]
    self.real_net_last_blstm_fb_orthogonal = tf.reduce_sum(tf.reduce_mean(tf.square(losses.vec_dot_mul(f, b)), -1))

    ## frequency domain loss
    self.real_net_mag_mse = losses.batch_time_fea_real_mse(est_clean_mag_batch, self.clean_mag_batch)
    self.real_net_reMagMse = losses.batch_real_relativeMSE(est_clean_mag_batch, self.clean_mag_batch, PARAM.relative_loss_AFD)
    self.real_net_spec_mse = losses.batch_time_fea_complex_mse(est_clean_spec_batch, self.clean_spec_batch)
    self.real_net_reSpecMse = losses.batch_complex_relativeMSE(est_clean_spec_batch, self.clean_spec_batch, PARAM.relative_loss_AFD)
    self.real_net_specTCosSimV1 = losses.batch_complexspec_timeaxis_cos_sim_V1(est_clean_spec_batch, self.clean_spec_batch) # *0.167
    self.real_net_specFCosSimV1 = losses.batch_complexspec_frequencyaxis_cos_sim_V1(est_clean_spec_batch, self.clean_spec_batch) # *0.167
    self.real_net_specTFCosSimV1 = losses.batch_complexspec_TF_cos_sim_V1(est_clean_spec_batch, self.clean_spec_batch) # *0.167

    ## time domain loss
    self.real_net_wav_L1 = losses.batch_wav_L1_loss(est_clean_wav_batch, self.clean_wav_batch)*10.0
    self.real_net_wav_L2 = losses.batch_wav_L2_loss(est_clean_wav_batch, self.clean_wav_batch)*100.0
    self.real_net_reWavL2 = losses.batch_wav_relativeMSE(est_clean_wav_batch, self.clean_wav_batch, PARAM.relative_loss_AFD)
    self.real_net_sdrV1 = losses.batch_sdrV1_loss(est_clean_wav_batch, self.clean_wav_batch)
    self.real_net_sdrV2 = losses.batch_sdrV2_loss(est_clean_wav_batch, self.clean_wav_batch)
    self.real_net_sdrV3 = losses.batch_sdrV3_loss(est_clean_wav_batch, self.clean_wav_batch, PARAM.sdrv3_bias) # *0.167
    if PARAM.sdrv3_bias:
      assert PARAM.sdrv3_bias > 0.0, 'sdrv3_bias is constrained larger than zero.'
      self.real_net_sdrV3 *= 1.0 + 60 * PARAM.sdrv3_bias
    self.real_net_cosSimV1 = losses.batch_cosSimV1_loss(est_clean_wav_batch, self.clean_wav_batch) # *0.167
    self.real_net_cosSimV1WT10 = self.real_net_cosSimV1*0.167 # loss weighted to 10 level
    self.real_net_cosSimV2 = losses.batch_cosSimV2_loss(est_clean_wav_batch, self.clean_wav_batch) # *0.334
    self.real_net_stSDRV3 = losses.batch_short_time_sdrV3_loss(est_clean_wav_batch, self.clean_wav_batch,
                                                               PARAM.st_frame_length_for_loss,
                                                               PARAM.st_frame_step_for_loss)
    # engregion losses

    loss = 0
    loss_names = PARAM.loss_name

    for i, name in enumerate(loss_names):
      name = name.replace('comp_net_', 'real_net_')
      loss_t = {
        'real_net_mag_mse': self.real_net_mag_mse,
        'real_net_reMagMse': self.real_net_reMagMse,
        'real_net_spec_mse': self.real_net_spec_mse,
        'real_net_reSpecMse': self.real_net_reSpecMse,
        'real_net_wav_L1': self.real_net_wav_L1,
        'real_net_wav_L2': self.real_net_wav_L2,
        'real_net_reWavL2': self.real_net_reWavL2,
        'real_net_sdrV1': self.real_net_sdrV1,
        'real_net_sdrV2': self.real_net_sdrV2,
        'real_net_sdrV3': self.real_net_sdrV3,
        'real_net_cosSimV1': self.real_net_cosSimV1,
        'real_net_cosSimV1WT10': self.real_net_cosSimV1WT10,
        'real_net_cosSimV2': self.real_net_cosSimV2,
        'real_net_specTCosSimV1': self.real_net_specTCosSimV1,
        'real_net_specFCosSimV1': self.real_net_specFCosSimV1,
        'real_net_specTFCosSimV1': self.real_net_specTFCosSimV1,
        'real_net_stSDRV3': self.real_net_stSDRV3,
        'real_net_last_blstm_fb_orthogonal': self.real_net_last_blstm_fb_orthogonal,
      }[name]
      if len(PARAM.loss_weight) > 0:
        loss_t *= PARAM.loss_weight[i]
      loss += loss_t
    return loss
