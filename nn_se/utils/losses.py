import tensorflow as tf

def vec_dot_mul(y1, y2):
  dot_mul = tf.reduce_sum(tf.multiply(y1, y2), -1)
  return dot_mul

def vec_normal(y):
  normal_ = tf.sqrt(tf.reduce_sum(tf.square(y), -1))
  return normal_

def batch_time_fea_real_mse(y1, y2):
  """
  y1: real, [batch, time, fft_dot]
  y2: real, [batch, time, fft_dot]
  """
  loss = tf.square(y1-y2)
  loss = tf.reduce_mean(tf.reduce_sum(loss, 0))
  return loss

def batch_time_fea_complex_mse(y1, y2):
  """
  y1: complex, [batch, time, fft_dot]
  y2: conplex, [batch, time, fft_dot]
  """
  y1_real = tf.math.real(y1)
  y1_imag = tf.math.imag(y1)
  y2_real = tf.math.real(y2)
  y2_imag = tf.math.imag(y2)
  loss_real = batch_time_fea_real_mse(y1_real, y2_real)
  loss_imag = batch_time_fea_real_mse(y1_imag, y2_imag)
  loss = loss_real + loss_imag
  return loss

def batch_real_relativeMSE(y1, y2, axis_fit_degree, index_=2.0):
  # y1, y2 : [batch, time, feature]
  # refer_sum = tf.maximum(tf.abs(y1)+tf.abs(y2),1e-12)
  # small_val_debuff = tf.pow(refer_sum*axis_fit_degree*1.0,-1.0)+1.0-tf.pow(axis_fit_degree*1.0,-1.0)
  # relative_loss = tf.abs(y1-y2)/refer_sum/small_val_debuff
  refer = 1.0/axis_fit_degree+(1.0-1.0/axis_fit_degree)*(tf.abs(y1)+tf.abs(y2))
  relative_loss = tf.abs(y1-y2)/refer
  cost = tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, index_), 0))
  return cost

def batch_complex_relativeMSE(y1, y2, axis_fit_degree, index_=2.0):
  """
  y1: complex, [batch, time, fft_dot]
  y2: conplex, [batch, time, fft_dot]
  """
  y1_real = tf.math.real(y1)
  y1_imag = tf.math.imag(y1)
  y2_real = tf.math.real(y2)
  y2_imag = tf.math.imag(y2)
  loss_real = batch_real_relativeMSE(y1_real, y2_real, axis_fit_degree)
  loss_imag = batch_real_relativeMSE(y1_imag, y2_imag, axis_fit_degree)
  loss = 0.5*loss_real+0.5*loss_imag
  return loss

def batch_wav_L1_loss(y1, y2):
  loss = tf.reduce_mean(tf.reduce_sum(tf.abs(y1-y2), 0))
  return loss

def batch_wav_L2_loss(y1, y2):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(y1-y2), 0))
  return loss

def batch_wav_relativeMSE(y1, y2, AFD):
  loss = batch_real_relativeMSE(y1, y2, AFD)
  return loss

def batch_sdrV1_loss(est, ref):
  loss_s1 = tf.divide(tf.reduce_sum(tf.multiply(est, est), -1),
                      tf.square(tf.reduce_sum(tf.multiply(est, ref), -1)))
  loss = tf.reduce_sum(loss_s1)
  return loss

def batch_sdrV2_loss(est, ref): # 1/cos^2
  loss_s1 = tf.divide(tf.multiply(tf.reduce_sum(tf.multiply(est, est), -1),
                                  tf.reduce_sum(tf.multiply(ref, ref), -1)),
                      tf.square(tf.reduce_sum(tf.multiply(est, ref), -1)))
  loss = tf.reduce_sum(loss_s1)
  return loss

def batch_sdrV3_loss(est, ref, bias=None): # sin^2 (1-cos^2)
  if bias:
    est += bias
    ref += bias
  loss_s1 = 1.0 - tf.divide(tf.square(tf.reduce_sum(tf.multiply(est, ref), -1)),
                            tf.multiply(tf.reduce_sum(tf.multiply(est, est), -1),
                                        tf.reduce_sum(tf.multiply(ref, ref), -1)))

  loss = tf.reduce_sum(loss_s1)
  return loss

def batch_short_time_sdrV3_loss(est, ref, st_frame_length, st_frame_step): # sin^2 (1-cos^2)
  st_est = tf.signal.frame(est, frame_length=st_frame_length, # [batch, frame, st_wav]
                           frame_step=st_frame_step, pad_end=True)
  st_ref = tf.signal.frame(ref, frame_length=st_frame_length,
                           frame_step=st_frame_step, pad_end=True)
  loss_s1 = 1.0 - tf.divide(tf.square(vec_dot_mul(st_est, st_ref)), # [batch, frame]
                            tf.multiply(vec_dot_mul(st_est, st_est),
                                        vec_dot_mul(st_ref, st_ref)))

  loss = tf.reduce_sum(tf.reduce_mean(loss_s1, -1))
  return loss

def batch_cosSimV1_loss(est, ref): # 1-cos
  cos_sim = 1.0 - tf.divide(vec_dot_mul(est, ref),
                            tf.multiply(vec_normal(est), vec_normal(ref)))
  loss = tf.reduce_sum(cos_sim, 0)
  return loss

def batch_cosSimV2_loss(est, ref): # (1-cos)^2
  cos_sim = tf.square(1.0 - tf.divide(vec_dot_mul(est, ref),
                                      tf.multiply(vec_normal(est), vec_normal(ref))))
  loss = tf.reduce_sum(cos_sim, 0)
  return loss

def batch_realspec_timeaxis_cos_sim_V1(est, ref):
  # est, ref : [batch, time, fre]
  est = tf.transpose(est, [0,2,1]) # [batch, f, t]
  ref = tf.transpose(ref, [0,2,1])
  cos_sim_t = 1.0 - tf.divide(vec_dot_mul(est, ref),
                              tf.maximum(tf.multiply(vec_normal(est), vec_normal(ref)), 1e-12))
  loss = tf.reduce_sum(tf.reduce_mean(cos_sim_t, -1))
  return loss

def batch_complexspec_timeaxis_cos_sim_V1(est, ref):
  est_real = tf.math.real(est)
  est_imag = tf.math.imag(est)
  ref_real = tf.math.real(ref)
  ref_imag = tf.math.imag(ref)
  real_t_cossim = batch_realspec_timeaxis_cos_sim_V1(est_real, ref_real)
  imag_t_cossim = batch_realspec_timeaxis_cos_sim_V1(est_imag, ref_imag)
  loss = 0.5*real_t_cossim + 0.5*imag_t_cossim
  return loss

def batch_realspec_frequencyaxis_cos_sim_V1(est, ref):
  # est, ref : [batch, time, fre]
  cos_sim_t = 1.0 - tf.divide(vec_dot_mul(est, ref),
                              tf.multiply(vec_normal(est), vec_normal(ref)))
  loss= tf.reduce_sum(tf.reduce_mean(cos_sim_t, -1))
  return loss

def batch_complexspec_frequencyaxis_cos_sim_V1(est, ref):
  est_real = tf.math.real(est)
  est_imag = tf.math.imag(est)
  ref_real = tf.math.real(ref)
  ref_imag = tf.math.imag(ref)
  real_f_cossim = batch_realspec_frequencyaxis_cos_sim_V1(est_real, ref_real)
  imag_f_cossim = batch_realspec_frequencyaxis_cos_sim_V1(est_imag, ref_imag)
  loss = 0.5*real_f_cossim + 0.5*imag_f_cossim
  return loss

def batch_realspec_TF_cos_sim_V1(est, ref):
  loss_t_axis_cossim = batch_realspec_timeaxis_cos_sim_V1(est, ref)
  loss_f_axis_cossim = batch_realspec_frequencyaxis_cos_sim_V1(est, ref)
  loss = 0.5*loss_t_axis_cossim + 0.5*loss_f_axis_cossim
  return loss

def batch_complexspec_TF_cos_sim_V1(est, ref):
  est_real = tf.math.real(est)
  est_imag = tf.math.imag(est)
  ref_real = tf.math.real(ref)
  ref_imag = tf.math.imag(ref)
  real_tf_cossim = batch_realspec_TF_cos_sim_V1(est_real, ref_real)
  imag_tf_cossim = batch_realspec_TF_cos_sim_V1(est_imag, ref_imag)
  loss = 0.5*real_tf_cossim + 0.5*imag_tf_cossim
  return loss

def batch_wav_cos_Lp_loss(y1, y2, p):
  pass
