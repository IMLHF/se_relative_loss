class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_VALIDATE_KEY = 'val'
  MODEL_INFER_KEY = 'infer'

  # dataset name
  train_name="train"
  validation_name="validation"
  test_name="test"

  def config_name(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  VISIBLE_GPU = "0"
  root_dir = '/home/lhf/worklhf/se_relative_loss_paper_exp/'
  datasets_name = 'vctk_musan_datasets_8k'
  '''
  # dir to store log, model and results files:
  $root_dir/$datasets_name: datasets dir
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/test_records: test results
  $root_dir/exp/$config_name/hparams
  '''

  min_TF_version = "1.14.0"

  # _1_preprocess param
  n_train_set_records = 72000
  n_val_set_records = 4800
  n_test_set_records = 4800
  train_val_snr = [-5, 15]
  train_val_wav_seconds = 3.0

  sampling_rate = 8000

  n_processor_gen_tfrecords = 16
  tfrecords_num_pre_set = 160
  shuffle_records = True
  batch_size = 64
  n_processor_tfdata = 4

  """
  @param model_name:
  CNN_RNN_FC_REAL_MASK_MODEL, CCNN_CRNN_CFC_COMPLEX_MASK_MODEL,
  RC_HYBIRD_MODEL, RR_HYBIRD_MODEL
  """
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"

  """
  @param loss_name:
  real_net_mag_mse, real_net_spec_mse, real_net_wav_L1, real_net_wav_L2,
  real_net_reMagMse, real_net_HardReMagMse,  real_net_reSpecMse, real_net_reWavL2,
  real_net_sdrV1, real_net_sdrV2, real_net_sdrV3, real_net_stSDRV3, real_net_cosSimV1, real_net_cosSimV1WT10, real_net_cosSimV2,
  real_net_specTCosSimV1, real_net_specFCosSimV1, real_net_specTFCosSimV1,
  real_net_last_blstm_fb_orthogonal,
  """
  relative_loss_AFD = 50.0
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 128
  sdrv3_bias = None # float, a bias will be added before vector dot multiply.
  loss_name = ["real_net_mag_mse"]
  loss_weight = []
  net_out = 'mask' # mask | spectrum | mat_mask
  frame_length = 256
  frame_step = 64
  no_cnn = False
  blstm_layers = 2
  lstm_layers = 0
  rnn_units = 256
  rlstmCell_implementation = 1
  fft_dot = 129
  max_keep_ckpt = 30
  learning_rate = 0.001
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.45

  s_epoch = 1
  max_epoch = 30
  batches_to_logging = 300

  max_model_abandon_time = 3
  noStop_noAbandon = False
  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 4000. # for (use_lr_warmup == true)
  start_halving_impr = 0.01 # no use for (use_lr_warmup == true)
  lr_halving_rate = 0.7 # no use for (use_lr_warmup == true)

  cnn_shortcut = None # None | "add" | "multiply"

  mask_type = 'IRM' # IRM | PSM
  ReLU_outputs = False

  mag_reMAE_v_range = [0.1, 0.3, 0.5, 0.7, 0.9]

  mnv_mag_feature = False


class p40(BaseConfig):
  n_processor_gen_tfrecords = 56
  n_processor_tfdata = 8
  GPU_PARTION = 0.2
  root_dir = '/home/zhangwenbo5/lihongfeng/se_relative_loss_paper_exp'

class debug(p40):
  blstm_layers = 2
  lstm_layers = 0
  no_cnn = True
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"

# region delete

class WRL_PSM_ReLU_A10(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_reMagMse"]
  relative_loss_AFD = 10.0

class WRL_PSM_ReLU_A50(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_reMagMse"]
  relative_loss_AFD = 50.0

class WRL_PSM_ReLU_A100(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_reMagMse"]
  relative_loss_AFD = 100.0

class WRL_PSM_ReLU_A500(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_reMagMse"]
  relative_loss_AFD = 500.0

class WRL_IRM_ReLU_A50(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_reMagMse"]
  relative_loss_AFD = 50.0

class WRL_IRM_ReLU_A100(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_reMagMse"]
  relative_loss_AFD = 100.0

class WRL_IRM_ReLU_A500(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_reMagMse"]
  relative_loss_AFD = 500.0

# endregion

class MSE_IRM_ReLU(p40): # done p40
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_mag_mse"]

class MSE_IRM_ReLU_maskRawMVN(p40): # done p40 /MSE_IRM_ReLU_MVN
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_mag_mse"]
  mnv_mag_feature = True

class MSE_PSM_ReLU(p40): # done p40
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_mag_mse"]

class MSE_PSM_ReLU_maskRawMVN(p40): # done p40
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_mag_mse"]
  mnv_mag_feature = True

class HRL_IRM_ReLU_AINF(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 1e18

class HRL_PSM_ReLU_AINF(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 1e18

class HRL_IRM_ReLU_A05(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 5.0

class HRL_IRM_ReLU_A10(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 10.0

class HRL_IRM_ReLU_A50(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 50.0

class HRL_IRM_ReLU_A100(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 100.0

class HRL_IRM_ReLU_A500(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 500.0

class HRL_PSM_ReLU_A05(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 5.0

class HRL_PSM_ReLU_A10(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 10.0

class HRL_PSM_ReLU_A50(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 50.0

class HRL_PSM_ReLU_A100(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 100.0

class HRL_PSM_ReLU_A500(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = True
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 500.0

class MSE_IRM_Real(p40): # done p40
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = False
  loss_name = ["real_net_mag_mse"]

class MSE_IRM_Real_MVN(p40): # run p40
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = False
  loss_name = ["real_net_mag_mse"]
  mnv_mag_feature = True

class MSE_IRM_Real_matMask(p40): # done p40
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = False
  net_out = 'mat_mask'
  GPU_PARTION = 0.3
  loss_name = ["real_net_mag_mse"]

class MSE_PSM_Real(p40): # done p40
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = False
  loss_name = ["real_net_mag_mse"]

class MSE_PSM_Real_MVN(p40): # run p40
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = False
  loss_name = ["real_net_mag_mse"]
  mnv_mag_feature = True

class HRL_IRM_Real_A05(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 5.0

class HRL_IRM_Real_A10(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 10.0

class HRL_IRM_Real_A50(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 50.0

class HRL_IRM_Real_A100(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 100.0

class HRL_IRM_Real_A500(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 500.0

class HRL_PSM_Real_A05(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 5.0

class HRL_PSM_Real_A10(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 10.0

class HRL_PSM_Real_A50(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 50.0

class HRL_PSM_Real_A100(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 100.0

class HRL_PSM_Real_A500(p40): # done
  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "PSM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 500.0

class HRL_IRM_Real_A10_16k(p40):
  datasets_name = 'vctk_musan_datasets_16k'
  sampling_rate = 16000
  frame_length = 512
  frame_step = 128
  blstm_layers = 2
  lstm_layers = 0
  rnn_units = 512
  rlstmCell_implementation = 1
  fft_dot = 257

  no_cnn = True
  blstm_layers = 2
  lstm_layers = 0
  model_name = "CNN_RNN_FC_REAL_MASK_MODEL"
  mask_type = "IRM"
  ReLU_outputs = False
  loss_name = ["real_net_HardReMagMse"]
  relative_loss_AFD = 10.0


PARAM = HRL_IRM_Real_A10_16k

# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python -m xxx._2_train
