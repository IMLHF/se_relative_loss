import tensorflow as tf
import abc
import collections
from typing import Union

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils


class RealVariables(object):
  """
  Real Value NN Variables
  """
  def __init__(self):
    with tf.compat.v1.variable_scope("compat.v1.var", reuse=tf.compat.v1.AUTO_REUSE):
      self._global_step = tf.compat.v1.get_variable('global_step', dtype=tf.int32,
                                                    initializer=tf.constant(1), trainable=False)
      self._lr = tf.compat.v1.get_variable('lr', dtype=tf.float32, trainable=False,
                                           initializer=tf.constant(PARAM.learning_rate))

    # CNN
    self.conv2d_layers = []
    if PARAM.no_cnn:
      pass
    else:
      conv2d_1 = tf.keras.layers.Conv2D(16, [5,5], padding="same", name='se_net/conv2_1') # -> [batch, time, fft_dot, 8]
      conv2d_2 = tf.keras.layers.Conv2D(32, [5,5], dilation_rate=[2,2], padding="same", name='se_net/conv2_2') # -> [batch, t, f, 16]
      conv2d_3 = tf.keras.layers.Conv2D(16, [5,5], dilation_rate=[4,4], padding="same", name='se_net/conv2_3') # -> [batch, t, f, 8]
      conv2d_4 = tf.keras.layers.Conv2D(1, [5,5], padding="same", name='se_net/conv2_4') # -> [batch, t, f, 1]
      self.conv2d_layers = [conv2d_1, conv2d_2, conv2d_3, conv2d_4]

    # BLSTM
    self.N_RNN_CELL = PARAM.rnn_units
    self.blstm_layers = []
    for i in range(1, PARAM.blstm_layers+1):
      forward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, implementation=PARAM.rlstmCell_implementation,
                                          return_sequences=True, name='fwlstm_%d' % i)
      backward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, implementation=PARAM.rlstmCell_implementation,
                                           return_sequences=True, name='bwlstm_%d' % i, go_backwards=True)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
                                            merge_mode='concat', name='se_net/blstm_%d' % i)
      self.blstm_layers.append(blstm)
    # self.blstm_layers = []
    # if PARAM.blstm_layers > 0:
    #   forward_lstm = tf.keras.layers.RNN(
    #       [tf.keras.layers.LSTMCell(
    #           self.N_RNN_CELL, dropout=0.2, name="lstm_%d" % i) for i in range(PARAM.blstm_layers)],
    #       return_sequences=True, name="fwlstm")
    #   backward_lstm = tf.keras.layers.RNN(
    #       [tf.keras.layers.LSTMCell(
    #           self.N_RNN_CELL, dropout=0.2, name="lstm_%d" % i) for i in range(PARAM.blstm_layers)],
    #       return_sequences=True, name="bwlstm", go_backwards=True)
    #   self.blstm_layers.append(tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
    #                                                          merge_mode='concat', name='blstm'))

    #LSTM
    self.lstm_layers = []
    for i in range(1, PARAM.lstm_layers+1):
      lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                  return_sequences=True, implementation=PARAM.rlstmCell_implementation,
                                  name='se_net/lstm_%d' % i)
      self.lstm_layers.append(lstm)

    # FC
    self.out_fc = tf.keras.layers.Dense(PARAM.fft_dot, name='se_net/out_fc')


class Module(object):
  """
  speech enhancement base.
  Discriminate spec and mag:
    spec: spectrum, complex value.
    mag: magnitude, real value.
  """
  def __init__(self,
               mode,
               variables: Union[RealVariables],
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    del noise_wav_batch
    self.mixed_wav_batch = mixed_wav_batch

    self.variables = variables
    self.mode = mode

    # global_step, lr, vars
    self._global_step = self.variables._global_step
    self._lr = self.variables._lr
    self.save_variables = [self.global_step, self._lr]

    # for lr halving
    self.new_lr = tf.compat.v1.placeholder(tf.float32, name='new_lr')
    self.assign_lr = tf.compat.v1.assign(self._lr, self.new_lr)

    # for lr warmup
    if PARAM.use_lr_warmup:
      self._lr = misc_utils.noam_scheme(self._lr, self.global_step, warmup_steps=PARAM.warmup_steps)


    # nn forward
    forward_outputs = self.forward(mixed_wav_batch)
    self._est_clean_wav_batch = forward_outputs[-1]
    self._set_clean_spec_batch = forward_outputs[-2]
    self._est_clean_mag_batch = forward_outputs[-3]

    trainable_variables = tf.compat.v1.trainable_variables()
    self.save_variables.extend([var for var in trainable_variables])
    self.saver = tf.compat.v1.train.Saver(self.save_variables, max_to_keep=PARAM.max_keep_ckpt, save_relative_paths=True)

    # other calculate node
    self._calc_mag_ph = tf.compat.v1.placeholder(tf.float32, [None, None], name='calc_mag_ph')
    self._calc_mag = tf.abs(misc_utils.tf_batch_stft(self._calc_mag_ph, PARAM.frame_length, PARAM.frame_step))

    if mode == PARAM.MODEL_INFER_KEY:
      return

    # labels
    self.clean_wav_batch = clean_wav_batch
    self.clean_spec_batch = misc_utils.tf_batch_stft(clean_wav_batch, PARAM.frame_length, PARAM.frame_step) # complex label
    # self.noise_wav_batch = mixed_wav_batch - clean_wav_batch
    # self.noise_spec_batch = misc_utils.tf_batch_stft(self.noise_wav_batch, PARAM.frame_length, PARAM.frame_step)
    # self.nosie_mag_batch = tf.math.abs(self.noise_spec_batch)
    self.clean_mag_batch = tf.math.abs(self.clean_spec_batch) # mag label
    # self.debug_clean = self.clean_mag_batch
    # self.debug_mixed = self.mixed_wav_batch
    self.clean_angle_batch = tf.math.angle(self.clean_spec_batch)
    if PARAM.mask_type == "IRM":
      print("Use IRM")
      pass
    elif PARAM.mask_type == "PSM":
      print("Use PSM")
      self.clean_mag_batch *= tf.math.cos(self.mixed_angle_batch - self.clean_angle_batch)
    else:
      raise ValueError('mask type %s error.' % PARAM.mask_type)

    self._se_loss = self.get_loss(forward_outputs)

    self._loss = self._se_loss

    if mode == PARAM.MODEL_VALIDATE_KEY:
      return

    # optimizer
    # opt = tf.keras.optimizers.Adam(learning_rate=self._lr)
    opt = tf.compat.v1.train.AdamOptimizer(self._lr)
    no_d_params = tf.compat.v1.trainable_variables(scope='se_net*')
    # misc_utils.show_variables(no_d_params)
    gradients = tf.gradients(
      self._se_loss,
      no_d_params,
      colocate_gradients_with_ops=True
    )
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, PARAM.max_gradient_norm)
    self._train_op = opt.apply_gradients(zip(clipped_gradients, no_d_params),
                                         global_step=self.global_step)


  def CNN_RNN_FC(self, mixed_mag_batch, training=False):
    outputs = tf.expand_dims(mixed_mag_batch, -1) # [batch, time, fft_dot, 1]
    _batch_size = tf.shape(outputs)[0]

    # CNN
    for conv2d in self.variables.conv2d_layers:
      outputs = conv2d(outputs)
    if len(self.variables.conv2d_layers) > 0:
      outputs = tf.squeeze(outputs, [-1]) # [batch, time, fft_dot]
      if PARAM.cnn_shortcut == "add":
        outputs = tf.add(outputs, mixed_mag_batch)
      elif PARAM.cnn_shortcut == "multiply":
        outputs = tf.multiply(outputs, mixed_mag_batch)


    # print(outputs.shape.as_list())
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])

    # BLSTM
    self.blstm_outputs = []
    for blstm in self.variables.blstm_layers:
      outputs = blstm(outputs, training=training)
      self.blstm_outputs.append(outputs)

    # LSTM
    for lstm in self.variables.lstm_layers:
      outputs = lstm(outputs, training=training)

    # FC
    if len(self.variables.blstm_layers) > 0 and len(self.variables.lstm_layers) <= 0:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL*2])
    else:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL])
    outputs = self.variables.out_fc(outputs)
    if PARAM.ReLU_outputs:
      outputs = tf.nn.relu(outputs)
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])
    return outputs


  def real_networks_forward(self, mixed_wav_batch):
    mixed_spec_batch = misc_utils.tf_batch_stft(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step)
    mixed_mag_batch = tf.math.abs(mixed_spec_batch)
    self.mixed_angle_batch = tf.math.angle(mixed_spec_batch)
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    mask = self.CNN_RNN_FC(mixed_mag_batch, training)

    if PARAM.net_out == 'mask':
      est_clean_mag_batch = tf.multiply(mask, mixed_mag_batch) # mag estimated
    elif PARAM.net_out == 'spectrum':
      est_clean_mag_batch = mask
    else:
      raise ValueError('net_out %s type error.' % PARAM.net_out)

    est_clean_spec_batch = tf.complex(est_clean_mag_batch, 0.0) * tf.exp(tf.complex(0.0, self.mixed_angle_batch)) # complex
    _mixed_wav_len = tf.shape(mixed_wav_batch)[-1]
    _est_clean_wav_batch = misc_utils.tf_batch_istft(est_clean_spec_batch, PARAM.frame_length, PARAM.frame_step)
    est_clean_wav_batch = tf.slice(_est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


  @abc.abstractmethod
  def forward(self, mixed_wav_batch):
    """
    Returns:
      forward_outputs: pass to get_loss
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "forward not implement, code: 939iddfoollvoae")


  @abc.abstractmethod
  def get_loss(self, forward_outputs):
    """
    Returns:
      loss
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "get_loss not implement, code: 67hjrethfd")


  @abc.abstractmethod
  def get_discriminator_loss(self, forward_outputs):
    """
    Returns:
      loss
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "get_discriminator_loss not implement, code: qyhhtwgrff")


  def change_lr(self, sess, new_lr):
    sess.run(self.assign_lr, feed_dict={self.new_lr:new_lr})

  @property
  def mixed_wav_batch_in(self):
    return self.mixed_wav_batch

  @property
  def global_step(self):
    return self._global_step

  @property
  def train_op(self):
    return self._train_op

  @property
  def loss(self):
    return self._loss

  @property
  def lr(self):
    return self._lr

  @property
  def est_clean_wav_batch(self):
    return self._est_clean_wav_batch

  @property
  def est_clean_spec_batch(self):
    return self._est_clean_spec_batch

  @property
  def est_clean_mag_batch(self):
    return self._est_clean_mag_batch

  @property
  def calc_mag(self):
    return self._calc_mag

  @property
  def calc_mag_ph(self):
    return self._calc_mag_ph
