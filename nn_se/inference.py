import tensorflow as tf
import collections
import numpy as np

from .models import model_builder
from .models import modules
from .utils import misc_utils
from .FLAGS import PARAM


class SMG(
    collections.namedtuple("SMG",
                           ("session", "model", "graph"))):
  pass

class EnOut(
    collections.namedtuple("EnOut",
                           ("enhanced_wav", "enhanced_mag", "mask"))):
  pass

def build_SMG(ckpt_name=None, batch_size=None, finalizeG=True):
  g = tf.Graph()
  with g.as_default():
    with tf.name_scope("inputs"):
      mixed_batch = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, None], name='mixed_batch')

    ModelC, VariablesC = model_builder.get_model_class_and_var()

    variables = VariablesC()
    infer_model = ModelC(PARAM.MODEL_INFER_KEY, variables, mixed_batch)
    init = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
    # misc_utils.show_variables(infer_model.save_variables)
    # misc_utils.show_variables(val_model.save_variables)
  g.finalize()

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = PARAM.GPU_RAM_ALLOW_GROWTH
  # config.gpu_options.per_process_gpu_memory_fraction = 0.45
  config.allow_soft_placement = False
  sess = tf.compat.v1.Session(config=config, graph=g)
  sess.run(init)

  if ckpt_name:
    ckpt_dir = tf.train.get_checkpoint_state(str(ckpt_name)).model_checkpoint_path
  else:
    ckpt_dir = tf.train.get_checkpoint_state(str(misc_utils.ckpt_dir())).model_checkpoint_path

  if ckpt_dir:
    # test_log_file = misc_utils.test_log_file_dir()
    # misc_utils.print_log("Restore from " + ckpt_dir + "\n", log_file=str(test_log_file), no_prt=True)
    infer_model.saver.restore(sess, ckpt_dir)

  if finalizeG:
    g.finalize()
  return SMG(session=sess, model=infer_model, graph=g)


def enhance_one_wav(smg: SMG, wav):
  wav_batch = np.array([wav], dtype=np.float32)
  enhanced_wav_batch, enhanced_mag_batch, mask = smg.session.run([smg.model.est_clean_wav_batch,
                                                                  smg.model.est_clean_mag_batch,
                                                                  smg.model.est_mask],
                                                                 feed_dict={smg.model.mixed_wav_batch_in: wav_batch})
  enhanced_wav = enhanced_wav_batch[0]
  enhanced_mag = enhanced_mag_batch[0]
  return EnOut(enhanced_wav=enhanced_wav,
               enhanced_mag=enhanced_mag,
               mask=mask[0])

def enhance_for_test(smg: SMG, wav_mixed, wav_clean):
  mixed_wav_batch = np.array([wav_mixed], dtype=np.float32)
  clean_wav_batch = np.array([wav_clean], dtype=np.float32)
  enhanced_wav_batch, enhanced_mag_batch, clean_mag_batch = smg.session.run(
      [smg.model.est_clean_wav_batch,
       smg.model.est_clean_mag_batch,
       smg.model.calc_mag],
      feed_dict={smg.model.mixed_wav_batch_in: mixed_wav_batch,
                 smg.model.calc_mag_ph: clean_wav_batch})
  enhanced_wav = enhanced_wav_batch[0]
  enhanced_mag = enhanced_mag_batch[0]
  clean_mag = clean_mag_batch[0]
  return enhanced_wav, enhanced_mag, clean_mag
