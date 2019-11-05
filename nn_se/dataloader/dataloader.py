import tensorflow as tf
import collections
import os
from pathlib import Path

from ..FLAGS import PARAM
from ..utils import misc_utils


class DataSetsOutputs(
    collections.namedtuple("DataSetOutputs",
                           ("initializer", "clean", "mixed"))):
  pass


def parse_func(record_proto):
  wav_len = int(PARAM.sampling_rate*PARAM.train_val_wav_seconds)
  features = {
      'clean': tf.io.FixedLenFeature([wav_len], tf.float32),
      'mixed': tf.io.FixedLenFeature([wav_len], tf.float32)
  }
  record = tf.io.parse_single_example(record_proto, features=features)
  return record['clean'], record['mixed']


def get_batch_inputs_from_dataset(sub_dataset_name):
  """
  dataset_name: PARAM.train_name, PARAM.val_name, PARAM.test_name
  """
  tfrecords_list = misc_utils.datasets_dir().joinpath(sub_dataset_name, "tfrecords", "*.tfrecords")
  files = tf.data.Dataset.list_files(str(tfrecords_list))
  # files = files.take(FLAGS.PARAM.MAX_TFRECORD_FILES_USED)
  if PARAM.shuffle_records:
    files = files.shuffle(PARAM.tfrecords_num_pre_set)
  if not PARAM.shuffle_records:
    dataset = files.interleave(tf.data.TFRecordDataset,
                               cycle_length=1,
                               block_length=PARAM.batch_size,
                               # num_parallel_calls=1,
                               )
  else:  # shuffle
    dataset = files.interleave(tf.data.TFRecordDataset,
                               cycle_length=8,
                               block_length=PARAM.batch_size//8,
                               num_parallel_calls=PARAM.n_processor_tfdata,
                               )
  if PARAM.shuffle_records:
    dataset = dataset.shuffle(PARAM.batch_size*10)

  dataset = dataset.map(parse_func, num_parallel_calls=PARAM.n_processor_tfdata)
  dataset = dataset.batch(batch_size=PARAM.batch_size, drop_remainder=True)
  # dataset = dataset.prefetch(buffer_size=PARAM.batch_size)
  dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
  clean, mixed = dataset_iter.get_next()
  return DataSetsOutputs(initializer=dataset_iter.initializer,
                         clean=clean,
                         mixed=mixed)
