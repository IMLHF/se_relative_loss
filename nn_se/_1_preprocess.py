import os
from pathlib import Path
import numpy as np
import shutil
import time
import multiprocessing
import tensorflow as tf
from functools import partial
from tqdm import tqdm
import sys

from .FLAGS import PARAM
from .utils import audio
from .utils import misc_utils

"""
Generate metadata and tfrecords.
[($root_dir/$datasets_name) fold structure]
$root_dir/$datasets_name:
  train:
    speech:
    noise:
    speech.list
    noise.list
    train.meta
  validation:
    speech:
    noise:
    speech.list
    noise.list
    validation.meta
  test:
    speech:
    noise:
    speech.list
    noise.list
    test.meta
"""

train_name = PARAM.train_name
validation_name = PARAM.validation_name
test_name = PARAM.test_name

def prepare_train_validation_test_set(sub_dataset_name): # for train/val/test
    # get speech_list and noise_list
    set_root = misc_utils.datasets_dir().joinpath(sub_dataset_name) # "/xx/$datasets_name/train"
    speech_list = list(set_root.joinpath("speech").glob("*/*.wav"))
    noise_list = list(set_root.joinpath("noise").glob("*.wav"))

    speech_list_flie = set_root.joinpath("speech.list").open("w")
    [speech_list_flie.write(str(speech)+"\n") for speech in speech_list]
    speech_list_flie.close()

    noise_list_file = set_root.joinpath("noise.list").open("w")
    [noise_list_file.write(str(noise)+"\n") for noise in noise_list]
    noise_list_file.close()

    # get train.meta or validation.meta or test.meta
    # train.meta/validation.meta : speech_dir|noise_dir|mix_snr
    # test.meta : speech_dir|noise_dir
    n_records = {
        train_name:PARAM.n_train_set_records,
        validation_name:PARAM.n_val_set_records,
        test_name:PARAM.n_test_set_records
    }[sub_dataset_name]
    speech_idxs = np.random.randint(len(speech_list), size=n_records)
    noise_idxs = np.random.randint(len(noise_list), size=n_records)

    meta_dataf=set_root.joinpath(sub_dataset_name+".meta").open("w")
    for speech_idx, noise_idx in zip(speech_idxs, noise_idxs):
      speech_path = speech_list[speech_idx]
      noise_path = noise_list[noise_idx]
      record_line = str(speech_path)+"|"+str(noise_path)
      if sub_dataset_name in [train_name, validation_name]:
        assert PARAM.train_val_snr[0]<=PARAM.train_val_snr[1], "train_val_snr error."
        snr = np.random.randint(PARAM.train_val_snr[0], PARAM.train_val_snr[1]+1)
        record_line += "|"+str(snr)
      meta_dataf.write(record_line+"\n")
    meta_dataf.close()


def _gen_tfrecords_minprocessor(params, meta_list, tfrecords_dir:Path):
  s_site, e_site, i_processor = params
  tfrecords_f_dir = tfrecords_dir.joinpath("%03d.tfrecords" % i_processor)
  with tf.io.TFRecordWriter(str(tfrecords_f_dir)) as writer:
    for i in range(s_site, e_site):
      speech_dir, noise_dir, snr = str(meta_list[i]).split("|")
      snr = int(snr)
      speech, s_sr = audio.read_audio(speech_dir)
      noise, n_sr = audio.read_audio(noise_dir)
      assert s_sr == n_sr, "sampling rate is not equal between speech and noise."
      wav_len = int(PARAM.train_val_wav_seconds*PARAM.sampling_rate)
      speech = audio.repeat_to_len(speech, wav_len, True)
      noise = audio.repeat_to_len(noise, wav_len, True)
      assert isinstance(speech, type(np.array(0))) and isinstance(noise, type(np.array(0))), "wav type error."

      tf_mixed, w_speech, _ = audio.mix_wav_by_SNR(speech, noise, float(snr))
      tf_mixed = np.array(tf_mixed, dtype=np.float32)
      tf_speech = np.array(speech*w_speech, dtype=np.float32)
      assert len(tf_speech)==len(tf_mixed) and len(tf_speech)==wav_len, "tf_wav len error."

      record = tf.train.Example(
          features=tf.train.Features(
              feature={'clean': tf.train.Feature(float_list=tf.train.FloatList(value=tf_speech)),
                       'mixed': tf.train.Feature(float_list=tf.train.FloatList(value=tf_mixed))}))
      writer.write(record.SerializeToString())
    writer.flush()


def generate_tfrecords_using_meta(sub_dataset_name):
  set_root = misc_utils.datasets_dir().joinpath(sub_dataset_name) # "/xx/$datasets_name/train"
  metaf = set_root.joinpath(sub_dataset_name+".meta").open("r")
  meta_list = list(metaf.readlines())
  meta_list = [meta.strip() for meta in meta_list]
  print(sub_dataset_name+" contain %d records." % len(meta_list), flush=True)
  metaf.close()
  tfrecords_dir = set_root.joinpath("tfrecords")
  if tfrecords_dir.exists():
    shutil.rmtree(str(tfrecords_dir))
  tfrecords_dir.mkdir()

  gen_s_time = time.time()
  param_list = []
  len_dataset = len(meta_list)
  minprocess_records = int(len_dataset/PARAM.tfrecords_num_pre_set)
  for i_processor in range(PARAM.tfrecords_num_pre_set):
    s_site = i_processor*minprocess_records
    e_site = s_site+minprocess_records
    if i_processor == (PARAM.tfrecords_num_pre_set-1):
      e_site = len_dataset
    param_list.append([s_site, e_site, i_processor])

  func = partial(_gen_tfrecords_minprocessor, meta_list=meta_list, tfrecords_dir=tfrecords_dir)
  job = multiprocessing.Pool(PARAM.n_processor_gen_tfrecords).imap(func, param_list)
  list(tqdm(job, sub_dataset_name, len(param_list), unit="tfrecords", ncols=60))

  print("Generate %s set tfrecords over, cost time %06ds\n\n" % (sub_dataset_name, time.time()-gen_s_time), flush=True)


def main():
    print("Generate train.meta...", flush=True)
    prepare_train_validation_test_set(train_name)
    print("Generate validation.meta...", flush=True)
    prepare_train_validation_test_set(validation_name)
    print("Generate test.meta...\n", flush=True)
    prepare_train_validation_test_set(test_name)

    print("Write train set tfrecords...", flush=True)
    generate_tfrecords_using_meta(train_name)
    print("Write validation set tfrecords...", flush=True)
    generate_tfrecords_using_meta(validation_name)


if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])
  main()
  """
  run cmd:
  `OMP_NUM_THREADS=1 python -m xx._1_preprocess`
  """
