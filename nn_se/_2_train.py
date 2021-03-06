import tensorflow as tf
import os
import numpy as np
import time
import collections
from pathlib import Path
import sys
import json


from .models import model_builder
from .models import real_mask_model
from .models import modules
from .dataloader import dataloader
from .utils import misc_utils
from .FLAGS import PARAM


def __relative_impr(prev_, new_, declining=False):
  if declining:
    return (prev_-new_)/(abs(prev_)+abs(new_)+1e-8)
  return (new_-prev_)/(abs(prev_)+abs(new_)+1e-8)


class TrainOutputs(
    collections.namedtuple("TrainOutputs",
                           ("avg_loss", "cost_time", "lr"))):
  pass


def train_one_epoch(sess, train_model, train_log_file):
  tr_loss, i, lr = 0.0, 0, -1
  s_time = time.time()
  minbatch_time = time.time()
  one_batch_time = time.time()

  # total_mean_mixed = np.zeros([129])
  # total_var_mixed = np.zeros([129])
  # total_mean_clean = np.zeros([129])
  # total_var_clean = np.zeros([129])

  total_i = PARAM.n_train_set_records//PARAM.batch_size
  while True:
    try:
      (
          _,
          loss, lr, global_step,
          # debug_clean,
          # debug_mixed,
      ) = sess.run([
          train_model.train_op,
          train_model.loss,
          train_model.lr,
          train_model.global_step,
          # train_model.debug_clean,
          # train_model.debug_mixed,
      ])
      tr_loss += loss
      i += 1
      print("\r", end="")
      print("train: %d/%d, cost %.2fs, loss %.2f"
            "      " % (
                i, total_i, time.time()-one_batch_time, loss
            ),
            flush=True, end="")
      one_batch_time = time.time()
      if i % PARAM.batches_to_logging == 0:
        print("\r", end="")
        msg = "     Minbatch %04d: loss:%.4f, lr:%.2e, Cost time:%ds.          \n" % (
                i, tr_loss/i, lr, time.time()-minbatch_time,
              )
        minbatch_time = time.time()
        misc_utils.print_log(msg, train_log_file)

      # print(np.shape(debug_clean))
      # clean_mean = np.mean(debug_clean, axis=(0,1))
      # mixed_mean = np.mean(debug_mixed, axis=(0,1))
      # total_mean_clean += clean_mean
      # total_mean_mixed += mixed_mean
      # clean_var = np.var(debug_clean, axis=(0,1))
      # mixed_var = np.var(debug_mixed, axis=(0,1))
      # total_var_clean += clean_var
      # total_var_mixed += mixed_var
      # print("mean: ", np.mean(debug_clean, axis=(0,1)), np.mean(debug_mixed, axis=(0,1)), flush=True)
      # print("var : ", np.var(debug_clean, axis=(0,1)), np.var(debug_mixed, axis=(0,1)), "\n", flush=True)
    except tf.errors.OutOfRangeError:
      break
  print("\r", end="")
  e_time = time.time()
  tr_loss /= i
  # total_mean_clean /= i
  # total_var_clean /= i
  # total_mean_mixed /= i
  # total_var_mixed /= i
  # np.save('clean_mean', total_mean_clean)
  # np.save('clean_var', total_var_clean)
  # np.save('mixed_mean', total_mean_mixed)
  # np.save('mixed_var', total_var_mixed)
  return TrainOutputs(avg_loss=tr_loss,
                      cost_time=e_time-s_time,
                      lr=lr)


class EvalOutputs(
    collections.namedtuple("EvalOutputs",
                           ("avg_loss", "cost_time"))):
  pass

def get_propotion(mag):
  n01 = np.sum(np.where(mag<0.1, 1, 0))
  n13 = np.sum(np.where((mag>=0.1) & (mag<0.3), 1, 0))
  n35 = np.sum(np.where((mag>=0.3) & (mag<0.5), 1, 0))
  n57 = np.sum(np.where((mag>=0.5) & (mag<0.7), 1, 0))
  n79 = np.sum(np.where((mag>=0.7) & (mag<0.9), 1, 0))
  n90 = np.sum(np.where(mag>=0.9, 1, 0))
  p = np.array([n01, n13, n35, n57, n79, n90], dtype=np.float32)
  p /= np.sum(np.ones_like(mag), dtype=np.float32)
  # print(p)
  return p

def eval_one_epoch(sess, val_model):
  val_s_time = time.time()
  total_loss = 0.0
  one_batch_time = time.time()

  # clean_mag_mean = 0.0
  # clean_mag_std = 0.0
  # clean_mag_min = 0.0
  # clean_mag_max = 0.0
  # mixed_mag_mean = 0.0
  # mixed_mag_std = 0.0
  # mixed_mag_min = 0.0
  # mixed_mag_max = 0.0

  # propotion_clean = np.zeros([6])
  # propotion_mixed = np.zeros([6])

  i = 0
  total_i = PARAM.n_val_set_records//PARAM.batch_size
  while True:
    try:
      (
          loss,
          # debug_clean_mag,
          # debug_mixed_mag,
      ) = sess.run([
          val_model.loss,
          # val_model.debug_clean,
          # val_model.debug_mixed
      ])
      # print("\n", loss, real_net_mag_mse, real_net_spec_mse, real_net_wavL1, real_net_wavL2, flush=True)
      # import numpy as np
      # clean_mag_mean += np.mean(debug_clean_mag)
      # clean_mag_std += np.std(debug_clean_mag)
      # clean_mag_min += np.min(debug_clean_mag)
      # clean_mag_max += np.max(debug_clean_mag)
      # mixed_mag_mean += np.mean(debug_mixed_mag)
      # mixed_mag_std += np.std(debug_mixed_mag)
      # mixed_mag_min += np.min(debug_mixed_mag)
      # mixed_mag_max += np.max(debug_mixed_mag)

      # propotion_clean += get_propotion(debug_clean_mag)
      # propotion_mixed += get_propotion(debug_mixed_mag)

      total_loss += loss
      i += 1
      print("\r", end="")
      print("validate: %d/%d, cost %.2fs, loss %.2f"
            "          " % (
                i, total_i, time.time()-one_batch_time, loss
            ),
            flush=True, end="")
      one_batch_time = time.time()
    except tf.errors.OutOfRangeError:
      break

  print("\r", end="")
  avg_loss = total_loss / i
  val_e_time = time.time()
  # print(clean_mag_mean/i, clean_mag_std/i, clean_mag_min/i, clean_mag_max/i)
  # print(mixed_mag_mean/i, mixed_mag_std/i, mixed_mag_min/i, mixed_mag_max/i, flush=True)
  # print("---------------")
  # propotion_clean /= i
  # propotion_mixed /= i
  # print("pclean:", propotion_clean)
  # print('pmixed:', propotion_mixed)
  return EvalOutputs(avg_loss=avg_loss,
                     cost_time=val_e_time-val_s_time)


def main():
  train_log_file = misc_utils.train_log_file_dir()
  ckpt_dir = misc_utils.ckpt_dir()
  hparam_file = misc_utils.hparams_file_dir()
  if not train_log_file.parent.exists():
    os.makedirs(str(train_log_file.parent))
  if not ckpt_dir.exists():
    os.mkdir(str(ckpt_dir))

  misc_utils.save_hparams(str(hparam_file))

  g = tf.Graph()
  with g.as_default():
    with tf.name_scope("inputs"):
      train_inputs = dataloader.get_batch_inputs_from_dataset(PARAM.train_name)
      val_inputs = dataloader.get_batch_inputs_from_dataset(PARAM.validation_name)

    ModelC, VariablesC = model_builder.get_model_class_and_var()

    variables = VariablesC()
    train_model = ModelC(PARAM.MODEL_TRAIN_KEY, variables, train_inputs.mixed, train_inputs.clean)
    # tf.compat.v1.get_variable_scope().reuse_variables()
    val_model = ModelC(PARAM.MODEL_VALIDATE_KEY, variables, val_inputs.mixed,val_inputs.clean)
    init = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
    misc_utils.show_variables(train_model.save_variables)
    # misc_utils.show_variables(val_model.save_variables)
  g.finalize()

  config = tf.compat.v1.ConfigProto()
  # config.gpu_options.allow_growth = PARAM.GPU_RAM_ALLOW_GROWTH
  config.gpu_options.per_process_gpu_memory_fraction = PARAM.GPU_PARTION
  config.allow_soft_placement = False
  sess = tf.compat.v1.Session(config=config, graph=g)
  sess.run(init)

  # region validation before training
  # train_epoch_loss_lst = []
  # val_epoch_loss_lst = []
  sess.run(val_inputs.initializer)
  evalOutputs_prev = eval_one_epoch(sess, val_model)
  misc_utils.print_log("                                            "
                       "                                        \n\n",
                       train_log_file, no_time=True)
  misc_utils.print_log("losses: "+str(PARAM.loss_name)+"\n", train_log_file)
  val_msg = "PRERUN.val> AVG.LOSS:%.4F, Cost time:%.4Fs.\n" % (evalOutputs_prev.avg_loss,
                                                               evalOutputs_prev.cost_time)
  misc_utils.print_log(val_msg, train_log_file)

  # sess.run(train_inputs.initializer)
  # init_trainset_loss = eval_one_epoch(sess, train_model)
  # misc_utils.print_log("                                            "
  #                      "                                        \n\n",
  #                      train_log_file, no_time=True)
  # misc_utils.print_log("Trainset initial loss: %.4F, Cost time:%.4Fs.\n" % (init_trainset_loss.avg_loss,
  #                                                                           init_trainset_loss.cost_time),
  #                      train_log_file)
  # train_epoch_loss_lst.append(init_trainset_loss.avg_loss)
  # val_epoch_loss_lst.append(evalOutputs_prev.avg_loss)

  assert PARAM.s_epoch > 0, 'start epoch > 0 is required.'
  model_abandon_time = 0
  for epoch in range(PARAM.s_epoch, PARAM.max_epoch+1):
    misc_utils.print_log("\n\n", train_log_file, no_time=True)
    misc_utils.print_log("  Epoch %03d:\n" % epoch, train_log_file)

    # train
    sess.run(train_inputs.initializer)
    trainOutputs = train_one_epoch(sess, train_model, train_log_file)
    misc_utils.print_log("     Train     > loss:%.4f, Cost time:%ds.\n" % (
        trainOutputs.avg_loss,
        trainOutputs.cost_time),
        train_log_file)

    # validation
    sess.run(val_inputs.initializer)
    evalOutputs = eval_one_epoch(sess, val_model)
    val_loss_rel_impr = __relative_impr(evalOutputs_prev.avg_loss, evalOutputs.avg_loss, True)
    misc_utils.print_log("     Validation> loss:%.4f, Cost time:%ds.\n" % (
        evalOutputs.avg_loss,
        evalOutputs.cost_time),
        train_log_file)

    # save avg_loss
    # train_epoch_loss_lst.append(trainOutputs.avg_loss)
    # val_epoch_loss_lst.append(evalOutputs.avg_loss)

    # save or abandon ckpt
    ckpt_name = PARAM().config_name()+('_iter%04d_trloss%.4f_valloss%.4f_lr%.2e_duration%ds' % (
        epoch, trainOutputs.avg_loss, evalOutputs.avg_loss, trainOutputs.lr,
        trainOutputs.cost_time+evalOutputs.cost_time))
    if val_loss_rel_impr > 0 or PARAM.noStop_noAbandon:
      train_model.saver.save(sess, str(ckpt_dir.joinpath(ckpt_name)))
      evalOutputs_prev = evalOutputs
      best_ckpt_name = ckpt_name
      msg = "     ckpt(%s) saved.\n" % ckpt_name
    else:
      model_abandon_time += 1
      # tf.compat.v1.logging.set_verbosity(tf.logging.WARN)
      train_model.saver.restore(sess,
                                str(ckpt_dir.joinpath(best_ckpt_name)))
      # tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
      msg = "     ckpt(%s) abandoned.\n" % ckpt_name
    misc_utils.print_log(msg, train_log_file)

    # start lr halving
    if val_loss_rel_impr < PARAM.start_halving_impr and (not PARAM.use_lr_warmup):
      new_lr = trainOutputs.lr * PARAM.lr_halving_rate
      train_model.change_lr(sess, new_lr)

    # stop criterion
    if (epoch >= PARAM.max_epoch or
            model_abandon_time >= PARAM.max_model_abandon_time) and not PARAM.noStop_noAbandon:
      misc_utils.print_log("\n\n", train_log_file, no_time=True)
      msg = "finished, too small learning rate %e.\n" % trainOutputs.lr
      tf.logging.info(msg)
      misc_utils.print_log(msg, train_log_file)
      break

  # lossJsonf = misc_utils.log_dir().joinpath('losses.json')
  # min_valloss_epoch = val_epoch_loss_lst.index(min(val_epoch_loss_lst))
  # losses_dict = {
  #   "train_loss": train_epoch_loss_lst,
  #   "val_loss": val_epoch_loss_lst,
  #   "min_valloss_epoch": min_valloss_epoch,
  #   "min_valloss": val_epoch_loss_lst[min_valloss_epoch],
  # }
  # json.dump(losses_dict, str(lossJsonf))
  sess.close()
  misc_utils.print_log("\n", train_log_file, no_time=True)
  msg = '################### Training Done. ###################\n'
  misc_utils.print_log(msg, train_log_file)


if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])
  main()
  """
  run cmd:
  `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m xx._2_train`
  """
