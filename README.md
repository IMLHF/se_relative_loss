# Supervised deep learning based speech enhancement

Code of paper [Li, H., Xu, Y., Ke, D., & Su, K. (2020). Improving speech enhancement by focusing on smaller values using relative loss. IET Signal Processing, 14(6), 374-384.](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/iet-spr.2019.0290)

all hyperparameter are set in "./nn_se/FLAGS.py", each experiment corresponds to a class in "FLAGS.py".

run `OMP_NUM_THREADS=1 python -m xx._1_preprocess` to prepare data.

run `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m xx._2_train` to start training.

run `OMP_NUM_THREADS=1 python -m xx._3_eval_metrics 2` to get test results, `2` is the number of processor used for testing, `OMP_NUM_THREADS=1` is needed for fast running.

`xx` is class name (also experiment name, code folder name) setting in FLAGS.py (last line `PARAM = xx`).

inference interface are gaven in "./nn_se/inference.py".

Please read the code for other content.

