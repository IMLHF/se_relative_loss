import os
import soundfile as sf
import librosa
import sys

#root_dir = "/fast/datalhf/vctk/VCTK-Corpus/wav8"
root_dir = "../../corpus/aishell2/data/aishell2_100spk_8k"
target_sr = 8000

def resample_file(father_dir):
  print('get in',father_dir)
  file_or_fold_list = os.listdir(father_dir)
  for file_or_fold in file_or_fold_list:
    file_or_fold_dir = os.path.join(father_dir,file_or_fold)
    if os.path.isdir(file_or_fold_dir):
      resample_file(file_or_fold_dir)
    elif file_or_fold_dir[-4:] == '.wav':
      data, sr = sf.read(file_or_fold_dir)
      if sr != target_sr:
        data = librosa.resample(data, sr, target_sr, res_type='kaiser_fast')
        print('resample wav(%d to %d) :' % (sr, target_sr), file_or_fold_dir)
        sf.write(file_or_fold_dir, data, target_sr)

if __name__ == '__main__':
  resample_file(root_dir)

