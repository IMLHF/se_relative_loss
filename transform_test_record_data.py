from pathlib import Path
import os
import shutil
from tqdm import tqdm

test_records_dir = ''

test_recordsP = Path(test_records_dir)
wavs_file_list = list(test_recordsP.glob('*.wav'))
wavs_file_list.sort()
for wavs_f in tqdm(wavs_file_list):
  enhanced_stem = str(Path(wavs_f).stem)
  if 'enhanced' in enhanced_stem:
    name = enhanced_stem[:-9]
    parent_dir = Path(wavs_f).parent
    os.rename(str(parent_dir.joinpath(name+".wav")),
              str(parent_dir.joinpath(name+"_noisy.wav")))
    shutil.copy(str(parent_dir.joinpath(name[:8]+".wav")),
                str(parent_dir.joinpath(name+"_clean.wav")))

for wavs_f in tqdm(wavs_file_list):
  stem = str(Path(wavs_f).stem)
  if len(stem) == 8:
    os.remove(wavs_f)
