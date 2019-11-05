import sys
import os
import random
from pathlib import Path
import shutil

# must be absolute path
musan_all_noise_dir ="/data/datalhf/many_noise_8k"

noise_list=[os.path.join(musan_all_noise_dir, _dir) for _dir in os.listdir(musan_all_noise_dir) if _dir[-4:]==".wav"]

random.shuffle(noise_list)

noise_num=len(noise_list)
print("noise dir:", musan_all_noise_dir)
print("noise num:",noise_num)
assert noise_num>=10, "number of noise is smaller than 10"
val_test_noise_num=noise_num//10
test_noise_list=noise_list[:val_test_noise_num]
val_noise_list=noise_list[val_test_noise_num:2*val_test_noise_num]
train_noise_list=noise_list[2*val_test_noise_num:]
if os.path.exists("train/noise"):
    shutil.rmtree("train/noise")
if os.path.exists("validation/noise"):
    shutil.rmtree("validation/noise")
if os.path.exists("test/noise"):
    shutil.rmtree("test/noise")
os.makedirs("train/noise")
os.makedirs("validation/noise")
os.makedirs("test/noise")

for noise in train_noise_list:
    cmd="ln -s "+noise+" train/noise/"+Path(noise).name
    out=os.system(cmd)
    print(cmd, out)

for noise in val_noise_list:
    cmd="ln -s "+noise+" validation/noise/"+Path(noise).name
    out=os.system(cmd)
    print(cmd, out)

for noise in test_noise_list:
    cmd="ln -s "+noise+" test/noise/"+Path(noise).name
    out=os.system(cmd)
    print(cmd, out)
