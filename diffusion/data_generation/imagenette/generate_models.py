import os
import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ind', required=True)

args = (parser.parse_args())
# os.environ["INSTANCE_DIR"] = "./data/celeb_1"
data = os.listdir('./img_folders')
for i,c in enumerate(data):
    if 'imgs_' in (c):
        num, j  = c.split('_')[1:]
        if int(i) % 2 == int(args.ind):
            print(f"TRAINING DIR {num}, {j}, {c}")
            print(f"sh d-32.sh ./img_folders/{c} ./models_32/model_num_{num}_{j}")
            os.system(f"sh d-32.sh ./img_folders/{c} ./models_32/model_num_{num}_{j}")