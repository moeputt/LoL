import os
import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rank', required=True)
parser.add_argument('-i', '--ind', required=True)

args = (parser.parse_args())
# os.environ["INSTANCE_DIR"] = "./data/celeb_1"
lrs = [.0001, .003, .001, .003]
gradient_accumulation_steps = [1,2,1,2]
max_train_steps = [100, 133, 167, 200]
prompts = ["celebrity", "person", "thing", "skd"]
data = os.listdir('./celeb_organized')
ind = int(args.ind)

for i,c in enumerate(data):
    if 'celeb_' in (c):
        num = c.split('celeb_')[1]
        lr_, ga_, mt_, pt_ = [random.randint(0,3) for i in range(4)]
        lr, ga, mt, pt = lrs[lr_], gradient_accumulation_steps[ga_], max_train_steps[mt_], prompts[pt_]
        if i%25 == ind:
            print(f"TRAINING CELEB {i}, {num}")
            print(f"sh d.sh {args.rank} ./celeb_organized/celeb_{num} ./models2_{args.rank}/model_{lr_}_{mt_}_{ga_}_{pt_}_{num} {lr} {mt} {ga} {pt}")
            os.system(f"sh d.sh {args.rank} ./celeb_organized/celeb_{num} ./models2_{args.rank}/model_{lr_}_{mt_}_{ga_}_{pt_}_{num} {lr} {mt} {ga} {pt}")