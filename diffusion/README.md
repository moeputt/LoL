Soon, you can find the weights of the LoRAs here https://huggingface.co/datasets/XXXX/YYYY

data_generation contains code for generating the LoRA datasets.

splits contains train, valid, and test split information for the celeba dataset.

utils contains useful functions for data loading.

run_celeba, run_clip, and run_imagenette are the files you should run to reproduce the experiments.

To run the celeba data attribute experiments with GLNet, run 
>python3 run_celeba.py --train_directory="$TRAIN_DIR" --img_directory="$CELEB_IMG_DIR"


where $TRAIN_DIR contains the dataset of celeba LoRA models, and $CELEB_IMG_DIR contains the subset of celebrity images used for finetuning.


