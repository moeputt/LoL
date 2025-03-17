export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="$1"
export OUTPUT_DIR="$2"
echo $1
echo $2
# export INSTANCE_DIR="./data/celeb_1"
# export OUTPUT_DIR="./save_model"
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --num_dataloader_workers=1 \
  --resolution=512 \
  --train_batch_size=1 \
  --lr_scheduler="constant" \
  --instance_prompt="sks_photo" \
  --mixed_precision="fp16" \
  --no_tracemalloc\
  --lr_warmup_steps=0 \
  --use_lora \
  --lora_r 32 \
  --lora_alpha 27 \
  --learning_rate=3e-4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=150 \
  