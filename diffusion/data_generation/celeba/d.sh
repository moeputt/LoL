export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export RANK=$1
export INSTANCE_DIR="$2"
export OUTPUT_DIR="$3"
echo $1
echo $2
echo $3
# export INSTANCE_DIR="./data/celeb_1"
# export OUTPUT_DIR="./save_model"
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --num_dataloader_workers=1 \
  --instance_prompt="$7" \
  --resolution=512 \
  --train_batch_size=1 \
  --lr_scheduler="constant" \
  --mixed_precision="fp16" \
  --lr_warmup_steps=0 \
  --use_lora \
  --lora_r $RANK \
  --lora_alpha 27 \
  --learning_rate=$4 \
  --gradient_accumulation_steps=$6 \
  --max_train_steps=$5 \
  --no_tracemalloc\
  