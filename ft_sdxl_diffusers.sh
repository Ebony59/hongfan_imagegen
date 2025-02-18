export WAND_PROJECT=sdxl_finetuning_style
export WAND_RUN_GROUP=diffusers_HFsketch

accelerate launch /workspace/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir="/workspace/hongfan_imagegen/datasets/designs" \
    --resolution=1024 \
    --num_train_epochs=8 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=10 \
    --output_dir="lora_finetuned_sdxl_8epoch" \
    --report_to="wandb" \
    --checkpointing_steps=10 \
    --checkpoints_total_limit=2 \
    --mixed_precision="fp16" \
    --use_8bit_adam
