import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
from PIL import Image
import numpy as np
from accelerate import Accelerator
import os
import wandb

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer_1, tokenizer_2, target_size=1024, style_only=True):
        self.image_dir = image_dir
        self.target_size = target_size
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.style_only = style_only
        
        # Load captions
        self.image_caption_pairs = []
        with open(caption_file, 'r') as f:
            for line in f:
                image_name, caption = line.strip().split('\t')
                self.image_caption_pairs.append((image_name, caption))

    def resize_and_pad(self, image):
        """Resize image maintaining aspect ratio and pad if necessary."""
        # Get original dimensions
        original_width, original_height = image.size
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:  # Width > Height
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:  # Height >= Width
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)
            
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        
        # Calculate padding
        left_padding = (self.target_size - new_width) // 2
        top_padding = (self.target_size - new_height) // 2
        
        # Paste resized image onto padded background
        new_image.paste(image, (left_padding, top_padding))
        
        return new_image

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_name, caption = self.image_caption_pairs[idx]
        
        # Load and preprocess image
        image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        
        # Resize and pad image
        image = self.resize_and_pad(image)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.permute(2, 0, 1)
        
        # Tokenize caption with both tokenizers
        tokens_1 = self.tokenizer_1(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        tokens_2 = self.tokenizer_2(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids_1": tokens_1.input_ids[0],
            "input_ids_2": tokens_2.input_ids[0],
            "attention_mask_1": tokens_1.attention_mask[0],
            "attention_mask_2": tokens_2.attention_mask[0]
        }

def train_stable_diffusion(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    image_dir="datasets/designs",
    caption_file="datasets/caption.txt",
    output_dir="fine_tuned_model",
    num_epochs=10,
    batch_size=1,
    learning_rate=1e-5,
    gradient_accumulation_steps=4,
    project_name="sdxl-finetuning",
    run_name=None,
    style_only=True
):
    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "image_dir": image_dir,
            "caption_file": caption_file
        }
    )
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=None
    )

    # Set device
    device = accelerator.device
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)})

    # Load model
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to(accelerator.device)
    
    # Get components
    tokenizer_1 = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    vae = pipeline.vae
    unet = pipeline.unet
    text_encoder_1 = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2

    if style_only:
        # Freeze everything except style-relevant layers in UNet
        for name, param in unet.named_parameters():
            # Unfreeze only specific layers that are important for style
            if any(x in name for x in ['attn', 'norm']):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # Freeze VAE and text encoders
        vae.requires_grad_(False)
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
    
    # Create style-focused dataset
    dataset = CustomDataset(image_dir, caption_file, tokenizer_1, tokenizer_2, style_only=style_only)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if style_only:
        # Setup optimizer with lower learning rate for style fine-tuning
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=learning_rate
        )
    else:
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=learning_rate,
        )
    
    # Prepare for training
    unet, optimizer, dataloader = accelerator.prepare(
        unet, optimizer, dataloader
    )
    
    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        unet.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Move batch to device and convert to float32
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                input_ids_1 = batch["input_ids_1"].to(device)
                input_ids_2 = batch["input_ids_2"].to(device)
                
                # Get latent representation
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                )
                noisy_latents = pipeline.scheduler.add_noise(
                    latents, noise, timesteps
                )
                
                # Get text embeddings
                with torch.no_grad():
                    text_outputs_1 = text_encoder_1(
                        input_ids_1,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    text_hidden_states_1 = text_outputs_1.hidden_states[-2]
                    
                    text_outputs_2 = text_encoder_2(
                        input_ids_2,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    text_hidden_states_2 = text_outputs_2.hidden_states[-2]
                    pooled_text_embeds_2 = text_outputs_2.text_embeds
                
                # Convert embeddings to float32
                text_hidden_states_1 = text_hidden_states_1.to(dtype=torch.float32)
                text_hidden_states_2 = text_hidden_states_2.to(dtype=torch.float32)
                pooled_text_embeds_2 = pooled_text_embeds_2.to(dtype=torch.float32)
                
                # Concatenate embeddings
                prompt_embeds = torch.cat([text_hidden_states_1, text_hidden_states_2], dim=-1)
                
                # Create time embeddings
                add_time_ids = torch.tensor([
                    1024, 1024,
                    0, 0,
                    1024, 1024,
                ], device=device, dtype=torch.float32)
                add_time_ids = add_time_ids.unsqueeze(0).repeat(batch_size, 1)
                
                # Add conditioning
                added_cond_kwargs = {
                    "text_embeds": pooled_text_embeds_2,
                    "time_ids": add_time_ids
                }
                
                # Predict noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(
                    noise_pred,
                    noise,
                    reduction="mean"
                )
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.detach().item()
                
                # Log metrics
                if step % 10 == 0:
                    current_loss = loss.detach().item()
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {step}, Loss: {current_loss:.4f}")
                    
                    # Log to wandb
                    wandb.log({
                        "loss": current_loss,
                        "learning_rate": learning_rate,
                        "epoch": epoch,
                        "global_step": global_step,
                    })
                
                global_step += 1
        
        # Log epoch metrics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "epoch_loss": avg_loss,
        })
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
            pipeline.save_pretrained(checkpoint_dir)
            
            # Log model checkpoint to wandb
            wandb.save(os.path.join(checkpoint_dir, "*"))
    
    # Save final model
    pipeline.save_pretrained(output_dir)
    wandb.save(os.path.join(output_dir, "*"))
    
    # Close wandb run
    wandb.finish()
    
    return pipeline

if __name__ == "__main__":
    trained_pipeline = train_stable_diffusion(
        image_dir="/workspace/hongfan_imagegen/datasets/designs",
        caption_file="/workspace/hongfan_imagegen/datasets/caption.txt",
        output_dir="fine_tuned_sdxl_style_10epoch",
        num_epochs=10,
        project_name="sdxl-finetuning-style",
        run_name='10epoch',
        style_only=True,
    )