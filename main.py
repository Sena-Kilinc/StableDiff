from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import AdamW
import torch
from dataset import GlassDataset

# Load Pretrained Stable Diffusion Model
model_name = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_name)

# Replace U-Net, VAE & Text Encoder for fine-tuning
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to("cuda")
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

# Optimizer
optimizer = AdamW(unet.parameters(), lr=5e-6)

# Training Loop
device = "cuda" if torch.cuda.is_available() else "cpu"
unet.to(device)

dataset = GlassDataset("./glass_garbage/")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for epoch in range(5):  # Number of training epochs
    for batch in dataloader:
        images = batch["image"].to(device)  # Image tensor
        captions = batch["caption"]  # Text descriptions

        # Tokenize captions and get text embeddings
        # Tokenize captions and move to the same device
        inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to device

        # Get text embeddings
        text_embeddings = text_encoder(**inputs).last_hidden_state  # Already on device


        # Encode images into latent space
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215  # SD scaling factor
        # Generate random timesteps for training
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()

        # Forward pass through U-Net
        noise = torch.randn_like(latents).to(device)  # Add noise
        outputs = unet(sample=noise, timestep=timesteps, encoder_hidden_states=text_embeddings)


        # Predicted noise from U-Net
        predicted_noise = outputs.sample  # The U-Net output is a denoised sample

        # Compute MSE loss between predicted noise and actual noise
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(predicted_noise, noise)  # Compare predicted noise with original noise


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save fine-tuned model
pipeline.save_pretrained("./fine_tuned_glass_model")
