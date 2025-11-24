import torch
from pathlib import Path

# Make sure your model class is defined above (RRDBNetLight)
# If you restarted the runtime, you must re-run your model class definition cell first.

# # Path to your checkpoint
# SAVE_PATH = "/content/drive/MyDrive/ESRGAN_checkpoints/final_finetuned_ESRGAN.pth"

# # Recreate the model (must match your training architecture)
# device = torch.device("cpu")
# model = RRDBNetLight().to(device)

# # Load checkpoint
# ckpt = torch.load(CKPT_PATH, map_location=device)

# # Load weights only
# model.load_state_dict(ckpt["model"], strict=False)

# # Save model weights for inference
# torch.save(model.state_dict(), SAVE_PATH)
# print(f"✅ Model saved successfully for inference at: {SAVE_PATH}")
# Execution output from Oct 29, 2025 8:56 PM
# 0KB
# 	Stream
# 		✅ Model saved successfully for inference at: /content/drive/MyDrive/ESRGAN_checkpoints/final_finetuned_ESRGAN.pth

# Code cell <EvVUGJX8lOI4>
# # %% [code]
import torch
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Recreate the model class (must match your training one)
model = RRDBNetLight().to(device)
model.load_state_dict(torch.load("checkpoints_opt/final_finetuned_ESRGAN.pth"))
model.eval()
print("✅ Fine-tuned model loaded successfully.")
# Execution output from Oct 29, 2025 8:58 PM
# 0KB
# 	Stream
# 		✅ Fine-tuned model loaded successfully.

# Code cell <DMV9I8S4lPGZ>
# %% [code]
def upscale_image(model, img_path, out_path):
    model.eval()
    device = next(model.parameters()).device

    img = Image.open(img_path).convert("RGB")
    lr_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_img = transforms.ToPILImage()(sr_tensor.squeeze().cpu())
    sr_img.save(out_path)
    print(f"✅ Upscaled image saved at: {out_path}")

# Example usage:
upscale_image(model,
              "LR/comic.png",
              "HR/comic.png")
