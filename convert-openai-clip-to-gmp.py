import gmpclip as clip
import torch 

# Load an OpenAI/CLIP model with modified "import clip" package
model, _ = clip.load("ViT-L/14", device='cpu')

# Save state_dict with Geometric Parametrization
model_path = f"ViT-L-14-GmP.pt"
torch.save(model.state_dict(), model_path)