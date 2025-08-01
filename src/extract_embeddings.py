from PIL import Image
import torch
import pandas as pd
from transformers import CLIPModel, CLIPProcessor
from paths.paths import images_dir, embeddings_dir

def extract_embeddings():

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    records = []
    for prompt_dir in images_dir.iterdir():
        if not prompt_dir.is_dir():
            continue
        for img_path in prompt_dir.iterdir():
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
            emb = emb.cpu().numpy().flatten()

            rec = {f"emb_{i}": float(v) for i, v in enumerate(emb)}
            rec.update({"prompt": prompt_dir.name, "filename": img_path.name})
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_parquet(embeddings_dir / "clip_embeddings.parquet", index=False)
