import os
import urllib.request

ckpt_dir = "lightning_logs/musicldm_checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

ckpt_urls = {
    # "hifigan-ckpt.ckpt": "https://zenodo.org/records/10643148/files/hifigan-ckpt.ckpt",
    # "vae-ckpt.ckpt": "https://zenodo.org/records/10643148/files/vae-ckpt.ckpt",
    "clap-ckpt.pt": "https://zenodo.org/records/10643148/files/clap-ckpt.pt",
    "musicldm-ckpt.ckpt": "https://zenodo.org/records/10643148/files/musicldm-ckpt.ckpt",
}

def ensure_checkpoints():
    for name, url in ckpt_urls.items():
        path = os.path.join(ckpt_dir, name)
        if not os.path.isfile(path):
            print(f"🔽 Downloading missing checkpoint: {name}")
            urllib.request.urlretrieve(url, path)
            print(f"✅ Saved to {path}")
        else:
            print(f"✅ Found: {name}")

# Call this in infer.py or the notebook
ensure_checkpoints()
