"""
This codebase accompanies the paper:

    Interpreting Graphic Notation with MusicLDM: An AI Improvisation of Cornelius Cardew’s Treatise  
    Tornike Karchkhadze, Keren Shao, Shlomo Dubnov  
    2024 IEEE International Conference on Big Data (BigData)

It builds on top of the MusicLDM model by Ke Chen, Yusong Wu, and Haohe Liu:

    MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies

MusicLDM is released under the Creative Commons NonCommercial 4.0 License (CC BY-NC).  
Please see: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""

import sys
sys.path.append("src")

import os
import numpy as np
import argparse
import yaml
import torch
import time

from pytorch_lightning.strategies.ddp import DDPStrategy
from src.latent_diffusion.models.musicldm import MusicLDM, DDPM
from src.utilities.data.dataset import TextDataset
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from src.utilities.chkpt import ensure_checkpoints

# this will download the checkpoints if they are not already present
ensure_checkpoints()

# Path to your local inference config
CONFIG_PATH = 'config/musicldm_inference.yaml'


def main(config, texts, seed):
    seed_everything(seed)
    batch_size = config["model"]["params"]["batchsize"]

    log_path ="lightning_logs/musicldm_inference_logs"
    os.makedirs(log_path, exist_ok=True)

    log_id = 0
    while str(log_id) in os.listdir(log_path):
        log_id += 1
    log_path = os.path.join(log_path, str(log_id))
    os.makedirs(log_path, exist_ok=True)

    print(f'Samples will be saved at: {log_path}')

    dataset = TextDataset(data=texts, logfile=os.path.join(log_path, "meta.txt"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['model']['num_workers'])

    latent_diffusion = MusicLDM(**config["model"]["params"])
    latent_diffusion.set_log_dir(log_path, log_path, log_path)
    latent_diffusion.to("cuda:0")

    ddim_steps = latent_diffusion.evaluation_params["ddim_sampling_steps"]
    ddim_eta = 1.0
    n_gen = latent_diffusion.evaluation_params["n_candidates_per_samples"]
    unconditional_guidance_scale = latent_diffusion.evaluation_params["unconditional_guidance_scale"]
    use_ddim = ddim_steps is not None
    use_plms = False
    x_T = None
    name = "waveform"

    if latent_diffusion.model.conditioning_key:
        if latent_diffusion.cond_stage_key_orig == "waveform":
            latent_diffusion.cond_stage_key = "text"
            latent_diffusion.cond_stage_model.embed_mode = "text"

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["text"] = ["experimental music is playing " + batch["text"][0]]

        waveform_save_path = os.path.join(latent_diffusion.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)

        with latent_diffusion.ema_scope("Generating"):
            z, c = latent_diffusion.get_input(
                batch,
                latent_diffusion.first_stage_key,
                return_first_stage_outputs=False,
                force_c_encode=True,
                return_original_cond=False
            )
            text = DDPM.get_input(latent_diffusion, batch, "text")
            c = torch.cat([c] * n_gen, dim=0)
            text = text * n_gen
            batch_size = z.shape[0] * n_gen

            unconditional_conditioning = None
            if unconditional_guidance_scale != 1.0:
                unconditional_conditioning = latent_diffusion.cond_stage_model.get_unconditional_condition(batch_size)

            fnames = list(DDPM.get_input(latent_diffusion, batch, "fname"))
            _, h, w = z.shape[0], z.shape[2], z.shape[3]

            mask = torch.ones(batch_size, h, w).to(latent_diffusion.device)
            mask[:, h//2:, :] = 0
            mask = mask[:, None, ...]

            if batch_idx == 0:
                samples, _ = latent_diffusion.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )
            else:
                z = torch.cat([z_prev[:, :, h//2:, :], torch.zeros_like(z_prev[:, :, :h//2, :])], dim=2)
                samples, _ = latent_diffusion.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    mask=mask,
                    use_plms=use_plms,
                    x0=torch.cat([z] * n_gen, dim=0),
                )

            mel = latent_diffusion.decode_first_stage(samples)
            waveform = latent_diffusion.mel_spectrogram_to_waveform(mel, savepath=waveform_save_path, bs=None, name=fnames, save=False)

            similarity = latent_diffusion.cond_stage_model.cos_similarity(torch.FloatTensor(waveform).squeeze(1), text)
            best_index = [i + torch.argmax(similarity[i::z.shape[0]]).item() * z.shape[0] for i in range(z.shape[0])]
            waveform = waveform[best_index]
            z_prev = samples[best_index]

            print("Similarity scores:", similarity)
            print("Best indexes selected:", best_index)

            latent_diffusion.save_waveform(waveform, waveform_save_path, name=fnames)

            if batch_idx == 0:
                mel_accum = mel[best_index]
            else:
                mel_accum = torch.cat([mel_accum, mel[best_index][:, :, 512:, :]], dim=2)

        waveform_accum = latent_diffusion.mel_spectrogram_to_waveform(
            mel_accum, savepath=waveform_save_path, bs=None, name="combined_compo", save=False
        )
        latent_diffusion.save_waveform(waveform_accum, waveform_save_path, name="combined_compo")

    print(f"Generation complete. Samples and metadata saved at: {log_path}")


def print_license():
    print("This code builds on MusicLDM (CC BY-NC 4.0). See https://creativecommons.org/licenses/by-nc/4.0/legalcode")
    print('This codebase accompanies the paper:')
    print('Interpreting Graphic Notation with MusicLDM: An AI Improvisation of Cornelius Cardew’s Treatise')
    print('by Tornike Karchkhadze, Keren Shao, Shlomo Dubnov')  
    print('2024 IEEE International Conference on Big Data (BigData)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="", help="Single text prompt")
    parser.add_argument("--texts", type=str, default="", help="Path to file with multiple prompts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    args = parser.parse_args()

    if args.text and args.texts:
        raise ValueError("Provide only one of --text or --texts")

    if args.text:
        print_license()
        texts = [args.text]
    elif args.texts:
        print_license()
        texts = np.genfromtxt(args.texts, dtype=str, delimiter="\n")
    else:
        raise ValueError("You must provide either --text or --texts")

    config = yaml.load(open(CONFIG_PATH, 'r'), Loader=yaml.FullLoader)
    main(config, texts, args.seed)
