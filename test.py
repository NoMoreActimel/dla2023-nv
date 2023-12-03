import json
import torch
import numpy as np

from pathlib import Path

from utils import get_WaveGlow
import hw_nv.waveglow as waveglow


def move_batch_to_device(batch, device: torch.device):
    """
    Move all necessary tensors to the HPU
    """
    for tensor_for_gpu in ["mel", "wav"]:
        batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
    
    return batch


def run_inference(
        model,
        dataset,
        indices,
        inference_path,
        compute_losses=False,
        output_loss_filepath=None,
        criterion=None,
        dataset_type="train",
        epoch=None
    ):
    inference_path = Path(inference_path) / dataset_type
    inference_path.mkdir(exist_ok=True, parents=True)
    inference_paths = [inference_path / f"utterance_{ind}" for ind in indices]
    
    for i, path in enumerate(inference_paths):
        if epoch is not None:
            inference_paths[i] = path / f"epoch_{epoch}"
        inference_paths[i].mkdir(exist_ok=True, parents=True)
    
    if compute_losses:
        output_loss_filepath = Path(output_loss_filepath) / dataset_type
        output_loss_filepath.mkdir(exist_ok=True, parents=True)


    dataset_items = [dataset[ind] for ind in indices]
    batch = dataset.collate_fn(dataset_items)
    batch = move_batch_to_device(batch, device='cuda:0')

    with torch.no_grad():
        batch["wav_gen"] = model(**batch)
        batch["D_outputs"] = model.discriminate(wav=batch["wav"], wav_gen=batch["wav_gen"])
        
        if compute_losses:
            discriminator_losses = criterion["discriminator"](**batch)
            discriminator_loss_names = "discriminator_loss", "MPD_loss", "MSD_loss"
            for i, loss_name in enumerate(discriminator_loss_names):
                batch[loss_name] = discriminator_losses[i]

            generator_losses = criterion["generator"](**batch)
            generator_loss_names = "generator_loss", "GAN_loss", "mel_loss", "fm_loss"
            for i, loss_name in enumerate(generator_loss_names):
                batch[loss_name] = generator_losses[i]
            
            with open(output_loss_filepath, 'w', encoding='utf-8') as f:
                json.dump(output_loss_filepath, f, ensure_ascii=False, indent=4)    

    for i, (ind, wav, wav_gen) in enumerate(zip(indices, batch["wav"], batch["wav_gen"])):
        np.save(inference_paths[i] / f"wav_gen_{ind}.wav", wav_gen.cpu())
        np.save(inference_paths[i] / f"wav_{ind}.wav", wav.cpu())
        print(f"Saved utterance {ind} wav and wav_gen to {inference_paths[i]}")
