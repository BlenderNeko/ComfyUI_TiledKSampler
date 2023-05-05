import sys
import os

from tqdm.auto import trange

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.model_management
import comfy.sample
from . import tiling

MAX_RESOLUTION=8192

def expand_cond(cond, batch):
    copy = []
    for p in cond:
        t = p[0]
        t = t.expand([batch] + ([-1] * (len(t.shape) -1)))
        copy += [[t] + p[1:]]
    return copy


class TiledKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "tile_width": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "concurrent_tiles": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"


    def sample(self, model, add_noise, noise_seed, tile_width, tile_height, concurrent_tiles, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        end_at_step = min(end_at_step, steps + start_at_step)
        device = comfy.model_management.get_torch_device()
        samples = latent_image["samples"]
        noise_mask = latent_image["noise_mask"] if "noise_mask" in latent_image else None
        force_full_denoise = return_with_leftover_noise == "enable"
        if add_noise == "disabled":
            noise = torch.zeros(samples.size(), dtype=samples.dtype, layout=samples.layout, device="cpu")
        else:
            skip = latent_image["batch_index"] if "batch_index" in latent_image else 0
            noise = comfy.sample.prepare_noise(samples, noise_seed, skip)

        if noise_mask is not None:
            noise_mask = comfy.sample.prepare_mask(noise_mask, noise.shape, device)

        gen = torch.manual_seed(noise_seed)
        tiles = tiling.get_tiles_and_masks_rgrid(steps, samples.shape, tile_height, tile_width, 0, concurrent_tiles, gen, device)
        
        real_model = None
        comfy.model_management.load_model_gpu(model)
        real_model = model.model

        noise = noise.to(device)
        samples = samples.to(device)

        positive_copy = comfy.sample.broadcast_cond(positive, noise.shape[0], device)
        negative_copy = comfy.sample.broadcast_cond(negative, noise.shape[0], device)

        models = comfy.sample.load_additional_models(positive, negative)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        if noise_mask is not None:
            samples += sampler.sigmas[start_at_step] * noise_mask * noise
        else:
            samples += sampler.sigmas[start_at_step] * noise
        noise_tile = torch.zeros(samples.shape[:2] + (tile_height // 8, tile_width//8,), dtype=samples.dtype, device=device)

        steps_per_tile = 1
        cycle = len(tiles)
        masks = None

        total_steps = (end_at_step - start_at_step) // steps_per_tile
        pbar = comfy.utils.ProgressBar(total_steps)
        for i in trange(total_steps):
            for tiles_batch in tiles[i%cycle]:
                #get latent slices
                slices = [tiling.get_slice(samples, x1,x2,y1,y2) for x1,x2,y1,y2,_ in tiles_batch]
                
                #if we have mask get mask slices and see if we can skip slices
                if noise_mask is not None:
                    tile_masks = [tiling.get_slice(noise_mask, x1,x2,y1,y2) for x1,x2,y1,y2,_ in tiles_batch]
                    can_skip = [x.sum().cpu() == 0 for x in tile_masks]
                    if all(can_skip):
                        continue
                    tile_masks = [x for x,y in zip(tile_masks, can_skip) if not y]
                    slices =[x for x,y in zip(slices, can_skip) if not y]
                    tiles_batch =  [x for x,y in zip(tiles_batch, can_skip) if not y]
                    masks = torch.concat(tile_masks)

                num_slices = len(slices)
                latent_tiles = torch.concat(slices)
                noise_tile = torch.zeros_like(latent_tiles)
                
                #TODO: all other condition based stuff like area sets and GLIGEN should also happen here
                pos = expand_cond(positive_copy, num_slices)
                neg = expand_cond(negative_copy, num_slices)

                tile_result = sampler.sample(noise_tile, pos, neg, cfg=cfg, latent_image=latent_tiles, start_step=start_at_step + i * steps_per_tile, last_step=start_at_step + i*steps_per_tile + steps_per_tile, force_full_denoise=force_full_denoise and i+1 == end_at_step - start_at_step, denoise_mask=masks, disable_pbar=True)
                tiling.set_slice(samples, tile_result, [(x1,x2,y1,y2) for x1,x2,y1,y2,_ in tiles_batch], masks)
            pbar.update_absolute(i + 1, total_steps)
        comfy.sample.cleanup_additional_models(models)

        out = latent_image.copy()
        out["samples"] = samples.cpu()
        return (out, )
    

    
NODE_CLASS_MAPPINGS = {
    "BNK_TiledKSamplerAdvanced": TiledKSamplerAdvanced,
}