import sys
import os
import itertools
import numpy as np

from tqdm.auto import tqdm

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.sd
import comfy.model_management
import comfy.sample
from . import tiling

MAX_RESOLUTION=8192

def copy_cond(cond):
    return [(c1,c2.copy()) for c1,c2 in cond]

def slice_cond(tile_h, tile_h_len, tile_w, tile_w_len, cond, area):
    tile_h_end = tile_h + tile_h_len
    tile_w_end = tile_w + tile_w_len
    coords = area[0] #h_len, w_len, h, w,
    mask = area[1]
    if coords is not None:
        h_len, w_len, h, w = coords
        h_end = h + h_len
        w_end = w + w_len
        if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
            new_h = max(0, h - tile_h)
            new_w = max(0, w - tile_w)
            new_h_end = min(tile_h_end, h_end - tile_h)
            new_w_end = min(tile_w_end, w_end - tile_w)
            cond[1]['area'] = (new_h_end - new_h, new_w_end - new_w, new_h, new_w)
        else:
            return (cond, True)
    if mask is not None:
        new_mask = tiling.get_slice(mask, tile_h,tile_h_len,tile_w,tile_w_len)
        if new_mask.sum().cpu() == 0.0 and 'mask' in cond[1]:
            return (cond, True)
        else:
            cond[1]['mask'] = new_mask
    return (cond, False)

def slice_gligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen):
    tile_h_end = tile_h + tile_h_len
    tile_w_end = tile_w + tile_w_len
    if gligen is None:
        return
    gligen_type = gligen[0]
    gligen_model = gligen[1]
    gligen_areas = gligen[2]
    
    gligen_areas_new = []
    for emb, h_len, w_len, h, w in gligen_areas:
        h_end = h + h_len
        w_end = w + w_len
        if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
            new_h = max(0, h - tile_h)
            new_w = max(0, w - tile_w)
            new_h_end = min(tile_h_end, h_end - tile_h)
            new_w_end = min(tile_w_end, w_end - tile_w)
            gligen_areas_new.append((emb, new_h_end - new_h, new_w_end - new_w, new_h, new_w))

    if len(gligen_areas_new) == 0:
        del cond['gligen']
    else:
        cond['gligen'] = (gligen_type, gligen_model, gligen_areas_new)

def slice_cnet(h, h_len, w, w_len, model:comfy.sd.ControlNet, img):
    if img is None:
        img = model.cond_hint_original
    model.cond_hint = tiling.get_slice(img, h*8, h_len*8, w*8, w_len*8).to(model.control_model.dtype).to(model.device)

def slices_T2I(h, h_len, w, w_len, model:comfy.sd.T2IAdapter, img):
    model.control_input = None
    if img is None:
        img = model.cond_hint_original
    model.cond_hint = tiling.get_slice(img, h*8, h_len*8, w*8, w_len*8).float().to(model.device)

class TiledKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "tile_width": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
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


    def sample(self, model, add_noise, noise_seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        end_at_step = min(end_at_step, steps)
        device = comfy.model_management.get_torch_device()
        samples = latent_image["samples"]
        noise_mask = latent_image["noise_mask"] if "noise_mask" in latent_image else None
        force_full_denoise = return_with_leftover_noise == "enable"
        if add_noise == "disable":
            noise = torch.zeros(samples.size(), dtype=samples.dtype, layout=samples.layout, device="cpu")
        else:
            skip = latent_image["batch_index"] if "batch_index" in latent_image else 0
            noise = comfy.sample.prepare_noise(samples, noise_seed, skip)

        if noise_mask is not None:
            noise_mask = comfy.sample.prepare_mask(noise_mask, noise.shape, device)

        shape = samples.shape
        
        real_model = None
        comfy.model_management.load_model_gpu(model)
        real_model = model.model

        samples = samples.to(device)

        models = comfy.sample.load_additional_models(positive, negative)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

        if tiling_strategy != 'padded':
            if noise_mask is not None:
                samples += sampler.sigmas[start_at_step] * noise_mask * noise.to(device)
            else:
                samples += sampler.sigmas[start_at_step] * noise.to(device)
            

        #cnets
        cnets = [m for m in models if isinstance(m, comfy.sd.ControlNet)]
        cnet_imgs = [
            torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
            if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 else None
            for m in cnets]

        #T2I
        T2Is = [m for m in models if isinstance(m, comfy.sd.T2IAdapter)]
        T2I_imgs = [
            torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
            if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 or (m.channels_in == 1 and m.cond_hint_original.shape[1] != 1) else None
            for m in T2Is
        ]
        T2I_imgs = [
            torch.mean(img, 1, keepdim=True) if img is not None and m.channels_in == 1 and m.cond_hint_original.shape[1] else img
            for m, img in zip(T2Is, T2I_imgs)
        ]
        
        #cond area and mask
        spatial_conds_pos = [
            (c[1]['area'] if 'area' in c[1] else None, 
             comfy.sample.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
            for c in positive
        ]
        spatial_conds_neg = [
            (c[1]['area'] if 'area' in c[1] else None, 
             comfy.sample.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
            for c in negative
        ]

        #gligen
        gligen_pos = [
            c[1]['gligen'] if 'gligen' in c[1] else None
            for c in positive
        ]
        gligen_neg = [
            c[1]['gligen'] if 'gligen' in c[1] else None
            for c in negative
        ]

        positive_copy = comfy.sample.broadcast_cond(positive, shape[0], device)
        negative_copy = comfy.sample.broadcast_cond(negative, shape[0], device)

        gen = torch.manual_seed(noise_seed)
        if tiling_strategy == 'random':
            tiles = tiling.get_tiles_and_masks_rgrid(end_at_step - start_at_step, samples.shape, tile_height, tile_width, gen)
        elif tiling_strategy == 'padded':
            tiles = tiling.get_tiles_and_masks_padded(end_at_step - start_at_step, samples.shape, tile_height, tile_width)
        else:
            tiles = tiling.get_tiles_and_masks_simple(end_at_step - start_at_step, samples.shape, tile_height, tile_width)

        total_steps = sum([num_steps for img_pass in tiles for steps_list in img_pass for _,_,_,_,num_steps,_ in steps_list])
        current_step = [0]
        
        with tqdm(total=total_steps) as pbar_tqdm:
            pbar = comfy.utils.ProgressBar(total_steps)
            
            def callback(step, x0, x, total_steps):
                current_step[0] += 1
                pbar.update_absolute(current_step[0])
                pbar_tqdm.update(1)

            for img_pass in tiles:
                for i in range(len(img_pass)):
                    for tile_h, tile_h_len, tile_w, tile_w_len, tile_steps, tile_mask in img_pass[i]:
                    
                        #if we have masks get mask slices and see if we can skip slices
                        if noise_mask is not None or tile_mask is not None:
                            if noise_mask is not None:
                                tiled_mask = tiling.get_slice(noise_mask, tile_h, tile_h_len, tile_w, tile_w_len)
                                if tile_mask is not None:
                                    tiled_mask *= tile_mask.to(device)
                            else:
                                tiled_mask = tile_mask.to(device)
                            if tiled_mask.sum().cpu() == 0.0:
                                continue
                        else:
                            tiled_mask = None
                                
                        tiled_latent = tiling.get_slice(samples, tile_h, tile_h_len, tile_w, tile_w_len)
                        if tiling_strategy == 'padded':
                            tiled_noise = tiling.get_slice(noise, tile_h, tile_h_len, tile_w, tile_w_len).to(device)
                        else:
                            if tiled_mask is None:
                                tiled_noise = torch.zeros_like(tiled_latent)
                            else:
                                tiling.get_slice(noise, tile_h, tile_h_len, tile_w, tile_w_len).to(device) * (1 - tiled_mask)
                        
                        #TODO: all other condition based stuff like area sets and GLIGEN should also happen here

                        #cnets
                        for m, img in zip(cnets, cnet_imgs):
                            slice_cnet(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
                        
                        #T2I
                        for m, img in zip(T2Is, T2I_imgs):
                            slices_T2I(tile_h, tile_h_len, tile_w, tile_w_len, m, img)

                        pos = copy_cond(positive_copy)
                        neg = copy_cond(negative_copy)

                        #cond areas
                        pos = [slice_cond(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(pos, spatial_conds_pos)]
                        pos = [c for c, ignore in pos if not ignore]
                        neg = [slice_cond(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(neg, spatial_conds_neg)]
                        neg = [c for c, ignore in neg if not ignore]

                        #gligen
                        for (_, cond), gligen in zip(pos, gligen_pos):
                            slice_gligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)
                        for (_, cond), gligen in zip(neg, gligen_neg):
                            slice_gligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)

                        tile_result = sampler.sample(tiled_noise, pos, neg, cfg=cfg, latent_image=tiled_latent, start_step=start_at_step + i * tile_steps, last_step=start_at_step + i*tile_steps + tile_steps, force_full_denoise=force_full_denoise and i+1 == end_at_step - start_at_step, denoise_mask=tiled_mask, callback=callback, disable_pbar=True)
                        tiling.set_slice(samples, tile_result, tile_h, tile_h_len, tile_w, tile_w_len, tiled_mask)
                        

        comfy.sample.cleanup_additional_models(models)

        out = latent_image.copy()
        out["samples"] = samples.cpu()
        return (out, )
    

    
NODE_CLASS_MAPPINGS = {
    "BNK_TiledKSamplerAdvanced": TiledKSamplerAdvanced,
}