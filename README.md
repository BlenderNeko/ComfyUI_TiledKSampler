# WIP tiled sampling for ComfyUI

this repo contains a tiled sampler for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). It allows for denoising larger images by splitting it up into smaller tiles and denoising these. It tries to minimize any seams for showing up in the end result by gradually denoising all tiles one step at the time and randomizing tile positions for every step.

### settings

The tiled sampler comes with some additional settings to further control it's behavior:

- **tile_width**: the width of the tiles.
- **tile_height**: the height of the tiles.
- **concurrent_tiles**: determines how many tiles to try and denoise concurrently.

If results look tiled, it might help to increase the number of steps and to use an ancestral sampler

roadmap:

 - [x] latent masks
 - [x] image wide control nets
 - [x] T2I adaptors (requires forked version of comfy for now)
 - [ ] tile wide control nets and T2I adaptors (e.g. style models)
 - [ ] area conditioning
 - [ ] area mask conditioning
 - [ ] GLIGEN
