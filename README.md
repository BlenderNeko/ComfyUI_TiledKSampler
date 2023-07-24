# Tiled sampling for ComfyUI

![panorama of the ocean, sailboats and large moody clouds](https://github.com/BlenderNeko/ComfyUI_TiledKSampler/blob/master/examples/ComfyUI_02010_.png)

this repo contains a tiled sampler for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). It allows for denoising larger images by splitting it up into smaller tiles and denoising these. It tries to minimize any seams for showing up in the end result by gradually denoising all tiles one step at the time and randomizing tile positions for every step.

### settings

The tiled samplers comes with some additional settings to further control it's behavior:

- **tile_width**: the width of the tiles.
- **tile_height**: the height of the tiles.
- **tiling_strategy**: how to do the tiling

## Tiling strategies

### random:
The random tiling strategy aims to reduce the presence of seams as much as possible by slowly denoising the entire image step by step, randomizing the tile positions for each step. It does this by alternating between horizontal and vertical brick patterns, randomly offsetting the pattern each time. As the number of steps grows to infinity the strength of seams shrinks to zero. Although this random offset eliminates seams, it comes at the cost of additional overhead per step and makes this strategy incompatible with uni samplers.

<details>
<summary>
visual explanation
</summary>

![gif showing of the random brick tiling](https://github.com/BlenderNeko/ComfyUI_TiledKSampler/blob/master/examples/tiled_random.gif)
</details>

<details>
<summary>
example seamless image
</summary>

This tiling strategy is exceptionally good in hiding seams, even when starting off from complete noise, repetitions are visible but seams are not.

![gif showing of the random brick tiling](https://github.com/BlenderNeko/ComfyUI_TiledKSampler/blob/master/examples/ComfyUI_02006_.png)
</details>

### random strict:

One downside of random is that it can unfavorably crop border tiles, random strict uses masking to ensure no border tiles have to be cropped. This tiling strategy does not play nice with the SDE sampler.

### padded:

The padded tiling strategy tries to reduce seams by giving each tile more context of its surroundings through padding. It does this by further dividing each tile into 9 smaller tiles, which are denoised in such a way that a tile is always surrounded by static contex during denoising. This strategy is more prone to seams but because the location of the tiles is static, this strategy is compatible with uni samplers and has no overhead between steps. However the padding makes it so that up to 4 times as many tiles have to be denoised.

<details>
<summary>
visual explanation
</summary>

![gif showing of padded tiling](https://github.com/BlenderNeko/ComfyUI_TiledKSampler/blob/master/examples/tiled_padding.gif)
</details>

### simple

The simple tiling strategy divides the image into a static grid of tiles and denoises these one by one.

### roadmap:

 - [x] latent masks
 - [x] image wide control nets
 - [x] T2I adaptors
 - [ ] tile wide control nets and T2I adaptors (e.g. style models)
 - [x] area conditioning
 - [x] area mask conditioning
 - [x] GLIGEN
