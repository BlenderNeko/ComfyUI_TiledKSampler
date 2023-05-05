import torch
import itertools
import numpy as np

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def create_batches(n, iterable):
    groups = itertools.groupby(iterable, key= lambda x: (x[1], x[3]))
    for _, x in groups:
        for y in grouper(n, x):
            yield y


def get_slice(tensor, h, h_len, w, w_len):
    t = tensor.narrow(-2, h, h_len)
    t = t.narrow(-1, w, w_len)
    return t

def set_slice(tensor1,tensor2, slices, masks=None):
    for i in range(len(slices)):
        h = slices[i][0]
        h_len = slices[i][1]
        w = slices[i][2]
        w_len = slices[i][3]
        b_size = tensor1.shape[0]
        if masks is not None:
            tensor1[:,:,h:h+h_len,w:w+w_len] = tensor1[:,:,h:h+h_len,w:w+w_len] * ( 1 - masks[i*b_size:(i+1)*b_size]) +  tensor2[i*b_size:(i+1)*b_size] * masks[i*b_size:(i+1)*b_size]
        else:
            tensor1[:,:,h:h+h_len,w:w+w_len] = tensor2[i*b_size:(i+1)*b_size]

def get_tiles_and_masks_rgrid(steps, latent_shape, tile_height, tile_width, padding, max_batches, generator, device):

    def calc_coords(latent_size, tile_size, jitter):
        tile_coords = int((latent_size + jitter - 1) // tile_size + 1)
        tile_coords = [np.clip(tile_size * c - jitter, 0, latent_size) for c in range(tile_coords + 1)]
        tile_coords = [(c1, c2-c1) for c1, c2 in zip(tile_coords, tile_coords[1:])]
        return tile_coords
    
    #calc stuff
    batch_size = latent_shape[0]
    latent_size_h = latent_shape[-2]
    latent_size_w = latent_shape[-1]
    tile_size_h = int(tile_height // 8)
    tile_size_w = int(tile_width // 8)

    tiles_all = []

    for s in range(steps):
        rands = torch.rand((2,), dtype=torch.float32, generator=generator, device='cpu').numpy()

        jitter_w1 = int(rands[0] * tile_size_w)
        jitter_w2 = int(((rands[0] + .5) % 1.0) * tile_size_w)
        jitter_h1 = int(rands[1] * tile_size_h)
        jitter_h2 = int(((rands[1] + .5) % 1.0) * tile_size_h)

        #calc number of tiles
        tiles_h = [
            calc_coords(latent_size_h, tile_size_h, jitter_h1),
            calc_coords(latent_size_h, tile_size_h, jitter_h2)
        ]
        tiles_w = [
            calc_coords(latent_size_w, tile_size_w, jitter_w1),
            calc_coords(latent_size_w, tile_size_w, jitter_w2)
        ]

        tiles = []
        if s % 2 == 0:
            for i, h in enumerate(tiles_h[0]):
                for w in tiles_w[i%2]:
                    tiles.append((h[0], h[1], w[0], w[1], None))
        else:
            for i, w in enumerate(tiles_w[0]):
                for h in tiles_h[i%2]:
                    tiles.append((h[0], h[1], w[0], w[1], None))
        tiles = list(create_batches(max_batches, tiles))
        tiles_all.append(tiles)
    return tiles_all