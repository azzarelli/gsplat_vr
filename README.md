# gsplat (stereo-lean)

A forward-only trim of [gsplat](https://github.com/nerfstudio-project/gsplat)
1.6.0 trunk, kept to what real-time stereo rendering of 3DGS needs. The full
library (training, backward passes, 2DGS, 3DGUT, lidar, sparse, multi-GPU)
is on the `uwgs-fork` branch.

Output is bit-identical to `uwgs-fork` for the kept paths.

## API

```python
from gsplat import rasterization

colors, alphas, meta = rasterization(
    means, quats, scales, opacities, sh,  # [N,3] [N,4] [N,3] [N] [N,K,3]
    viewmats, Ks, width, height,          # [C,4,4] [C,3,3]
    sh_degree=3, render_mode="RGB+ED",
    packed=True, stereo=True,             # stereo: shared SH for C == 2
)
```

## Pipeline

`gsplat/rendering.py` calls one C++ function, `rasterization_3dgs` in
`gsplat/cuda/csrc/Rendering.cpp`, which runs:

| Stage | Host | Kernel |
|---|---|---|
| 1. Project (EWA, pinhole) | `Projection.cpp` | `ProjectionEWA3DGSPacked.cu` (packed), `ProjectionEWA3DGSFused.cu` (dense) |
| 2. SH -> RGB (+ depth channel) | `SphericalHarmonics.cpp` | `SphericalHarmonicsCUDA.cu` (per-entry SH; fused SH+depth for dense) |
| 3. Tile intersect + radix sort | `Intersect.cpp` | `IntersectTile.cu` |
| 4. Rasterize (alpha compositing) | `Rasterization.cpp` | `RasterizeToPixels3DGSSerialBatchFwd.cu` |
| 5. Expected depth | `Rendering.cpp` | ATen |

Shared device code: `include/Common.h` (types, constants), `include/Utils.cuh`
(projection math), `include/Dispatch.h` (compile-time dispatch),
`csrc/RasterizeToPixels3DGSDevice.cuh`, `csrc/SphericalHarmonics.cuh`.

`packed=True` keeps only surviving (camera, gaussian) pairs; `packed=False`
keeps all of them, colours them in one fused kernel and can sort per image
(`segmented=True`). Stages 1 (packed) and 3 each read a count back to the host.

## Build

The editable install JIT-compiles on first import into
`$TORCH_EXTENSIONS_DIR/gsplat_cuda` (about 30 s):

```bash
pip install -e . --no-build-isolation   # or BUILD_NO_CUDA=1 to skip the AOT build
VERBOSE=1 python -c "from gsplat.cuda._backend import _C"
```

Build knobs (see `gsplat/cuda/build.py`): `NUM_CHANNELS="1,3,4"` (rasterizer
channel counts, default in `csrc/Config.h`), `WITH_SYMBOLS=1` (`-lineinfo`
for Nsight), `DEBUG=1`, `FAST_MATH=0`, `NVCC_FLAGS`.

## License

Apache-2.0, see `LICENSE`. Original work by the Nerfstudio team and NVIDIA;
please cite gsplat (`CITATION.bib`).
