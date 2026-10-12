# gsplat (stereo-lean)

A forward-only trim of [gsplat](https://github.com/nerfstudio-project/gsplat)
1.6.0 trunk, keeping only whats necessary for rasterizing 3D-GS representations that are rendered out as GL_Textures. 

- Some shortcuts have been made to render to VR headsets (2x 4k images, left and right)
- The changes are mostly to do with hand-offs with moderngl and handling simplifications by pre-baking paramters that don't require back-prop so their use can be binned for pre-built or built-on-init parameters.
- Much faster than [gsplat](https://github.com/nerfstudio-project/gsplat) and [VRSplat](https://github.com/Cekavis/VRSplat); for an average 7mil in-frustum render load, on a 4090, for 2x 4K RGB+D+A renders: gsplat1.6.0=(24.18ms+-4ms), VRSplat=(42.56ms+-5ms) and Ours=(13.48ms+-1ms).


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
