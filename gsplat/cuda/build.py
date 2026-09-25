# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build flags for the CUDA extension, shared by the JIT loader and setup.py.

Env vars: DEBUG=1, FAST_MATH=0, WITH_SYMBOLS=1 (adds -lineinfo for Nsight),
NVCC_FLAGS="...", NUM_CHANNELS="1,3,4", VERBOSE=1, MAX_JOBS=N.
"""

import json
import os
import shlex
import shutil
import time
from types import SimpleNamespace

import torch
import torch.utils.cpp_extension as jit
from torch.utils.cpp_extension import CUDA_HOME

PATH = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("DEBUG", "0") == "1"
FAST_MATH = os.getenv("FAST_MATH", "1") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "1" if DEBUG else "0") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", "")
NUM_CHANNELS = os.getenv("NUM_CHANNELS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"

# Every translation unit in the extension, in pipeline order.
SOURCES = [
    "ext.cpp",
    "csrc/Rendering.cpp",  # orchestration: project -> colour -> sort -> rasterize
    "csrc/Projection.cpp",
    "csrc/ProjectionEWA3DGSPacked.cu",
    "csrc/ProjectionEWA3DGSFused.cu",
    "csrc/SphericalHarmonics.cpp",
    "csrc/SphericalHarmonicsCUDA.cu",
    "csrc/Intersect.cpp",
    "csrc/IntersectTile.cu",
    "csrc/Rasterization.cpp",
    "csrc/RasterizeToPixels3DGSSerialBatchFwd.cu",
]


def get_build_parameters():
    include_paths = [
        os.path.join(PATH, "include"),
        os.path.join(PATH, "csrc", "third_party", "glm"),
    ]
    # CUDA installed into a conda env keeps its headers under targets/<arch>/include.
    if CUDA_HOME and os.path.isdir(os.path.join(CUDA_HOME, "targets")):
        for arch in os.listdir(os.path.join(CUDA_HOME, "targets")):
            p = os.path.join(CUDA_HOME, "targets", arch, "include")
            if os.path.isdir(p):
                include_paths.append(p)
                if os.path.isdir(os.path.join(p, "cccl")):  # CUDA >= 13
                    include_paths.append(os.path.join(p, "cccl"))

    cflags = ["-std=c++20", "-Wno-attributes", "-Wno-unknown-pragmas"]
    cflags += ["-g", "-O0"] if DEBUG else ["-O3", "-DNDEBUG"]
    if "backend: OpenMP" in torch.__config__.parallel_info():
        cflags += ["-DAT_PARALLEL_OPENMP", "-fopenmp"]

    cuda_cflags = ["--forward-unknown-opts", "--expt-relaxed-constexpr"]
    cuda_cflags += ["-use_fast_math"] if FAST_MATH else []
    cuda_cflags += ["-lineinfo"] if WITH_SYMBOLS else []
    # 3189: C++20 `module` keyword vs torch::python::module; 20012/186: glm noise.
    cuda_cflags += ["-diag-suppress", "3189,20012,186"]
    cuda_cflags += cflags
    if NUM_CHANNELS is not None:
        # nvcc needs the commas escaped; gcc does not.
        cuda_cflags += ["-DGSPLAT_NUM_CHANNELS=" + NUM_CHANNELS.replace(",", "\\,")]
        cflags += [f"-DGSPLAT_NUM_CHANNELS={NUM_CHANNELS}"]
    cuda_cflags += NVCC_FLAGS.split() if NVCC_FLAGS else []

    return SimpleNamespace(
        name="gsplat_cuda",
        extra_include_paths=include_paths,
        sources=[os.path.join(PATH, s) for s in SOURCES],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=[] if WITH_SYMBOLS else ["-s"],
    )


def _jit_cuda_cflags(cuda_cflags):
    # The JIT goes through a shell, so the escaped comma list needs quoting.
    return [
        shlex.quote(f) if f.startswith("-DGSPLAT_NUM_CHANNELS=") else f
        for f in cuda_cflags
    ]


def build_and_load_gsplat():
    """JIT-compile (or reuse) the extension under $TORCH_EXTENSIONS_DIR/gsplat_cuda."""
    params = get_build_parameters()
    build_dir = jit._get_build_directory(params.name, verbose=False)

    # A killed build can leave a stale lock behind.
    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass

    # Flag changes need a clean build; ninja only tracks source edits.
    params_file = os.path.join(build_dir, "build_params.json")
    if os.path.exists(params_file):
        with open(params_file) as f:
            if json.load(f) != params.__dict__:
                print("gsplat: build flags changed, rebuilding from scratch")
                shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    with open(params_file, "w") as f:
        json.dump(params.__dict__, f)

    fresh = not os.path.exists(os.path.join(build_dir, f"{params.name}.so"))
    if fresh:
        print(f"gsplat: compiling CUDA extension in {build_dir}")
    tic = time.time()
    os.environ.setdefault("NINJA_STATUS", "[%f/%t %r %es] ")
    try:
        module = jit.load(
            name=params.name,
            sources=params.sources,
            extra_cflags=params.extra_cflags,
            extra_cuda_cflags=_jit_cuda_cflags(params.extra_cuda_cflags),
            extra_include_paths=params.extra_include_paths,
            extra_ldflags=params.extra_ldflags,
            build_directory=build_dir,
            verbose=VERBOSE,
        )
    except OSError:
        # Another process holds the lock but the module is already built.
        module = jit._import_module_from_library(params.name, build_dir, True)
    if fresh:
        print(f"gsplat: CUDA extension ready in {time.time() - tic:.1f}s")
    return module
