# SPDX-FileCopyrightText: Copyright 2023-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""pip install -e . --no-build-isolation   (BUILD_NO_CUDA=1: skip the
ahead-of-time build and let gsplat JIT-compile on first use instead)."""

import importlib.util
import os

from setuptools import find_packages, setup

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"

__version__ = None
with open(os.path.join(HERE, "gsplat", "version.py")) as f:
    exec(f.read())


def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension

    # Load build.py by path: importing gsplat here would need the extension
    # that is being built.
    spec = importlib.util.spec_from_file_location(
        "gsplat_cuda_build", os.path.join(HERE, "gsplat", "cuda", "build.py")
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    p = build.get_build_parameters()
    return [
        CUDAExtension(
            "gsplat.csrc",
            sources=[os.path.relpath(s, HERE) for s in p.sources],
            include_dirs=p.extra_include_paths,
            extra_compile_args={"cxx": p.extra_cflags, "nvcc": p.extra_cuda_cflags},
            extra_link_args=p.extra_ldflags,
        )
    ]


def get_cmdclass():
    from torch.utils.cpp_extension import BuildExtension

    return {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)}


setup(
    name="gsplat",
    version=__version__,
    description="Forward-only 3D Gaussian rasterizer (stereo-focused trim of gsplat)",
    url="https://github.com/nerfstudio-project/gsplat",
    python_requires=">=3.8",
    install_requires=["ninja", "torch>=2.7"],
    ext_modules=[] if BUILD_NO_CUDA else get_extensions(),
    cmdclass={} if BUILD_NO_CUDA else get_cmdclass(),
    packages=find_packages(),
    include_package_data=True,
)
