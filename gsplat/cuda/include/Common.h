/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ATen/DeviceGuard.h>
#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>

namespace gsplat
{
//
// Some Macros.
//
#define CHECK_DEVICE(x)     TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_DEVICE(x);   \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten) const at::OptionalDeviceGuard device_guard(device_of(_ten));

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...)                                                    \
    do                                                                            \
    {                                                                             \
        size_t temp_storage_bytes = 0;                                            \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                           \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();      \
        auto temp_storage       = caching_allocator.allocate(temp_storage_bytes); \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);                \
    } while(false)

//
// Convenience typedefs for CUDA types
//
using vec2   = glm::vec<2, float>;
using vec3   = glm::vec<3, float>;
using vec4   = glm::vec<4, float>;
using mat2   = glm::mat<2, 2, float>;
using mat3   = glm::mat<3, 3, float>;
using mat4   = glm::mat<4, 4, float>;
using mat3x2 = glm::mat<3, 2, float>;

// Only PINHOLE is used; the projection kernels still switch on the others.
enum CameraModelType
{
    PINHOLE = 0,
    ORTHO   = 1,
    FISHEYE = 2,
};

#define N_THREADS_PACKED 256

// CUDA caps grid.y (and grid.z) at 65535; only grid.x reaches 2^31 - 1. Kernels
// that map a batch (or batch-camera) dimension onto grid.y must reject launches
// that would exceed this.
constexpr uint32_t kMaxCudaGridDimY = 65535;

#define ALPHA_THRESHOLD         (1.f / 255.f)
// GAUSSIAN_EXTEND determines where the gaussian is truncated in standard deviations."
#define GAUSSIAN_EXTEND         3.33f
// A maximal-opacity Gaussian has to be hit twice to reach the threshold:
// TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
#define MAX_ALPHA               0.99f
#define TRANSMITTANCE_THRESHOLD 1e-4f

// Floor for the antialiased compensation factor (sqrt(det_orig / det_blur)).
// Prevents compensation from reaching zero for extremely small Gaussians.
#define MIN_COMPENSATION 0.005f
} // namespace gsplat
