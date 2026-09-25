/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <ATen/Functions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <mutex>
#include <utility>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat
{
namespace
{
    // Surfaces are made per call, since a mapped GL texture's array can change
    // between maps, and destroyed once the kernel that used them has finished.
    std::mutex retired_mutex;
    std::vector<std::pair<cudaEvent_t, std::vector<cudaSurfaceObject_t>>> retired;

    void destroy_finished_surfaces()
    {
        auto done = retired.begin();
        while(done != retired.end())
        {
            const cudaError_t state = cudaEventQuery(done->first);
            if(state == cudaErrorNotReady)
            {
                (void)cudaGetLastError(); // not an error, but it lingers as one
                break;
            }
            C10_CUDA_CHECK(state);
            for(cudaSurfaceObject_t surface : done->second)
            {
                cudaDestroySurfaceObject(surface);
            }
            cudaEventDestroy(done->first);
            ++done;
        }
        retired.erase(retired.begin(), done);
    }

    void check_target(int64_t handle, int64_t width, int64_t height, int channels)
    {
        cudaChannelFormatDesc desc;
        cudaExtent extent;
        unsigned int flags;
        C10_CUDA_CHECK(cudaArrayGetInfo(&desc, &extent, &flags, reinterpret_cast<cudaArray_t>(handle)));
        const bool is_float = desc.f == cudaChannelFormatKindFloat && desc.x == 32 && desc.y == (channels > 1 ? 32 : 0)
                              && desc.z == (channels > 2 ? 32 : 0) && desc.w == (channels > 3 ? 32 : 0);
        TORCH_CHECK(is_float, "target must be a ", channels == 4 ? "RGBA32F" : "R32F", " array");
        TORCH_CHECK(
            static_cast<int64_t>(extent.width) == width && static_cast<int64_t>(extent.height) == height,
            "target is ",
            extent.width,
            "x",
            extent.height,
            ", image is ",
            width,
            "x",
            height
        );
        TORCH_CHECK(flags & cudaArraySurfaceLoadStore, "target needs surface load/store: register the GL texture with SURFACE_LDST");
    }

    cudaSurfaceObject_t make_surface(int64_t handle)
    {
        cudaResourceDesc res{};
        res.resType         = cudaResourceTypeArray;
        res.res.array.array = reinterpret_cast<cudaArray_t>(handle);
        cudaSurfaceObject_t surface;
        const cudaError_t err = cudaCreateSurfaceObject(&surface, &res);
        TORCH_CHECK(err == cudaSuccess, "cudaCreateSurfaceObject failed: ", cudaGetErrorString(err));
        return surface;
    }
} // namespace

RasterizeResult rasterize_to_pixels_3dgs(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &isect_offsets,
    const at::Tensor &flatten_ids,
    bool expected_depth,
    const std::vector<int64_t> &color_targets,
    const std::vector<int64_t> &alpha_targets
)
{
    DEVICE_GUARD(means2d);
    TORCH_CHECK(tile_size == 4 || tile_size == 16, "tile_size must be 4 or 16, got ", tile_size);
    TORCH_CHECK(
        isect_offsets.size(-2) * tile_size >= image_height && isect_offsets.size(-1) * tile_size >= image_width,
        "tile grid does not cover the image"
    );
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(isect_offsets);
    CHECK_INPUT(flatten_ids);
    if(backgrounds.has_value())
    {
        CHECK_INPUT(backgrounds.value());
    }

    const int64_t I = isect_offsets.numel() / (isect_offsets.size(-2) * isect_offsets.size(-1));
    SurfaceTargets targets{};
    at::Tensor renders, alphas;
    if(!color_targets.empty())
    {
        TORCH_CHECK(
            static_cast<int64_t>(color_targets.size()) == I && static_cast<int64_t>(alpha_targets.size()) == I,
            "need one colour and one alpha target per image (",
            I,
            ")"
        );
        TORCH_CHECK(I <= SurfaceTargets::kMax, "at most ", SurfaceTargets::kMax, " images can write to targets");
        for(int64_t i = 0; i < I; ++i)
        {
            check_target(color_targets[i], image_width, image_height, 4);
            check_target(alpha_targets[i], image_width, image_height, 1);
        }
        std::lock_guard<std::mutex> lock(retired_mutex);
        destroy_finished_surfaces();
        for(int64_t i = 0; i < I; ++i)
        {
            targets.color[i] = make_surface(color_targets[i]);
            targets.alpha[i] = make_surface(alpha_targets[i]);
        }
        targets.enabled = true;
    }
    else
    {
        auto opt = means2d.options();
        at::DimVector image_dims(isect_offsets.sizes().slice(0, isect_offsets.dim() - 2));
        auto with = [&](std::initializer_list<int64_t> tail)
        {
            at::DimVector s(image_dims);
            s.append(tail);
            return s;
        };
        renders = at::empty(with({image_height, image_width, colors.size(-1)}), opt);
        alphas  = at::empty(with({image_height, image_width, 1}), opt);
    }

    launch_rasterize_to_pixels_3dgs_fwd_kernel(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        c10::nullopt, // masks
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        expected_depth,
        renders,
        alphas,
        targets
    );

    if(targets.enabled)
    {
        cudaEvent_t done;
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(done, c10::cuda::getCurrentCUDAStream()));
        std::vector<cudaSurfaceObject_t> used;
        for(int64_t i = 0; i < I; ++i)
        {
            used.push_back(targets.color[i]);
            used.push_back(targets.alpha[i]);
        }
        std::lock_guard<std::mutex> lock(retired_mutex);
        retired.emplace_back(done, std::move(used));
    }
    return {.renders = renders, .alphas = alphas};
}
} // namespace gsplat
