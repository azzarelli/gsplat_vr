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

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Dispatch.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat
{
namespace
{
    // One camera's view of one gaussian.
    struct CameraProjection
    {
        vec2 mean2d;
        vec3 conic; // upper triangle of the inverse 2D covariance
        float depth;
        float opacity; // times the blur compensation when antialiased
        int32_t radius_x;
        int32_t radius_y;
    };
} // namespace

// One thread per gaussian, projected into all C cameras: the 3D covariance is
// built once and shared, each camera gets its own 2D projection and culling.
// Pass 1 (block_cnts) counts survivors per block and camera, plus gaussians
// kept by any camera. Pass 2 (block_accum, their inclusive cumsums) writes the
// survivors camera-major in ascending gaussian order, the union's ids, and
// each entry's slot in the union.
template<uint32_t C>
__global__ void projection_ewa_3dgs_packed_kernel(
    const uint32_t N,
    const float *__restrict__ means,     // [N, 3]
    const float *__restrict__ covars,    // [N, 6] optional, else quats + scales
    const float *__restrict__ quats,     // [N, 4]
    const float *__restrict__ scales,    // [N, 3]
    const float *__restrict__ opacities, // [N]
    const float *__restrict__ viewmats,  // [C, 4, 4]
    const float *__restrict__ Ks,        // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool antialiased,
    const bool with_union,
    // pass 1
    int32_t *__restrict__ block_cnts, // [C + with_union, n_blocks]
    // pass 2
    const int32_t *__restrict__ block_accum, // [C + with_union, n_blocks], cumsum per row
    int64_t *__restrict__ batch_ids,         // [nnz]
    int64_t *__restrict__ camera_ids,        // [nnz]
    int64_t *__restrict__ gaussian_ids,      // [nnz]
    int32_t *__restrict__ radii,             // [nnz, 2]
    float *__restrict__ means2d,             // [nnz, 2]
    float *__restrict__ depths,              // [nnz]
    float *__restrict__ conics,              // [nnz, 3]
    float *__restrict__ proj_opacities,      // [nnz]
    int64_t *__restrict__ union_ids,         // [U]
    int64_t *__restrict__ union_slots        // [nnz]
)
{
    const uint32_t n_blocks = gridDim.x;
    const uint32_t gid      = blockIdx.x * blockDim.x + threadIdx.x;
    const bool first_pass   = block_cnts != nullptr;

    CameraProjection proj[C];
    uint32_t visible = 0; // bit c: kept by camera c
    if(gid < N)
    {
        const vec3 mean_w = glm::make_vec3(means + gid * 3);
        mat3 covar;
        bool have_covar = false;
#pragma unroll
        for(uint32_t c = 0; c < C; ++c)
        {
            // glm is column-major but viewmats are row-major
            const float *vm = viewmats + c * 16;
            const mat3 R    = mat3(vm[0], vm[4], vm[8], vm[1], vm[5], vm[9], vm[2], vm[6], vm[10]);
            const vec3 t    = vec3(vm[3], vm[7], vm[11]);
            vec3 mean_c;
            posW2C(R, t, mean_w, mean_c);
            if(mean_c.z < near_plane || mean_c.z > far_plane)
            {
                continue;
            }

            if(!have_covar)
            {
                if(covars != nullptr)
                {
                    const float *cv = covars + gid * 6;
                    covar           = mat3(cv[0], cv[1], cv[2], cv[1], cv[3], cv[4], cv[2], cv[4], cv[5]);
                }
                else
                {
                    quat_scale_to_covar_preci(
                        glm::make_vec4(quats + gid * 4), glm::make_vec3(scales + gid * 3), &covar, nullptr
                    );
                }
                have_covar = true;
            }
            mat3 covar_c;
            covarW2C(R, covar, covar_c);

            const float *K = Ks + c * 9;
            mat2 covar2d;
            vec2 mean2d;
            persp_proj(mean_c, covar_c, K[0], K[4], K[2], K[5], image_width, image_height, covar2d, mean2d);

            float compensation;
            const float det = add_blur(eps2d, covar2d, compensation);
            if(det <= 0.f)
            {
                continue;
            }

            float opacity = opacities[gid];
            if(antialiased)
            {
                opacity *= compensation;
            }
            if(opacity < ALPHA_THRESHOLD)
            {
                continue;
            }
            // Opacity-aware tight bounding box, https://arxiv.org/pdf/2402.00525 Section B.2
            const float extend   = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
            const float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
            const float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));
            if(radius_x <= radius_clip && radius_y <= radius_clip)
            {
                continue;
            }
            if(mean2d.x + radius_x <= 0
               || mean2d.x - radius_x >= image_width
               || mean2d.y + radius_y <= 0
               || mean2d.y - radius_y >= image_height)
            {
                continue;
            }

            visible |= 1u << c;
            if(!first_pass)
            {
                const mat2 covar2d_inv = glm::inverse(covar2d);
                proj[c]                = {
                    mean2d,
                    vec3(covar2d_inv[0][0], covar2d_inv[0][1], covar2d_inv[1][1]),
                    mean_c.z,
                    opacity,
                    static_cast<int32_t>(radius_x),
                    static_cast<int32_t>(radius_y),
                };
            }
        }
    }

    if(first_pass)
    {
        using BlockReduce = cub::BlockReduce<int32_t, N_THREADS_PACKED>;
        __shared__ typename BlockReduce::TempStorage reduce_storage;
#pragma unroll
        for(uint32_t c = 0; c <= C; ++c)
        {
            if(c == C && !with_union)
            {
                break;
            }
            const int32_t keep  = c < C ? (visible >> c) & 1u : visible != 0;
            const int32_t count = BlockReduce(reduce_storage).Sum(keep);
            if(threadIdx.x == 0)
            {
                block_cnts[c * n_blocks + blockIdx.x] = count;
            }
            __syncthreads(); // before reduce_storage is reused
        }
        return;
    }

    using BlockScan = cub::BlockScan<int32_t, N_THREADS_PACKED>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    auto write_offset = [&](const uint32_t row, const int32_t keep)
    {
        int32_t rank;
        BlockScan(scan_storage).ExclusiveSum(keep, rank);
        __syncthreads(); // before scan_storage is reused
        // camera rows are cumsummed as one array, so this runs on across rows
        const uint32_t cell = row * n_blocks + blockIdx.x;
        const int32_t base  = cell == 0 ? 0 : block_accum[cell - 1];
        return static_cast<int64_t>(base) + rank;
    };

    int64_t slot = 0;
    if(with_union)
    {
        // union row is cumsummed on its own, so its first block starts at 0
        int32_t rank;
        BlockScan(scan_storage).ExclusiveSum(static_cast<int32_t>(visible != 0), rank);
        __syncthreads();
        const int32_t base = blockIdx.x == 0 ? 0 : block_accum[C * n_blocks + blockIdx.x - 1];
        slot               = static_cast<int64_t>(base) + rank;
        if(visible != 0)
        {
            union_ids[slot] = gid;
        }
    }

#pragma unroll
    for(uint32_t c = 0; c < C; ++c)
    {
        const bool keep = (visible >> c) & 1u;
        const int64_t e = write_offset(c, keep);
        if(!keep)
        {
            continue;
        }
        const CameraProjection &p = proj[c];
        batch_ids[e]              = 0;
        camera_ids[e]             = c;
        gaussian_ids[e]           = gid;
        radii[e * 2]              = p.radius_x;
        radii[e * 2 + 1]          = p.radius_y;
        means2d[e * 2]            = p.mean2d.x;
        means2d[e * 2 + 1]        = p.mean2d.y;
        depths[e]                 = p.depth;
        conics[e * 3]             = p.conic.x;
        conics[e * 3 + 1]         = p.conic.y;
        conics[e * 3 + 2]         = p.conic.z;
        proj_opacities[e]         = p.opacity;
        if(with_union)
        {
            union_slots[e] = slot;
        }
    }
}

void launch_projection_ewa_3dgs_packed_kernel(
    const at::Tensor means,
    const at::optional<at::Tensor> covars,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor opacities,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool antialiased,
    const bool with_union,
    at::optional<at::Tensor> block_cnts,
    at::optional<at::Tensor> block_accum,
    const ProjectionPackedResult *out
)
{
    const uint32_t N = means.size(0);
    const uint32_t C = viewmats.size(0);
    if(N == 0)
    {
        return;
    }
    const uint32_t n_blocks = (N + N_THREADS_PACKED - 1) / N_THREADS_PACKED;
    auto ptr                = [](const at::Tensor &t) { return t.defined() ? t.data_ptr() : nullptr; };

    const bool dispatched = dispatch::dispatch(
        dispatch::IntParam<1, 2, 3, 4>{static_cast<int>(C)},
        [&]<typename CamConst>()
        {
            projection_ewa_3dgs_packed_kernel<CamConst::value>
                <<<n_blocks, N_THREADS_PACKED, 0, at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    means.const_data_ptr<float>(),
                    covars.has_value() ? covars.value().const_data_ptr<float>() : nullptr,
                    quats.const_data_ptr<float>(),
                    scales.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    viewmats.const_data_ptr<float>(),
                    Ks.const_data_ptr<float>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    antialiased,
                    with_union,
                    block_cnts.has_value() ? block_cnts.value().data_ptr<int32_t>() : nullptr,
                    block_accum.has_value() ? block_accum.value().const_data_ptr<int32_t>() : nullptr,
                    out ? static_cast<int64_t *>(ptr(out->batch_ids)) : nullptr,
                    out ? static_cast<int64_t *>(ptr(out->camera_ids)) : nullptr,
                    out ? static_cast<int64_t *>(ptr(out->gaussian_ids)) : nullptr,
                    out ? static_cast<int32_t *>(ptr(out->radii)) : nullptr,
                    out ? static_cast<float *>(ptr(out->means2d)) : nullptr,
                    out ? static_cast<float *>(ptr(out->depths)) : nullptr,
                    out ? static_cast<float *>(ptr(out->conics)) : nullptr,
                    out ? static_cast<float *>(ptr(out->opacities)) : nullptr,
                    out ? static_cast<int64_t *>(ptr(out->union_ids)) : nullptr,
                    out ? static_cast<int64_t *>(ptr(out->union_slots)) : nullptr
                );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
    TORCH_CHECK(dispatched, "packed projection supports 1 to 4 cameras, got ", C);
}

} // namespace gsplat
