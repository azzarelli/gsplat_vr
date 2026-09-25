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

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat
{
namespace cg = cooperative_groups;

template<typename scalar_t>
__global__ void projection_ewa_3dgs_fused_fwd_kernel(
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,     // [B, N, 3]
    const scalar_t *__restrict__ covars,    // [B, N, 6] optional
    const scalar_t *__restrict__ quats,     // [B, N, 4] optional
    const scalar_t *__restrict__ scales,    // [B, N, 3] optional
    const scalar_t *__restrict__ opacities, // [B, N] optional
    const scalar_t *__restrict__ viewmats,  // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,        // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,         // [B, C, N, 2]
    scalar_t *__restrict__ means2d,      // [B, C, N, 2]
    scalar_t *__restrict__ depths,       // [B, C, N]
    scalar_t *__restrict__ conics,       // [B, C, N, 3]
    scalar_t *__restrict__ compensations // [B, C, N] optional
)
{
    // parallelize over B * C * gaussian_count.
    int64_t idx         = cg::this_grid().thread_rank();
    const int64_t count = static_cast<int64_t>(B) * C * gaussian_count;
    if(idx >= count)
    {
        return;
    }
    const uint32_t bid       = idx / (C * gaussian_count);             // batch id
    const uint32_t cid       = (idx / gaussian_count) % C;             // camera id
    const uint32_t gid       = idx % gaussian_count + gaussian_offset; // gaussian id
    const int64_t global_idx = (static_cast<int64_t>(bid) * C + cid) * N + gid;

    // shift pointers to the current camera and gaussian
    means    += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    Ks       += bid * C * 9 + cid * 9;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    if(mean_c.z < near_plane || mean_c.z > far_plane)
    {
        radii[global_idx * 2]     = 0;
        radii[global_idx * 2 + 1] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3 covar;
    if(covars != nullptr)
    {
        covars += bid * N * 6 + gid * 6;
        covar   = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    }
    else
    {
        // compute from quaternions and scales
        quats  += bid * N * 4 + gid * 4;
        scales += bid * N * 3 + gid * 3;
        quat_scale_to_covar_preci(glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr);
    }
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch(camera_model)
    {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5], image_width, image_height, covar2d, mean2d);
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5], image_width, image_height, covar2d, mean2d);
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5], image_width, image_height, covar2d, mean2d);
        break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if(det <= 0.f)
    {
        radii[global_idx * 2]     = 0;
        radii[global_idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = GAUSSIAN_EXTEND;
    if(opacities != nullptr)
    {
        float opacity = opacities[bid * N + gid];
        if(compensations != nullptr)
        {
            // we assume compensation term will be applied later on.
            opacity *= compensation;
        }
        if(opacity < ALPHA_THRESHOLD)
        {
            radii[global_idx * 2]     = 0;
            radii[global_idx * 2 + 1] = 0;
            return;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

    if(radius_x <= radius_clip && radius_y <= radius_clip)
    {
        radii[global_idx * 2]     = 0;
        radii[global_idx * 2 + 1] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if(mean2d.x + radius_x <= 0
       || mean2d.x - radius_x >= image_width
       || mean2d.y + radius_y <= 0
       || mean2d.y - radius_y >= image_height)
    {
        radii[global_idx * 2]     = 0;
        radii[global_idx * 2 + 1] = 0;
        return;
    }

    // write to outputs
    radii[global_idx * 2]       = (int32_t)radius_x;
    radii[global_idx * 2 + 1]   = (int32_t)radius_y;
    means2d[global_idx * 2]     = mean2d.x;
    means2d[global_idx * 2 + 1] = mean2d.y;
    depths[global_idx]          = mean_c.z;
    conics[global_idx * 3]      = covar2d_inv[0][0];
    conics[global_idx * 3 + 1]  = covar2d_inv[0][1];
    conics[global_idx * 3 + 2]  = covar2d_inv[1][1];
    if(compensations != nullptr)
    {
        compensations[global_idx] = compensation;
    }
}

void launch_projection_ewa_3dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,                   // [..., N, 3]
    const at::optional<at::Tensor> covars,    // [..., N, 6] optional
    const at::optional<at::Tensor> quats,     // [..., N, 4] optional
    const at::optional<at::Tensor> scales,    // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,                // [..., C, 4, 4]
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                      // [..., C, N, 2]
    at::Tensor means2d,                    // [..., C, N, 2]
    at::Tensor depths,                     // [..., C, N]
    at::Tensor conics,                     // [..., C, N, 3]
    at::optional<at::Tensor> compensations // [..., C, N] optional
)
{
    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t C = viewmats.size(-3);       // number of cameras
    uint32_t B = means.numel() / (N * 3); // number of batches

    int64_t n_elements             = B * C * N;
    constexpr unsigned int threads = 256;
    unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_fused_fwd_kernel",
        [&]()
        {
            projection_ewa_3dgs_fused_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                0,
                N,
                B,
                C,
                N,
                means.const_data_ptr<scalar_t>(),
                covars.has_value() ? covars.value().const_data_ptr<scalar_t>() : nullptr,
                quats.has_value() ? quats.value().const_data_ptr<scalar_t>() : nullptr,
                scales.has_value() ? scales.value().const_data_ptr<scalar_t>() : nullptr,
                opacities.has_value() ? opacities.value().const_data_ptr<scalar_t>() : nullptr,
                viewmats.const_data_ptr<scalar_t>(),
                Ks.const_data_ptr<scalar_t>(),
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                camera_model,
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<scalar_t>(),
                depths.data_ptr<scalar_t>(),
                conics.data_ptr<scalar_t>(),
                compensations.has_value() ? compensations.value().data_ptr<scalar_t>() : nullptr
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}

} // namespace gsplat

