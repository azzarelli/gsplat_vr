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
#include <cub/cub.cuh>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat
{
namespace cg = cooperative_groups;

template<typename scalar_t>
__global__ void projection_ewa_3dgs_packed_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,     // [B, N, 3]
    const scalar_t *__restrict__ covars,    // [B, N, 6] Optional
    const scalar_t *__restrict__ quats,     // [B, N, 4] Optional
    const scalar_t *__restrict__ scales,    // [B, N, 3] Optional
    const scalar_t *__restrict__ opacities, // [B, N] optional
    const scalar_t *__restrict__ viewmats,  // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,        // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const int32_t *__restrict__ block_accum, // [B * C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    const bool antialiased,
    // outputs
    int32_t *__restrict__ block_cnts,    // [B * C * blocks_per_row] packing helper
    int32_t *__restrict__ indptr,        // [B * C + 1]
    int64_t *__restrict__ batch_ids,     // [nnz]
    int64_t *__restrict__ camera_ids,    // [nnz]
    int64_t *__restrict__ gaussian_ids,  // [nnz]
    int32_t *__restrict__ radii,         // [nnz, 2]
    scalar_t *__restrict__ means2d,      // [nnz, 2]
    scalar_t *__restrict__ depths,       // [nnz]
    scalar_t *__restrict__ conics,       // [nnz, 3]
    scalar_t *__restrict__ compensations // [nnz] optional
)
{
    int32_t blocks_per_row = gridDim.x;
    int32_t row_idx        = blockIdx.y;
    int32_t block_col_idx  = blockIdx.x;
    int32_t block_idx      = row_idx * blocks_per_row + block_col_idx;
    int32_t col_idx        = block_col_idx * blockDim.x + threadIdx.x;
    const int32_t bid      = row_idx / C;
    const int32_t cid      = row_idx % C;
    const int32_t gid      = col_idx;

    bool valid = (bid < B) && (cid < C) && (gid < N);

    // check if points are with camera near and far plane
    vec3 mean_c;
    mat3 R;
    if(valid)
    {
        // shift pointers to the current camera and gaussian
        means    += bid * N * 3 + gid * 3;
        viewmats += bid * C * 16 + cid * 16;

        // glm is column-major but input is row-major
        R = mat3(
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
        posW2C(R, t, glm::make_vec3(means), mean_c);
        if(mean_c.z < near_plane || mean_c.z > far_plane)
        {
            valid = false;
        }
    }

    // check if the perspective projection is valid.
    mat2 covar2d;
    vec2 mean2d;
    mat2 covar2d_inv;
    float compensation;
    float det;
    if(valid)
    {
        // transform Gaussian covariance to camera space
        mat3 covar;
        if(covars != nullptr)
        {
            // if a precomputed covariance is provided
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
            // if not then compute it from quaternions and scales
            quats  += bid * N * 4 + gid * 4;
            scales += bid * N * 3 + gid * 3;
            quat_scale_to_covar_preci(glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr);
        }
        mat3 covar_c;
        covarW2C(R, covar, covar_c);

        Ks += bid * C * 9 + cid * 9;
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

        det = add_blur(eps2d, covar2d, compensation);
        if(det <= 0.f)
        {
            valid = false;
        }
        else
        {
            // compute the inverse of the 2d covariance
            covar2d_inv = glm::inverse(covar2d);
        }
    }

    // check if the points are in the image region
    float radius_x, radius_y;
    if(valid)
    {
        float extend = GAUSSIAN_EXTEND;
        if(opacities != nullptr)
        {
            float opacity = opacities[bid * N + gid];
            if(antialiased)
            {
                // we assume compensation term will be applied later on.
                opacity *= compensation;
            }
            if(opacity < ALPHA_THRESHOLD)
            {
                valid = false;
            }
            // Compute opacity-aware bounding box.
            // https://arxiv.org/pdf/2402.00525 Section B.2
            extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
        }

        // compute tight rectangular bounding box (non differentiable)
        // https://arxiv.org/pdf/2402.00525
        radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
        radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

        if(radius_x <= radius_clip && radius_y <= radius_clip)
        {
            valid = false;
        }

        // mask out gaussians outside the image region
        if(mean2d.x + radius_x <= 0
           || mean2d.x - radius_x >= image_width
           || mean2d.y + radius_y <= 0
           || mean2d.y - radius_y >= image_height)
        {
            valid = false;
        }
    }

    int32_t thread_data = static_cast<int32_t>(valid);
    if(block_cnts != nullptr)
    {
        // First pass: compute the block-wide sum
        int32_t aggregate;
        if(__syncthreads_or(thread_data))
        {
            typedef cub::BlockReduce<int32_t, N_THREADS_PACKED> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
        }
        else
        {
            aggregate = 0;
        }
        if(threadIdx.x == 0)
        {
            block_cnts[block_idx] = aggregate;
        }
    }
    else
    {
        // Second pass: write out the indices of the non zero elements
        if(__syncthreads_or(thread_data))
        {
            typedef cub::BlockScan<int32_t, N_THREADS_PACKED> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }
        if(valid)
        {
            if(block_idx > 0)
            {
                int32_t offset  = block_accum[block_idx - 1];
                thread_data    += offset;
            }
            // write to outputs
            batch_ids[thread_data]       = bid;
            camera_ids[thread_data]      = cid;
            gaussian_ids[thread_data]    = gid;
            radii[thread_data * 2]       = (int32_t)radius_x;
            radii[thread_data * 2 + 1]   = (int32_t)radius_y;
            means2d[thread_data * 2]     = mean2d.x;
            means2d[thread_data * 2 + 1] = mean2d.y;
            depths[thread_data]          = mean_c.z;
            conics[thread_data * 3]      = covar2d_inv[0][0];
            conics[thread_data * 3 + 1]  = covar2d_inv[0][1];
            conics[thread_data * 3 + 2]  = covar2d_inv[1][1];
            if(compensations != nullptr)
            {
                compensations[thread_data] = compensation;
            }
        }
        // lane 0 of the first block in each row writes the indptr
        if(threadIdx.x == 0 && block_col_idx == 0)
        {
            if(row_idx == 0)
            {
                indptr[0]     = 0;
                indptr[B * C] = block_accum[B * C * blocks_per_row - 1];
            }
            else
            {
                indptr[row_idx] = block_accum[block_idx - 1];
            }
        }
    }
}

void launch_projection_ewa_3dgs_packed_fwd_kernel(
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
    const at::optional<at::Tensor> block_accum, // [B * C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    const bool antialiased,
    // outputs
    at::optional<at::Tensor> block_cnts,   // [B * C * blocks_per_row] packing helper
    at::optional<at::Tensor> indptr,       // [B * C + 1]
    at::optional<at::Tensor> batch_ids,    // [nnz]
    at::optional<at::Tensor> camera_ids,   // [nnz]
    at::optional<at::Tensor> gaussian_ids, // [nnz]
    at::optional<at::Tensor> radii,        // [nnz, 2]
    at::optional<at::Tensor> means2d,      // [nnz, 2]
    at::optional<at::Tensor> depths,       // [nnz]
    at::optional<at::Tensor> conics,       // [nnz, 3]
    at::optional<at::Tensor> compensations // [nnz] optional
)
{
    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t C = viewmats.size(-3);       // number of cameras
    uint32_t B = means.numel() / (N * 3); // number of batches

    uint32_t nrows          = B * C;
    uint32_t ncols          = N;
    uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

    dim3 threads(N_THREADS_PACKED);
    // Each block-row maps one B*C batch-camera onto grid.y, so a larger B*C
    // would silently issue an invalid launch -- reject it with a clear error.
    TORCH_CHECK(
        nrows <= kMaxCudaGridDimY,
        "packed projection: B*C = ",
        nrows,
        " exceeds the CUDA grid.y limit of ",
        kMaxCudaGridDimY,
        " batch-camera rows per launch"
    );
    dim3 grid(blocks_per_row, nrows, 1);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if(B == 0 || N == 0 || C == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_packed_fwd_kernel",
        [&]()
        {
            projection_ewa_3dgs_packed_fwd_kernel<scalar_t>
                <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
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
                    block_accum.has_value() ? block_accum.value().const_data_ptr<int32_t>() : nullptr,
                    camera_model,
                    antialiased,
                    block_cnts.has_value() ? block_cnts.value().data_ptr<int32_t>() : nullptr,
                    indptr.has_value() ? indptr.value().data_ptr<int32_t>() : nullptr,
                    batch_ids.has_value() ? batch_ids.value().data_ptr<int64_t>() : nullptr,
                    camera_ids.has_value() ? camera_ids.value().data_ptr<int64_t>() : nullptr,
                    gaussian_ids.has_value() ? gaussian_ids.value().data_ptr<int64_t>() : nullptr,
                    radii.has_value() ? radii.value().data_ptr<int32_t>() : nullptr,
                    means2d.has_value() ? means2d.value().data_ptr<scalar_t>() : nullptr,
                    depths.has_value() ? depths.value().data_ptr<scalar_t>() : nullptr,
                    conics.has_value() ? conics.value().data_ptr<scalar_t>() : nullptr,
                    compensations.has_value() ? compensations.value().data_ptr<scalar_t>() : nullptr
                );
        }
    );
}

} // namespace gsplat

