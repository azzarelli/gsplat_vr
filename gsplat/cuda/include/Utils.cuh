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

// Device-side math for the projection kernels.

#pragma once

#include "Common.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#ifdef __CUDACC__
#    include <cooperative_groups.h>
#endif
#include <cmath>

namespace gsplat
{
#ifdef __CUDACC__
namespace cg = cooperative_groups;
#endif

inline __device__ float rsqrt_parse_safe(float x)
{
#ifdef __CUDA_ARCH__
    return rsqrt(x);
#else
    return 1.f / std::sqrt(x);
#endif
}

///////////////////////////////
// World -> camera
///////////////////////////////

// [R | t] is the world-to-camera transform.
inline __device__ void posW2C(const mat3 R, const vec3 t, const vec3 pW, vec3 &pC)
{
    pC = R * pW + t;
}

inline __device__ void covarW2C(const mat3 R, const mat3 covarW, mat3 &covarC)
{
    covarC = R * covarW * glm::transpose(R);
}

///////////////////////////////
// Quaternion / scale -> covariance
///////////////////////////////

// quat is (w, x, y, z); normalised here.
inline __device__ mat3 quat_to_rotmat(const vec4 quat)
{
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float inv_norm  = rsqrt_parse_safe(x * x + y * y + z * z + w * w);
    x              *= inv_norm;
    y              *= inv_norm;
    z              *= inv_norm;
    w              *= inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return mat3(
        1.f - 2.f * (y2 + z2),
        2.f * (xy + wz),
        2.f * (xz - wy), // 1st col
        2.f * (xy - wz),
        1.f - 2.f * (x2 + z2),
        2.f * (yz + wx), // 2nd col
        2.f * (xz + wy),
        2.f * (yz - wx),
        1.f - 2.f * (x2 + y2) // 3rd col
    );
}

// Covariance R S S R^T and/or its inverse; either output may be null.
inline __device__ void quat_scale_to_covar_preci(const vec4 quat, const vec3 scale, mat3 *covar, mat3 *preci)
{
    mat3 R = quat_to_rotmat(quat);
    if(covar != nullptr)
    {
        mat3 S = mat3(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        mat3 M = R * S;
        *covar = M * glm::transpose(M);
    }
    if(preci != nullptr)
    {
        // Degenerate (zero-scale) Gaussians are culled before reaching here.
        assert(scale[0] != 0.f && scale[1] != 0.f && scale[2] != 0.f);
        mat3 S = mat3(1.0f / scale[0], 0.f, 0.f, 0.f, 1.0f / scale[1], 0.f, 0.f, 0.f, 1.0f / scale[2]);
        mat3 M = R * S;
        *preci = M * glm::transpose(M);
    }
}

///////////////////////////////
// 2D covariance
///////////////////////////////

// Low-pass filter: adds eps2d to the diagonal. Returns the blurred
// determinant; compensation is the opacity scale for antialiased mode.
inline __device__ float add_blur(const float eps2d, mat2 &covar, float &compensation)
{
    float det_orig  = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0]    += eps2d;
    covar[1][1]    += eps2d;
    float det_blur  = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation    = sqrtf(glm::max(MIN_COMPENSATION * MIN_COMPENSATION, det_orig / det_blur));
    return det_blur;
}

///////////////////////////////
// Camera projections (mean + Jacobian-projected covariance)
///////////////////////////////

inline __device__ void ortho_proj(
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    mat2 &cov2d,
    vec2 &mean2d
)
{
    float x = mean3d[0], y = mean3d[1];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );
    cov2d  = J * cov3d * glm::transpose(J);
    mean2d = vec2({fx * x + cx, fy * y + cy});
}

inline __device__ void persp_proj(
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    mat2 &cov2d,
    vec2 &mean2d
)
{
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // Clamp the Jacobian's x/z, y/z to 1.3x the frustum so off-screen
    // Gaussians do not blow up.
    float tan_fovx  = 0.5f * width / fx;
    float tan_fovy  = 0.5f * height / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz  = 1.f / z;
    float rz2 = rz * rz;
    float tx  = z * glm::min(lim_x_pos, glm::max(-lim_x_neg, x * rz));
    float ty  = z * glm::min(lim_y_pos, glm::max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );
    cov2d  = J * cov3d * glm::transpose(J);
    mean2d = vec2({fx * x * rz + cx, fy * y * rz + cy});
}

inline __device__ void fisheye_proj(
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    mat2 &cov2d,
    vec2 &mean2d
)
{
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float eps    = 0.0000001f;
    float xy_len = glm::length(glm::vec2({x, y})) + eps;
    float theta  = glm::atan(xy_len, z + eps);
    mean2d       = vec2({x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy});

    float x2         = x * x + eps;
    float y2         = y * y;
    float xy         = x * y;
    float x2y2       = x2 + y2;
    float x2y2z2_inv = 1.f / (x2y2 + z * z);

    float b  = glm::atan(xy_len, z) / xy_len / x2y2;
    float a  = z * x2y2z2_inv / (x2y2);
    mat3x2 J = mat3x2(
        fx * (x2 * a + y2 * b),
        fy * xy * (a - b),
        fx * xy * (a - b),
        fy * (y2 * a + x2 * b),
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv
    );
    cov2d = J * cov3d * glm::transpose(J);
}
} // namespace gsplat
