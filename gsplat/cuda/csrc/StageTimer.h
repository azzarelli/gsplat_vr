/*
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

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

namespace gsplat
{
// CUDA-event timing of consecutive stages on the current stream: each mark()
// ends the stage begun by the previous one. Does nothing when disabled.
class StageTimer
{
  public:
    explicit StageTimer(bool enabled) : enabled_(enabled)
    {
        mark();
    }

    ~StageTimer()
    {
        for(cudaEvent_t e : events_)
        {
            cudaEventDestroy(e);
        }
    }

    StageTimer(const StageTimer &)            = delete;
    StageTimer &operator=(const StageTimer &) = delete;

    void mark()
    {
        if(!enabled_)
        {
            return;
        }
        cudaEvent_t e;
        C10_CUDA_CHECK(cudaEventCreate(&e));
        events_.push_back(e);
        C10_CUDA_CHECK(cudaEventRecord(e, c10::cuda::getCurrentCUDAStream()));
    }

    // [n_stages] float32 on the CPU, empty when disabled. Waits for the last mark.
    at::Tensor elapsed_ms() const
    {
        const int64_t n = enabled_ ? static_cast<int64_t>(events_.size()) - 1 : 0;
        at::Tensor out  = at::empty({n}, at::kFloat);
        if(n > 0)
        {
            C10_CUDA_CHECK(cudaEventSynchronize(events_.back()));
            float *ms = out.data_ptr<float>();
            for(int64_t i = 0; i < n; ++i)
            {
                C10_CUDA_CHECK(cudaEventElapsedTime(&ms[i], events_[i], events_[i + 1]));
            }
        }
        return out;
    }

  private:
    bool enabled_;
    std::vector<cudaEvent_t> events_;
};
} // namespace gsplat
