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

// Feature channel counts the rasterizer is compiled for (one kernel per
// entry): 1 = D/ED, 3 = RGB, 4 = RGB+D/ED. Override with NUM_CHANNELS="..."
// at build time.
#ifndef GSPLAT_NUM_CHANNELS
#    define GSPLAT_NUM_CHANNELS 1, 3, 4
#endif
