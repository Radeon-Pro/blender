/*
 * Copyright 2011-2015 Blender Foundation
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

#define __SVM_EVAL_NODES_SHADER_TYPE_SURFACE__

#include "kernel/kernel_compat_opencl.h"
#include "kernel/split/kernel_split_common.h"
#include "kernel/split/kernel_shadow_blocked_dl.h"

#define KERNEL_NAME shadow_blocked_dl
#include "kernel/kernels/opencl/kernel_split_function.h"
#undef KERNEL_NAME

