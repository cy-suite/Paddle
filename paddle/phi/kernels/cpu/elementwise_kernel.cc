// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/legacy/elementwise_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MinimumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  int axis = -1;
  RemainderRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  FloorDivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  int axis = -1;
  ElementwisePowRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void HeavisideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::ElementwiseHeavisideFunctor<T>, T>(
      dev_ctx, x, y, funcs::ElementwiseHeavisideFunctor<T>(), out);
}

template <typename T, typename Context>
void CopySignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    auto in_dims = x.dims();
    auto expand_shape = y.dims();
    if (in_dims.size() > expand_shape.size()) {
      in_dims = y.dims();
      expand_shape = x.dims();
    }
    auto vec_in_dims = common::vectorize<int>(in_dims);
    auto diff = expand_shape.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    std::vector<int> repeat_times(vec_in_dims.size());
    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      if (i < diff) {
        PADDLE_ENFORCE_GE(
            expand_shape[i],
            0,
            common::errors::InvalidArgument(
                "The expanded size (%d) for non-existing dimensions must be "
                "positive for expand_v2 op.",
                expand_shape[i]));
        repeat_times[i] = expand_shape[i];
      } else if (expand_shape[i] == 0) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i] == 1 || vec_in_dims[i] == expand_shape[i],
            true,
            common::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in shape for expand_v2 op.",
                vec_in_dims[i],
                expand_shape[i]));
        repeat_times[i] = 0;
      } else if (expand_shape[i] > 0) {
        if (vec_in_dims[i] == 0) {
          repeat_times[i] = 0;
        } else if (vec_in_dims[i] != 1) {
          PADDLE_ENFORCE_EQ(
              vec_in_dims[i],
              expand_shape[i],
              common::errors::InvalidArgument(
                  "The value (%d) of the non-singleton dimension does not match"
                  " the corresponding value (%d) in shape for expand_v2 op.",
                  vec_in_dims[i],
                  expand_shape[i]));
          repeat_times[i] = 1;
        } else {
          repeat_times[i] = expand_shape[i];
        }
      } else if (expand_shape[i] == -1) {
        repeat_times[i] = 1;
      }
    }
    DDim new_in_dims = common::make_ddim(vec_in_dims);
    DDim out_dims(new_in_dims);
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      if (repeat_times[i] == 0) {
        out_dims[i] = 0;
      } else if (expand_shape[i] == -1) {
        out_dims[i] = new_in_dims[i];
      } else {
        out_dims[i] *= repeat_times[i];
      }
    }
    out->Resize(out_dims);
    dev_ctx.template Alloc<T>(out);
    return;
  }
  dev_ctx.template Alloc<T>(out);
  if (x.dims().size() >= y.dims().size()) {
    funcs::ElementwiseCompute<funcs::CopySignFunctor<T>, T>(
        dev_ctx, x, y, funcs::CopySignFunctor<T>(), out);
  } else {
    funcs::ElementwiseCompute<funcs::InverseCopySignFunctor<T>, T>(
        dev_ctx, x, y, funcs::InverseCopySignFunctor<T>(), out);
  }
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::phi::dtype::bfloat16;

PD_REGISTER_KERNEL(
    fmax, CPU, ALL_LAYOUT, phi::FMaxKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    fmin, CPU, ALL_LAYOUT, phi::FMinKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(maximum,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(minimum,
                   CPU,
                   ALL_LAYOUT,
                   phi::MinimumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(remainder,
                   CPU,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(floor_divide,
                   CPU,
                   ALL_LAYOUT,
                   phi::FloorDivideKernel,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(elementwise_pow,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(heaviside,
                   CPU,
                   ALL_LAYOUT,
                   phi::HeavisideKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(copysign,
                   CPU,
                   ALL_LAYOUT,
                   phi::CopySignKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
