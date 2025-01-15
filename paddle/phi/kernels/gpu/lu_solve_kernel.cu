// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/lu_solve_kernel.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void LuSolveKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& lu,
                   const DenseTensor& pivots,
                   const std::string& trans,
                   DenseTensor* out) {
  // Validate input dimensions
  auto x_dims = x.dims();
  auto lu_dims = lu.dims();
  auto pivots_dims = pivots.dims();

  PADDLE_ENFORCE_EQ(lu_dims[lu_dims.size() - 1],
                    lu_dims[lu_dims.size() - 2],
                    phi::errors::InvalidArgument("LU matrix must be square"));

  int n = static_cast<int>(lu_dims[lu_dims.size() - 1]);
  int nrhs = static_cast<int>(x_dims[x_dims.size() - 2]);

  PADDLE_ENFORCE_EQ(
      x_dims[x_dims.size() - 1],
      n,
      phi::errors::InvalidArgument(
          "Dimensions of input matrix and LU matrix do not match"));

  PADDLE_ENFORCE_EQ(pivots_dims[pivots_dims.size() - 1],
                    n,
                    phi::errors::InvalidArgument(
                        "Length of pivots array must equal matrix dimension"));

  dev_ctx.template Alloc<T>(out);

  cublasOperation_t trans_op;
  if (trans == "N") {
    trans_op = CUBLAS_OP_N;
  } else if (trans == "T") {
    trans_op = CUBLAS_OP_T;
  } else if (trans == "C") {
    trans_op = CUBLAS_OP_C;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "trans must be one of ['N', 'T', 'C'], but got %s", trans));
  }

  int lda = n;

  DenseTensor info_tensor;
  info_tensor.Resize({1});
  dev_ctx.template Alloc<int>(&info_tensor);
  int* d_info = info_tensor.data<int>();

  auto handle = dev_ctx.cusolver_dn_handle();

  // Copy x to out since cusolverDn*getrs overwrites the input
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);

  if (std::is_same<T, float>::value) {
    auto* lu_ptr = reinterpret_cast<const float*>(lu.data<T>());
    auto* out_ptr = reinterpret_cast<float*>(out->data<T>());

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSgetrs(handle,
                                                         trans_op,
                                                         n,
                                                         nrhs,
                                                         lu_ptr,
                                                         lda,
                                                         pivots.data<int>(),
                                                         out_ptr,
                                                         lda,
                                                         d_info));
  } else if (std::is_same<T, double>::value) {
    auto* lu_ptr = reinterpret_cast<const double*>(lu.data<T>());
    auto* out_ptr = reinterpret_cast<double*>(out->data<T>());

    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDgetrs(handle,
                                                         trans_op,
                                                         n,
                                                         nrhs,
                                                         lu_ptr,
                                                         lda,
                                                         pivots.data<int>(),
                                                         out_ptr,
                                                         lda,
                                                         d_info));
  }

  // Synchronize to ensure the solve is complete
  dev_ctx.Wait();
}

}  // namespace phi

PD_REGISTER_KERNEL(
    lu_solve, GPU, ALL_LAYOUT, phi::LuSolveKernel, float, double) {}
