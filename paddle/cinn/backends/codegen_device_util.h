// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once

#include <absl/container/flat_hash_map.h>

#include <string>
#include <tuple>
#include <vector>
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#endif
#ifdef CINN_WITH_HIP
#include "paddle/cinn/backends/hip/codegen_hip_dev.h"
#endif
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace backends {

#define KERNEL_ARGS "kernel_args"
#define KERNEL_ARGS_NUM "kernel_args_num"
#define KERNEL_STREAM "kernel_stream"
#define TENSOR_SHAPE_ARGS "tensor_shape_args"

/**
 * Split a CINN Module into two separate modules, one contains the host
 * functions, the other contains the device kernels.
 *
 * This contains some process:
 *
 * - replace the original kernel function with a Call node and add it to the
 * first module, add a device kernel function to the second module.
 */
std::tuple<ir::Module, ir::Module> SplitDeviceAndHostModule(ir::Module module);

ir::Module CreateSwitchWithBroadcastConditionModule(
    const std::vector<ir::Expr>& broadcast_conditions,
    const std::vector<std::string>& case_func_names,
    const std::string& wrapper_func_name,
    const std::unordered_map<int, ir::Var>& symbolic_shape_var_index);

namespace detail {

struct BucketStrategyHostFunctionCollector {
  explicit BucketStrategyHostFunctionCollector(const std::string& module_name)
      : host_module_builder_(module_name + "_host",
                             cinn::common::DefaultHostTarget()),
        device_module_builder_(module_name + "_gpu_device",
                               cinn::common::DefaultDeviceTarget()),
        kernel_args_(KERNEL_ARGS, type_of<void*>()),
        kernel_args_num_(KERNEL_ARGS_NUM, type_of<int>()),
        kernel_stream_(KERNEL_STREAM, type_of<void*>()),
        tensor_shape_args_(TENSOR_SHAPE_ARGS, type_of<int64_t**>()) {}

  std::tuple<ir::Module, ir::Module> CollectAndBuild(ir::Module m) {
    CollectModuleInfo(m.As<ir::_Module_>());
    return std::make_tuple(host_module_builder_.Build(),
                           device_module_builder_.Build());
  }

 private:
  static bool compare_priority(
      const std::pair<int, std::pair<ir::LoweredFunc, Expr>>& a,
      const std::pair<int, std::pair<ir::LoweredFunc, Expr>>& b) {
    return a.first > b.first;
  }

  void CollectModuleInfo(ir::_Module_* op) {
    if (op->functions.size() == 1 && op->predicates.size() == 0) {
      op->predicates.push_back(ir::Expr(true));
    }
    PADDLE_ENFORCE_EQ(
        op->functions.size(),
        op->predicates.size(),
        ::common::errors::InvalidArgument(
            "The size of functions and predicates should be equal"));
    PADDLE_ENFORCE_EQ(
        op->functions.size(),
        op->priorities.size(),
        ::common::errors::InvalidArgument(
            "The size of functions and priorities should be equal"));
    // Sort funcitons and predicates according to the priority
    std::vector<std::pair<ir::LoweredFunc, Expr>> func_predicate;
    std::vector<std::pair<int, std::pair<ir::LoweredFunc, Expr>>>
        predicate_priority;
    VLOG(3) << "The number of the functions is " << op->functions.size();
    for (int i = 0; i < op->functions.size(); i++) {
      auto func_pair = std::make_pair(op->functions[i], op->predicates[i]);
      func_predicate.push_back(func_pair);
      predicate_priority.push_back(
          std::make_pair(op->priorities[i], func_pair));
    }
    sort(
        predicate_priority.begin(), predicate_priority.end(), compare_priority);
    predicate_priority[0].second.first;

    for (int i = 0; i < op->functions.size(); ++i) {
      ProcessLoweredFunc(predicate_priority[i].second.first,
                         predicate_priority[i].second.second);
      if (i == 0) {
        ProcessArgs(op->functions[i]);
      }
    }

    std::vector<ir::Argument> arguments = {
        ir::Argument(kernel_args_, ir::Argument::IO::kOutput),
        ir::Argument(kernel_args_num_, ir::Argument::IO::kInput),
        ir::Argument(kernel_stream_, ir::Argument::IO::kOutput)};
    std::vector<ir::Expr> body_stmts(arg_defs_);
    body_stmts.insert(body_stmts.end(), buckets_.begin(), buckets_.end());
    ir::LoweredFunc host_func = ir::_LoweredFunc_::Make(
        op->functions[0]->name, arguments, ir::Block::Make(body_stmts), {});
    host_module_builder_.AddFunctionWithoutOptim(host_func);

    // Parse LoweredFunc to infer output tensor's shape
    std::vector<ir::Expr> infer_shape_func_body_stmts(arg_defs_);
    infer_shape_func_body_stmts.insert(infer_shape_func_body_stmts.end(),
                                       op->infer_shape_func->body);
    if (temp_space_infer_shape_body_.defined()) {
      infer_shape_func_body_stmts.push_back(temp_space_infer_shape_body_);
    }

    std::vector<ir::Argument> infer_shape_arguments = {
        ir::Argument(kernel_args_, ir::Argument::IO::kOutput),
        ir::Argument(kernel_args_num_, ir::Argument::IO::kInput),
        ir::Argument(tensor_shape_args_, ir::Argument::IO::kOutput)};

    ir::LoweredFunc host_infer_shape_func =
        ir::_LoweredFunc_::Make(op->infer_shape_func->name,
                                infer_shape_arguments,
                                ir::Block::Make(infer_shape_func_body_stmts),
                                {});
    host_module_builder_.AddFunctionWithoutOptim(host_infer_shape_func);
  }

  void ProcessLoweredFunc(ir::LoweredFunc func, ir::Expr predicate);

  void ProcessArgs(ir::LoweredFunc func);

  ir::LoweredFunc CreateDeviceFunction(ir::LoweredFunc func,
                                       ir::Expr predicate);

  inline std::string GenDeviceKernelName(const std::string& fn_name,
                                         ir::Expr predicate);

 private:
  std::vector<ir::Expr> buckets_;
  std::vector<ir::Expr> arg_defs_;
  ir::Expr temp_space_infer_shape_body_;

  ir::Var kernel_args_;
  ir::Var kernel_args_num_;
  ir::Var kernel_stream_;
  ir::Var tensor_shape_args_;

  ir::Module::Builder host_module_builder_;
  ir::Module::Builder device_module_builder_;
};

}  // namespace detail

}  // namespace backends
}  // namespace cinn
