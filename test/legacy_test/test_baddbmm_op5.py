#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle.base import core


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support float16",
)
class TestBaddBmmFP16Op(OpTest):
    def setUp(self):
        self.op_type = "baddbmm"
        self.python_api = paddle.baddbmm
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((1, 10, 10)).astype(self.dtype),
            'X': np.random.random((1, 10, 10)).astype(self.dtype),
            'Y': np.random.random((1, 10, 10)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

        self.place = core.CUDAPlace(0)

    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        self.check_grad_with_place(self.place, ['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        self.check_grad_with_place(self.place, ['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', no_grad_set=None
        )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
