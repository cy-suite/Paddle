# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from tensorrt_test_base import TensorRTBaseTest

import paddle


class TestEinsumCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.einsum
        self.api_args = {
            "equation": "abc,c->ab",
            "operands": [
                np.random.randn(2, 3, 4).astype("float32"),
                np.random.randn(4).astype("float32"),
            ],
        }
        self.program_config = {"feed_list": ["operands_0", "operands_1"]}

        self.min_shape = {"operands_0": [1, 2, 3], "operands_1": [3]}
        self.opt_shape = {"operands_0": [2, 3, 4], "operands_1": [4]}
        self.max_shape = {"operands_0": [4, 5, 6], "operands_1": [6]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEinsumCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.einsum
        self.api_args = {
            "equation": "abcd,bcd->a",
            "operands": [
                np.random.randn(2, 3, 4, 5).astype("float32"),
                np.random.randn(3, 4, 5).astype("float32"),
            ],
        }
        self.program_config = {"feed_list": ["operands_0", "operands_1"]}

        self.min_shape = {"operands_0": [1, 2, 3, 4], "operands_1": [2, 3, 4]}
        self.opt_shape = {"operands_0": [2, 3, 4, 5], "operands_1": [3, 4, 5]}
        self.max_shape = {"operands_0": [4, 6, 8, 10], "operands_1": [6, 8, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEinsumCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.einsum
        self.api_args = {
            "equation": "abcd,ab->cd",
            "operands": [
                np.random.randn(2, 3, 4, 5).astype("float32"),
                np.random.randn(2, 3).astype("float32"),
            ],
        }
        self.program_config = {"feed_list": ["operands_0", "operands_1"]}

        self.min_shape = {"operands_0": [1, 2, 3, 4], "operands_1": [1, 2]}
        self.opt_shape = {"operands_0": [2, 3, 4, 5], "operands_1": [2, 3]}
        self.max_shape = {"operands_0": [4, 6, 8, 10], "operands_1": [4, 6]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
