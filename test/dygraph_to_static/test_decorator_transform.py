#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import warnings
from contextlib import contextmanager
from functools import wraps

import decos
import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_pt_only,
)

import paddle


def deco1(func):
    @wraps(func)
    def inner(*args, **kwargs):
        print('in deco1, added 1')
        _x = 2
        if _x < 1:
            _x += 1
        else:
            _x -= 1
        _t = paddle.to_tensor([1])
        _tt = func(*args, **kwargs)
        return paddle.add(_t, _tt)

    return inner


def deco2(fun):
    @wraps(fun)
    def inner(*args, **kwargs):
        print('in deco2, added 2')
        _t = paddle.to_tensor([2])
        _tt = fun(*args, **kwargs)
        return paddle.add(_t, _tt)

    return inner


def deco3(x=3):
    def inner_deco(func):
        @wraps(func)
        def inner(*args, **kwargs):
            print(f'in deco3, added {x}')
            _t = paddle.to_tensor(x)
            _tt = func(*args, **kwargs)
            return paddle.add(_t, _tt)

        return inner

    return inner_deco


def deco4(func=None, x=0):
    def decorated(pyfunc):
        @wraps(pyfunc)
        def inner_deco(*args, **kwargs):
            print(f'in deco4, added {x}')
            _t = paddle.to_tensor(x)
            _tt = pyfunc(*args, **kwargs)
            return paddle.add(_t, _tt)

        return inner_deco

    if func is None:
        return decorated
    return decorated(func)


def deco5():
    return deco2


def deco6(x=0):
    return deco2


@deco2
def fun1(x, y=0):
    a = paddle.to_tensor(y)
    print(f'in fun1, x={x}')
    return a


@deco1
@deco2
def fun2(x, y=0):
    a = paddle.to_tensor(y)
    print(f'in fun2, x={x}')
    return a


@deco3(3)
def fun3(x, y=0):
    a = paddle.to_tensor(y)
    print(f'in fun3, x={x}')
    return a


@deco4(x=4)
def fun4(x, y=0):
    a = paddle.to_tensor(y)
    print(f'in fun4, x={x}')
    return a


@deco2
@deco4()
def fun5(x, y=0):
    a = paddle.to_tensor(y)
    print(f'in fun5, x={x}')
    return a


@decos.deco1
@decos.deco2(2)
def fun6(x, y=0):
    a = paddle.to_tensor(y)
    print(f'in fun6, x={x}')
    return a


@deco5()
def fun7(x, y=0):
    a = paddle.to_tensor(y)
    print(f'in fun7, x={x}')
    return a


@deco6(2)
def fun8(x, y=0):
    a = paddle.to_tensor(y)
    print(f'in fun8, x={x}')
    return a


def forward():
    funcs = [fun1, fun2, fun3, fun4, fun5, fun6, fun7, fun8]
    out = []
    for idx, fun in enumerate(funcs):
        out.append(fun(idx + 1, idx + 1))
    return out


@contextmanager
def contextmanager_warning():
    yield


@contextmanager_warning()
def fun9():
    print('in fun9 want contextmanager warning')


def warn1():
    fun9()


@paddle.no_grad()
def fun10():
    print('in fun10, paddle api decorated')
    return True


def deco_with_paddle_api():
    return fun10()


class TestDecoratorTransform(Dy2StTestBase):
    def test_deco_transform(self):
        outs = paddle.jit.to_static(forward)()
        np.testing.assert_allclose(outs[0], np.array(3), rtol=1e-05)
        np.testing.assert_allclose(outs[1], np.array(5), rtol=1e-05)
        np.testing.assert_allclose(outs[2], np.array(6), rtol=1e-05)
        np.testing.assert_allclose(outs[3], np.array(8), rtol=1e-05)
        np.testing.assert_allclose(outs[4], np.array(7), rtol=1e-05)
        np.testing.assert_allclose(outs[5], np.array(9), rtol=1e-05)
        np.testing.assert_allclose(outs[6], np.array(9), rtol=1e-05)
        np.testing.assert_allclose(outs[7], np.array(10), rtol=1e-05)

    @test_ast_only
    @test_pt_only
    def test_contextmanager_warning(self):
        paddle.disable_static()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            paddle.jit.to_static(warn1)()
            flag = False
            for warn in w:
                if (
                    issubclass(warn.category, UserWarning)
                ) and "A context manager decorator is used" in str(
                    warn.message
                ):
                    flag = True
                    break
            self.assertTrue(flag)

    def test_deco_with_paddle_api(self):
        self.assertTrue(paddle.jit.to_static(deco_with_paddle_api)())


if __name__ == '__main__':
    unittest.main()
