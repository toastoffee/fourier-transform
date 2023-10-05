#! /usr/bin/python3
# -*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import numpy as np
from numpy import pi, cos, sin

# 中文显示问题
# plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def test1():
    """
    矩形周期方波，进行复数形式的级数展开
    f(t) = 0， -T/2 <= t < - tao/2,
         = h,  -tao/2 <= t < tao/2
         = 0,   tao/2 <= t < T/2

    cn = (h/(n*pi)) * sin((n*pi*tao)/T), n = +/- 1, +/- 2, ...
    c0 = h*tao / T

    f(t) =  h*tao/T + h/pi * (1/n) * sin((n*pi*tao)/T * e(i*(2npit)/T)
    """

    fs = 100  # 采样频率
    t = np.arange(-10, 10, 1.0 / fs)

    h = 4
    T = 5
    tao = 1

    x = h * tao / T

    for n in range(-50, 50):  # n= 0, +-1,+-2,...
        if n == 0:
            continue
        x = x + (h / pi) * (1 / n) * sin((n * pi * tao) / T) * np.exp(
            (2j * n * pi * t) / T
        )

    plt.plot(t, x)
    plt.show()


test1()
