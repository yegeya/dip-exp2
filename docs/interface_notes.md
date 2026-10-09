# 成员 A 公共接口与交接说明

## 数据约定

图像数组顺序为行、列，坐标从 0 开始；灰度图为二维 float64，物理灰度范围 [0,1]。RGB 图像使用 Pillow 转灰度；uint8/uint16 按位深归一化，浮点图像必须已在 [0,1]。不调整图像尺寸、不拉伸对比度。源文件为 688×688 的字符测试图。

## 公共函数

| 模块与函数 | 输入 | 输出与行为 |
|---|---|---|
| common_io.read_gray(path) | 8/16 位无符号灰度、RGB 或范围 [0,1] 的浮点图像 | 二维 float64 灰度数组；文件不存在或格式非法则报错 |
| common_io.save_gray(image,path) | 二维浮点数组和文件路径 | 截断到 [0,1] 后四舍五入到 8 位；自动创建父目录；已有文件会覆盖 |
| common_io.normalize_display(array,limits=None) | 有限二维数组及可选显示范围 | 用于频谱等诊断显示的 [0,1] 数组；不可当作物理灰度结果交接 |
| common_io.show_gray(image) | 二维数组 | 在用户主动调用时打开系统图像查看器 |
| project_04_03.gaussian_lowpass(shape,d0,center=None) | 正整数 (M,N)，D₀>0，中心 (row,col) | 二维响应 H；默认中心为 (M//2,N//2) |
| project_04_03.apply_frequency_filter(image,H) | 同尺寸二维图像和响应 | 中心化 F、G、复数逆变换；非中心响应可能有非零虚部 |
| project_04_03.filter_image(image,d0,padding=False) | 二维图像、尺度及填充开关 | (低通浮点图像,H,F,G,最大虚部残差)；只使用直流中心响应 |

## 给成员 B 的样例

在 `src/` 中可直接导入；其他位置需把 src 加入模块路径。

```python
import numpy as np
from common_io import read_gray, save_gray
from project_04_03 import filter_image

original = read_gray('data/Fig0441(a)(characters_test_pattern).tif')
lowpass, H, F, G, residual = filter_image(original, d0=15)
# 或读取 np.load('results/04_03_lowpass_d0_15.npy')
# Project 04-04 的减法应使用浮点图像；负值如何显示由该项目决定。
highpass = original - lowpass
```

`save_gray` 会截断负值，不能直接用它保存高通原始数值。高通数据应另外保存 NPY，显示时再按 04-04 的要求选取偏移或对称显示范围。切勿从按独立 min-max 拉伸的显示图计算减法。

## 参数与边界

主实验 D₀=15、688×688、不填充，周期边界。题目允许该简化。启用 `padding=True` 会扩展为 1376×1376 并裁剪，D₀ 的单位为实际 FFT 网格；为了保持归一化频率带宽，应同时扩大 D₀。无填充与零填充的边界行为不同。

## 交接状态

公共函数、图像读取、结果保存、七项测试、参数说明及目标输出均已验证。成员 B 自己环境中的接收运行仍由成员 B 确认。运行 `python -m unittest discover -s tests -v` 可复核七项数值检查。
