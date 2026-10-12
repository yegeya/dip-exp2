# Project 04-05：Correlation in the Frequency Domain（频率域相关）

## 项目封面与摘要

**课程名称：** 数字图像处理
**课程号：** 待填写
**项目题目：** Correlation in the Frequency Domain
**项目编号：** Project 04-05
**学生姓名：** 待填写
**截止日期：** 待填写
**提交日期：** 待填写

**摘要**

本实验项目复现例 4.11 的频率域相关过程，用 Fig. 4.41(b) 中的字母"T"定位 Fig. 4.41(a) 中的对应目标。提供的模板截图为 49×44 像素，程序从中裁出 38×42 像素的模板，保留原文件。将 256×256 的图像和裁剪模板分别补零至 298×298 后，计算图像频谱与模板频谱复共轭的乘积，再逆变换得到二维相关函数。所求得的最大相关值约为 21352516，位于从零开始计数的 $(x,y)=(108,107)$。该坐标表示模板左上角，绘制的匹配框覆盖原图中的"T"。空间域逐像素乘积求和也得到相同的峰值，支持实现与坐标解释的正确性。

## 1. 实验题目

### 1.1 实验内容

**在频域中进行相关分析：**

要求使用 Fig. 4.41(a) 的图像和 Fig. 4.41(b) 的模板，复现 Example 4.11 的拓展、频率域相关与结果显示过程，以获得与 Fig. 4.41(e) 对应的二维相关图，标出二维相关函数中最大值所在位置的 (x，y) 坐标，无需在图 4.41(f) 中绘制该分布曲线。

### 1.2 实验目的

- 理解模板平移时相关值如何衡量局部相似程度；
- 理解空间域相关为何可以通过频谱共轭相乘和逆变换实现；
- 准确区分数组索引、模板左上角坐标和图中实际亮笔画的位置。结果是否合理既要看相关峰是否落在"T"，也要检查数值计算与直接空间域求和是否一致。

## 2. 技术讨论

### 2.1 算法原理

#### 2.1.1 模板匹配与图像相关基本原理

本实验以 Fig. 4.41(a) 为输入图像，以 Fig. 4.41(b) 中的"T"为模板，利用二维相关确定模板在图像中的位置。将模板在图像上逐位置移动，在每个位置对模板像素与对应图像像素逐点相乘并求和，得到该位置的相关值。设输入图像为 $f(x,y)$，模板为 $w(m,n)$，模板宽为 $C$、高为 $D$，当模板左上角位于 $(x,y)$ 时，有

$$
r(x,y)=\sum_{n=0}^{D-1}\sum_{m=0}^{C-1}f(x+m,y+n)w(m,n).
$$

式中，$(m,n)$ 是模板内的像素坐标，$(x+m,y+n)$ 是输入图像中对应的像素坐标。相关函数最大值所在的位置为本实验得到的最佳匹配位置。本实验使用原始灰度相关值，输出的 $(x,y)$ 表示**裁剪后的模板左上角**在图像中的位置，坐标原点位于图像左上角，$x$ 为列、$y$ 为行，均从零开始。

#### 2.1.2 频域相关理论

若在空间域直接求取完整相关函数，需要对各个模板位置分别执行逐像素乘积求和。依据傅里叶变换的相关定理，可将这一计算转换为频率域中的逐点乘法，再通过逆变换得到二维相关函数。对零填充后的图像与模板分别求二维傅里叶变换，记其频谱为 $F(u,v)$ 和 $W(u,v)$，则

$$
r(x,y)=\mathcal{F}^{-1}\!\left\{F(u,v)W^{*}(u,v)\right\}.
$$

其中，$F(u,v)$ 为输入图像频谱，$W(u,v)$ 为模板频谱，$W^{*}(u,v)$ 为模板频谱的复共轭。两者在频率域逐点相乘后，经二维逆傅里叶变换得到空间域相关函数。程序取逆变换结果的实部，以消除浮点运算产生的微小虚部。

#### 2.1.3 零填充技术

有限尺寸的 FFT 在未充分补零时产生循环相关，边界处的响应可能周期性绕回并发生混叠。为获得所需的线性相关结果，图像和模板分别放置在零数组的左上角。输入图像宽、高均为 256 像素；原模板截图为 49×44 像素，程序使用半开裁剪框 `(6, 1, 44, 43)` 去除外围黑边，得到宽 $C=38$、高 $D=42$ 像素的模板，原文件不变。

设输入图像宽为 $A=256$、高为 $B=256$。线性相关所需的最小补零尺寸为

$$
P=A+C-1=256+38-1=293 \\
Q=B+D-1=256+42-1=297.
$$

因此，本实验采用宽、高均为 298 像素的零填充数组，同时满足 $P$ 和 $Q$ 的要求。代码中的 `padded_shape=(298, 298)` 按 NumPy 的 `(高度, 宽度)` 顺序表示该尺寸。

#### 2.1.4 基于 FFT 的频域相关算法流程

程序按照以下顺序执行：

1. 读取输入图像 `input/4.41a.jpg` 和模板截图 `input/4.41b.png`，转换为灰度图。
2. 按 `(6, 1, 44, 43)` 裁剪模板，得到 38×42 像素的计算模板。
3. 将输入图像和裁剪后的模板分别置于 298×298 零数组的左上角。
4. 分别进行二维 FFT，获得图像频谱 $F(u,v)$ 和模板频谱 $W(u,v)$。
5. 将 $F(u,v)$ 与 $W^{*}(u,v)$ 逐点相乘。
6. 对乘积进行二维 IFFT，取实部得到二维相关函数。
7. 在原始相关数组中搜索最大值，并将数组的 `(行, 列)` 索引转换为题目所需的 $(x,y)=(列,行)$ 坐标。本次实验得到 $(108,107)$。

### 2.2 实现技术

#### 2.2.1 实验技术实现

- `np.fft.fft2()` 实现二维 FFT；
- `np.conj()` 对模板频谱取复共轭；
- `np.fft.ifft2()` 将频域乘积变换为二维相关函数；
- `np.argmax()` 寻找原始相关数组的最大值，随后由 `np.unravel_index()` 还原二维索引。
- `frequency_correlation` 函数组织补零及频域运算，`main` 函数完成输入预处理和最大值位置提取。
- PNG 相关图仅用于显示；峰值位置从未经显示映射的浮点相关数组中确定。

#### 2.2.2 程序设计与实现

1. `load_grayscale` 使用 Pillow 读取两张输入图并转换为单通道灰度图。程序检查图像为 256×256、裁剪前模板为 49×44，以避免输入文件更换后仍采用不适用的裁剪和补零参数。
2. 从模板截图裁剪出 38×42 的"T"，转换为 `float64` 数组。`frequency_correlation` 检查补零尺寸，将图像与模板分别放入 298×298 零数组的左上角。
3. 对两个数组分别执行 `np.fft.fft2`，把图像频谱与模板频谱的复共轭逐点相乘。执行 `np.fft.ifft2` 并取实部，获得 298×298 的原始相关数组。
4. `np.argmax` 找到原始相关数组的全局最大值，`np.unravel_index` 将一维序号还原为 `(行,列)`，然后写为从零开始的 `(x,y)=(列,行)`。
5. 将原始数组存为 `.npy`，峰值与裁剪参数存为 `.json`，另将相关值映射为 PNG 用于观察。在线性显示图中最大非负值映射为 255，对数显示图则突出弱响应，两者都不改变原始相关值。
6. 以峰值坐标为左上角，按照模板的 38×42 尺寸绘制矩形标记，保存匹配位置图。默认使用 Matplotlib 显示原图、模板、相关图和标注图，`--no-show` 选项仅关闭交互显示。

#### 2.2.3 关键代码展示

以下仅列出 `src/project_04_05.py` 中与算法原理、实验方法对应的核心代码，完整可运行程序见该源文件。

**读取灰度图与裁剪模板**

```python
def load_grayscale(path):
    with Image.open(path) as image:
        return image.convert("L")


image = load_grayscale(arguments.image)
template_original = load_grayscale(arguments.template)
if image.size != (256, 256):
    raise ValueError(f"输入图像应为 256×256 像素，实际为 {image.size}")
if template_original.size != (49, 44):
    raise ValueError(f"裁剪前的模板应为 49×44 像素，实际为 {template_original.size}")

# Pillow 的裁剪框是 (左, 上, 右, 下)，右、下边界不包含在结果中。
template = template_original.crop(TEMPLATE_CROP)
```

`Image.open` 读取文件，`convert("L")` 将其转换为单通道灰度图。`TEMPLATE_CROP` 定义为 `(6, 1, 44, 43)`；`crop` 在内存中生成裁剪结果，不覆盖原始输入文件。尺寸检查确保固定裁剪框适用于当前输入图像。

**补零与频域相关：`frequency_correlation`**

```python
def frequency_correlation(image, template, padded_shape):
    image_array = np.asarray(image, dtype=np.float64)
    template_array = np.asarray(template, dtype=np.float64)
    image_height, image_width = image_array.shape
    template_height, template_width = template_array.shape
    padded_height, padded_width = padded_shape

    if padded_height < image_height + template_height - 1:
        raise ValueError("补零后的高度不足，无法计算线性相关")
    if padded_width < image_width + template_width - 1:
        raise ValueError("补零后的宽度不足，无法计算线性相关")

    # 两幅图都放在左上角，其余位置补零，避免循环相关造成的混叠。
    image_padded = np.zeros(padded_shape, dtype=np.float64)
    template_padded = np.zeros(padded_shape, dtype=np.float64)
    image_padded[:image_height, :image_width] = image_array
    template_padded[:template_height, :template_width] = template_array

    # 相关定理：空间域相关 = 图像频谱乘模板频谱的共轭，再做逆变换。
    image_spectrum = np.fft.fft2(image_padded)
    template_spectrum = np.fft.fft2(template_padded)
    correlation = np.fft.ifft2(image_spectrum * np.conj(template_spectrum)).real
    return correlation, image_padded, template_padded
```

`np.asarray(..., dtype=np.float64)` 将灰度图转换为浮点数组，`shape` 给出高度和宽度。`np.zeros` 分别建立大小为 `padded_shape=(298, 298)` 的零值数组，切片赋值将图像和模板置于各自数组的左上角。函数末尾依次执行两次 `fft2`、模板频谱取共轭并逐点相乘，以及 `ifft2` 逆变换。`.real` 去除数值误差引入的微小虚部；函数同时返回原始浮点相关数组和实际用于 FFT 的两个补零数组。

**寻找峰值并解释坐标**

```python
padded_shape = (298, 298)
correlation, image_padded, template_padded = frequency_correlation(
    image, template, padded_shape
)
peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
peak_x = int(peak_x)
peak_y = int(peak_y)
peak_value = float(correlation[peak_y, peak_x])
```

`np.argmax` 在 298×298 的**原始数组**中找到最大值的一维序号，`np.unravel_index` 将其还原为 `(行,列)`。因此变量先写 `peak_y`、后写 `peak_x`，之后再按 `(x,y)` 输出。`int` 和 `float` 将 NumPy 标量转换为便于打印和写入 JSON 的 Python 数值。

**空间域逐像素求和校验：`verify_spatial_correlation`**

```python
def verify_spatial_correlation(image, template, correlation):
    image_array = np.asarray(image, dtype=np.float64)
    template_array = np.asarray(template, dtype=np.float64)
    template_height, template_width = template_array.shape
    valid_height = image_array.shape[0] - template_height + 1
    valid_width = image_array.shape[1] - template_width + 1
    spatial = np.empty((valid_height, valid_width), dtype=np.float64)

    for row in range(valid_height):
        for column in range(valid_width):
            region = image_array[row:row + template_height, column:column + template_width]
            spatial[row, column] = np.sum(region * template_array)

    spatial_peak_y, spatial_peak_x = np.unravel_index(np.argmax(spatial), spatial.shape)
    max_error = float(np.max(np.abs(spatial - correlation[:valid_height, :valid_width])))
    if not np.allclose(spatial, correlation[:valid_height, :valid_width], rtol=0, atol=1e-6):
        raise ValueError(f"空间域与频域相关结果不一致，最大绝对误差为 {max_error}")
    return int(spatial_peak_x), int(spatial_peak_y), float(spatial[spatial_peak_y, spatial_peak_x]), max_error
```

该函数只对模板完全位于原图内的有效位置逐点计算空间域相关，再与 FFT 结果的相应区域比较。它是频域计算后的独立校验，不是本项目要求的频域相关算法。主程序还检查两种方法得到的最大值坐标是否相同，并将空间域峰值和最大绝对误差写入 `04_05_peak.json`。

**相关结果可视化：`display_correlation`**

```python
def display_correlation(correlation, logarithmic=False):
    # 仅为显示映射到 0–255；峰值定位仍使用原始浮点相关数组。
    nonnegative = np.maximum(correlation, 0)
    peak_value = nonnegative.max()
    if peak_value == 0:
        return Image.fromarray(np.zeros(correlation.shape, dtype=np.uint8))
    if logarithmic:
        scaled = np.log1p(nonnegative) / np.log1p(peak_value)
    else:
        scaled = nonnegative / peak_value
    return Image.fromarray(np.round(255 * scaled).astype(np.uint8))
```

图像文件需要可显示的灰度范围，这个函数先把负响应截为 0，再把最大非负值缩放为 255。`logarithmic=True` 时使用对数映射，便于看到较弱响应。这是**显示处理**，不用于求峰值，也不会修改 `correlation` 原数组。

**保存结果与标注匹配位置**

```python
arguments.results.mkdir(parents=True, exist_ok=True)
correlation_display = display_correlation(correlation)
template.save(arguments.results / "04_05_template_cropped.png")
Image.fromarray(image_padded.astype(np.uint8)).save(arguments.results / "04_05_image_padded.png")
Image.fromarray(template_padded.astype(np.uint8)).save(arguments.results / "04_05_template_padded.png")
correlation_display.save(arguments.results / "04_05_correlation_display.png")
display_correlation(correlation, logarithmic=True).save(arguments.results / "04_05_correlation_log_display.png")
np.save(arguments.results / "04_05_correlation.npy", correlation)

marked = image.convert("RGB")
draw = ImageDraw.Draw(marked)
draw.rectangle((peak_x, peak_y, peak_x + template.width - 1, peak_y + template.height - 1),outline="red",width=2,)
marked.save(arguments.results / "04_05_match_location.png")
```

`.npy` 文件保存完整浮点相关数组，PNG 文件用于结果展示。两张延拓图直接由参与 FFT 的补零数组转换为 8 位灰度图保存，不额外裁剪或缩放。矩形标记以峰值坐标为左上角，其宽度和高度等于裁剪模板的实际尺寸；绘制彩色标记前将原图转换为 RGB 格式。

最后将结果在交互窗口中弹窗展示，输出相关函数最大值位置坐标和最大值，并将结果保存到`results/`目录下。

#### 2.2.4 实验环境

- 操作系统：Windows。
- 编程语言及版本：Python 3.12.7。
- 使用的库及版本：NumPy 1.26.4、Pillow 10.4.0、Matplotlib 3.10.0。
- 输入图像及来源：项目提供的 Fig. 4.41(a) 图像和 Fig. 4.41(b) 模板截图。模板截图与例题标称尺寸不一致，因此记录了程序裁剪步骤。
- 输出文件名称与格式：`results/04_05_*.png` 为展示图，`04_05_correlation.npy` 为完整浮点相关数组，`04_05_peak.json` 记录输入、裁剪、补零与最大值信息。

## 3. 结果讨论

### 3.1 结果展示

图 04-05-1、04-05-2 展示原图和裁剪模板，图 04-05-3、04-05-4 展示分别补零至 298×298 后实际参与 FFT 的图像与模板，对应例 4.11 的延拓步骤。图 04-05-5 至 04-05-7 展示相关函数及峰值位置。延拓图只保存至结果目录，不在程序运行后的四图窗口中显示。图注中的尺寸为文件的实际像素尺寸，交互窗口中的模板图像经过显示缩放，算法输入时仍保持 38×42 像素。精确相关值以 `results/04_05_peak.json` 和 `results/04_05_correlation.npy` 为准。

- 图 04-05-1：输入图像 Fig. 4.41(a)，256×256 像素。
  ![输入图像](input/4.41a.jpg)
- 图 04-05-2：由 Fig. 4.41(b) 截图裁剪的 38×42 模板。
  ![裁剪模板](results/04_05_template_cropped.png)
- 图 04-05-3：输入图像左上角对齐并补零后的 298×298 灰度图，其余区域为零。
  ![补零后的输入图像](results/04_05_image_padded.png)
- 图 04-05-4：裁剪模板左上角对齐并补零后的 298×298 灰度图，其余区域为零。
  ![补零后的模板](results/04_05_template_padded.png)
- 图 04-05-5：298×298 二维相关函数，非负值相对峰值线性映射到 0–255。
  ![二维相关函数](results/04_05_correlation_display.png)
- 图 04-05-6：原图上的峰值位置与模板大小标注，标注框从最大值位置框选模板在原图匹配位置。
  ![匹配位置](results/04_05_match_location.png)
- 图 04-05-7：对数映射的相关函数辅助观察图。
  ![对数显示相关函数](results/04_05_correlation_log_display.png)

### 3.2 结果分析

**峰值与定位：** 二维相关数组尺寸为 298×298。全局最大值约 21352516，坐标为 $(108,107)$；将模板的左上角放在此处，可以看到 38×42 的标注框覆盖原图中的"T"。因此本次定位与例题对最高峰位于"T"顶部附近的描述一致。峰值坐标是相关函数数组的索引，也代表该模板左上角的放置位置。

**数值校验：** 程序中的 `verify_spatial_correlation` 对原图中所有允许模板完全放置的位置直接执行逐像素乘积求和。空间域相关函数的最大值为 21352516，坐标为 $(108,107)$；FFT 数组在该点的值约为 21352516。两个有效区域逐点比较的最大绝对误差约为 $1.49\times10^{-8}$，属于浮点运算误差。这既检查了峰值，也检查了有效区域的索引方向与频域公式，是对**现有输入和实现**的一致性验证。校验数值记录在 `results/04_05_peak.json` 中。

**显示与参考图：** 线性显示图中，"T"所在区域形成最明显的响应；对数显示图放大较弱了响应，也能看到其他字母附近的相关结构。两张显示图都把负值截为零，并对非负值作可视化映射，因此图上的灰度不等于原始相关值。与 Fig. 4.41(e) 对比，可以核对峰值所在对象及总体形态，经核对对比及上述多重校验可判断，此实验所得结果是正确的。

## 4. 创新点及思考

**独立验证与可复现记录：** 在题目要求的相关图和峰值坐标之外，我保留了原始相关数组与 JSON 参数记录，并通过空间域逐像素求和交叉验证频域结果的正确性。两种计算路径在有效区域内达到浮点误差量级的一致性，这样做便于发现共轭遗漏、行列颠倒或填充不足等实现问题。

**验证依据的适用范围：** 匹配位置标记覆盖目标区域，说明模板定位结果在视觉上与预期一致，但该结果只能验证匹配位置的合理性，不能单独证明相关函数计算过程正确。由于相关函数可能存在其他局部峰值，还需要进一步确认最大值来源于原始相关数组，而非归一化或增强后的显示图像。在满足零填充条件下，根据傅里叶相关定理，空间域直接计算得到的相关函数应与通过 FFT 频域乘法并逆傅里叶变换得到的相关函数保持一致。因此，本实验中视觉核对主要用于验证定位结果，数值校验用于验证相关计算过程。

**此方法的局限与可继续探索的问题：** 当前采用未归一化灰度相关，相关值直接由像素乘积累加得到，因此对图像整体亮度变化较为敏感，当其他区域具有较高平均灰度时可能产生较大的相关响应。此外，该方法依赖模板与目标区域之间的像素对应关系，当目标发生旋转、缩放或笔画变化时，匹配峰值可能降低。因此考虑后续可以进一步引入零均值归一化相关方法，比较不同亮度条件下的匹配鲁棒性，同时研究不同模板裁剪范围对匹配位置和相关峰值的影响。

## 5. 实验总结

本实验利用傅里叶变换的相关定理实现了图像与模板的二维相关，并通过相关函数的最大值确定了模板在输入图像中的位置。所得结果与原图中"T"的位置一致，空间域计算也验证了频域结果的数值正确性。通过本实验，我进一步理解了零填充、频谱复共轭相乘和逆变换在频域相关中的作用，也认识到相关峰值与模板匹配位置之间的关系。

## 附录：程序清单与运行说明

- 程序文件/清单：`src/project_04_05.py`。
- 运行入口及命令：在 `project_04_05/` 目录运行 `python src/project_04_05.py`，结果保存后弹出四图窗口；无界面运行可加 `--no-show`。
- 输入图像与模板：`input/4.41a.jpg`、`input/4.41b.png`。
- 输出图像及数据：运行后写入 `results/`。裁剪模板、两张补零延拓图、线性相关图、对数相关图和匹配标注图为 PNG；完整相关数组为 `.npy`；坐标和参数为 `.json`。
- 关键参数及复现步骤：安装 `requirements.txt` 所列依赖，保持两张输入图在 `input/`，在本目录执行上述命令。程序用裁剪区域 `(6, 1, 44, 43)` 得到 38×42 模板，补零至 298×298，再输出结果；无需手动裁图。重新运行会从原始模板开始处理并更新 `results/` 中的输出。
- 核查方式：运行程序时会自动执行空间域校验。检查 `results/04_05_peak.json` 中的 `peak_xy_zero_based` 与 `spatial_peak_xy_zero_based` 是否均为 `[108, 107]`，以及 `spatial_fft_max_absolute_error_valid_region` 是否接近零；同时确认匹配位置标记覆盖“T”。JSON 中记录的结果路径是运行时的绝对路径，在其他设备上复现时应以项目目录下的实际文件位置为准。
