# 数字图像处理实验二

本次交付包括成员 A 的公共框架与 Project 04-03 高斯低通滤波。`docs/04_03_report.md` 是按原模板组织的独立报告，`docs/interface_notes.md` 说明提供给成员 B 的接口。

## 运行

```bash
python -m pip install -r requirements.txt
python src/project_04_03.py
python -m unittest discover -s tests -v
```

默认使用 data 中的字符测试图；参数 5、15、30、80、230 对应参考图 4.18(b)-(f)，目标 (c) 使用 15。主实验不填充，使用周期边界。默认会覆盖同名结果文件；新实验可指定 `--output results/new_run`。

## 文件

- `src/common_io.py`：读取、灰度化、归一化、保存及显示。
- `src/project_04_03.py`：可配置尺寸及中心的高斯响应、频域滤波、结果与参数导出。
- `results/04_03_output_d0_15.png`：目标结果显示图。
- `results/04_03_lowpass_d0_15.npy`：未经量化的低通结果，供后续项目使用。
- `results/04_03_metrics.csv` 和 `04_03_metadata.json`：实测指标、哈希和环境记录。
- `docs/04_03_report.md`：报告；只含 04-03。
- `docs/interface_notes.md`：A→B 交接说明。

报告中的姓名、学号、课程号、截止和提交日期保留待填写。原分工方案及空白模板保留，便于后续小组合并。

## 图像来源

输入 `Fig0441(a)(characters_test_pattern).tif` 。
