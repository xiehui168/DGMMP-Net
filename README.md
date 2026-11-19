# DGMMP-Net

DGMMP-Net 是一个面向乳腺癌术后放疗预后评估的多模态深度学习项目。当前实现将 CT、放疗剂量、病理图像 patch 和临床特征作为输入，联合预测生存风险、复发类型及肺/心脏毒性，并生成风险分层和模型评估图。

## 项目结构

```text
.
├── main.py                 # 当前主流程：数据读取、模型、训练、验证和绘图
├── dataset.py              # 另一种预处理数据格式的 Dataset 实现
├── model.py                # 独立模型实现
├── train.py                # 独立训练工具
├── inference_and_xai.py    # 推理、不确定性估计和注意力提取工具
├── evaluation_tools.py     # 评估指标工具
├── risk_analysis.py        # 风险阈值及生存分析工具
├── clinical_data.csv       # 临床特征及结局标签
├── train.csv               # 训练集清单（运行 main.py 时会重新生成）
├── val.csv                 # 验证集清单（运行 main.py 时会重新生成）
├── rawdata/
│   ├── ct/                 # 按患者 ID 存放的 DICOM CT、 RT Dose/RT Structure
│   └── pathology/          # 按患者 ID 存放的 PNG 病理图像 patch
└── results/                # 训练评估图输出目录
```

## 环境

使用 Python 虚拟环境，并安装以下依赖：

```bash
pip install numpy pandas torch matplotlib lifelines scikit-learn SimpleITK pydicom opencv-python timm rt-utils
```

请根据本机 CUDA 版本，从 PyTorch 官方安装页面选择对应的 PyTorch 安装命令。没有可用 GPU 时会退回 CPU，但训练及病理 ViT 推理可能非常耗时。首次创建预训练 ViT 时，`timm` 可能需要下载预训练权重或使用本地缓存。

## 数据准备

### 临床表格

`clinical_data.csv` 每行对应一名患者，至少需要包含以下列：

| 列 | 含义 |
| --- | --- |
| `patient_id` | 患者 ID；用于匹配影像文件夹 |
| `time` | 随访时间 |
| `event` | 生存结局事件指示 |
| `recurrence_type` | 复发类型类别标签 |
| `tox_lung` | 肺毒性标签 |
| `tox_heart` | 心脏毒性标签 |

除上述 ID 和标签列以外，`main.py` 按 CSV 中的原始列顺序取最多 32 个临床特征。当前模型配置要求 32 个特征，因此请保持特征列数量和顺序与模型输入一致，并确保特征可转换为数值。布尔型特征应使用可转换为数值的值。

运行时，`main.py` 会用固定随机种子将完整临床表格按 70%/30% 划分为训练集和验证集，并覆盖项目根目录下的 `train.csv` 和 `val.csv`。这两个文件不是 `main.py` 的输入划分配置。

### CT、剂量和结构文件

每位患者对应一个 CT 文件夹，目录名需要与 `patient_id` 一致：

```text
rawdata/ct/<patient_id>/
```

文件夹中放置该患者的 DICOM CT 序列。RT Dose（文件名以 `RD` 开头）和 RT Structure（文件名以 `RS` 开头）为可选文件。代码尝试使用结构集中的靶区 ROI 定位裁剪区域，并将剂量重采样至 CT 空间。

### 病理图像

病理 PNG patch 按患者 ID 放置：

```text
rawdata/pathology/<patient_id>/*.png
```

图像会缩放为 256×256，并转换为 RGB。训练阶段最多采样 256 个 patch，验证阶段最多采样 512 个 patch。

## 运行

先检查 `main.py` 中 `Config` 类的 `DATA_ROOT`、`CLINICAL_CSV_PATH` 和 `RESULTS_DIR`，以及 `main()` 中 `train_csv`、`val_csv` 的输出路径，将它们改为本机路径，然后在项目根目录执行：

```bash
python main.py
```

训练使用 AdamW 和早停策略，最多训练 100 个 epoch。运行结束后，结果保存在 `results/`：

- `fig3c_km_curve.png`：按预测风险分组的 Kaplan–Meier 曲线
- `fig3b_calibration.png`：校准曲线
- `fig3a_time_roc.png`：时间依赖 ROC 曲线

验证集 C-index 会输出到终端。当前主流程不保存训练好的模型权重。

## 实现说明与注意事项

- `main.py` 是端到端运行入口；`dataset.py`、`model.py` 和 `train.py` 包含独立实现，数据格式和接口与 `main.py` 内部实现不完全相同，不能假设可直接混用。
- 当前 `main.py` 使用 Windows 绝对路径配置。更换机器或目录时，需要先修改 `Config` 中的路径。
- 影像读取发生异常、CT 缺失或病理目录/PNG 缺失时，当前加载器会以随机生成的数据代替，而不是中止训练。正式实验前请检查数据是否完整，并确认运行日志/代码未将异常样本纳入训练。
- CT、RT Dose、RT Structure 和病理图像需要按同一患者 ID 对齐；临床标签与影像的对应关系应在训练前单独核验。
- `evaluation_tools.py` 和 `risk_analysis.py` 提供辅助分析函数；其中部分评估实现使用简化计算，正式临床或研究报告应采用经验证的生存分析方法并进行独立验证。
- 本项目面向研究用途，不应将模型输出作为临床诊断或治疗决策的唯一依据。
