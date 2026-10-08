# 干豆品种辅助分类

本项目为干豆整理人员提供一个小型分类工具：输入一粒干豆经过图像处理得到的 16 项形状测量值，输出 7 个候选品种之一，帮助进行初步分组。

**分类**是从几个固定类别中选择一个结果。本工具不是品种鉴定设备，预测结果必须由人复核，不能替代专家判断或生产验证。

## 1. 项目结果概览

- 真实数据：Dry Bean Dataset，共 13,611 条记录。
- 输入：16 个数值形状特征。
- 输出：BARBUNYA、BOMBAY、CALI、DERMASON、HOROZ、SEKER 或 SIRA。
- 简单基线：永远预测训练集中最多的类别。
- 主要候选：标准化加逻辑回归。
- 固定评估：分层 80% 训练、20% 测试，`random_state=42`。
- 核心指标：Macro F1，即先分别评价每个类别，再让 7 类拥有相同权重。
- 多数类基线 Macro F1：0.0590。
- 逻辑回归 Macro F1：0.9335。
- 随机森林 Macro F1：0.9342。

随机森林只比逻辑回归提高 0.0007，测试错误只少 1 条，因此不能称为明显提升。详细证据见 `report.md`。

## 2. 使用边界

这个程序只接收已经测量好的形状数值，不直接读取豆子照片。

可以得出的最小结论：在本项目固定测试集上，逻辑回归和随机森林都明显优于多数类基线。

不能得出的结论：

- 不能保证在不同相机、光照、产地或新批次上保持相同准确率；
- 不能把“模型置信度”当成预测一定正确的概率；
- 不能由本项目推出形状特征和豆类品种之间的因果关系；
- 不能用程序结果替代人工复核。

## 3. 真实数据契约

- 所有者/作者：Murat KOKLU、Ilker Ali OZKAN。
- 数据集：Dry Bean Dataset。
- Kaggle 页面：https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset
- 原始来源：https://www.muratkoklu.com/datasets/
- 论文 DOI：https://doi.org/10.1016/j.compag.2020.105507
- Kaggle 页面许可标签：CC0: Public Domain。
- 本项目下载日期：2026-08-24。
- 预期文件：`data/raw/Dry_Bean_Dataset.arff`。
- 预期结构：13,611 行、16 个数值输入、1 个 `Class` 类别列、7 个类别、0 个缺失值。
- 原始数据不提交 GitHub：`.gitignore` 已排除 `data/raw/`。
- 限制：这些特征来自特定图像采集和分割过程，不能代表所有真实环境。

### 下载和放置数据

1. 在浏览器打开上面的 Kaggle 页面。

2. 登录 Kaggle 并下载数据集压缩包。

3. 解压后找到 `Dry_Bean_Dataset.arff`。

4. 在本项目根目录创建 `data/raw/`，把 ARFF 文件放进去。

5. 确认最终路径为：

```text
dry-bean-final-project/data/raw/Dry_Bean_Dataset.arff
```

如果 Kaggle 无法访问，不要生成替代数据；请联系教师取得同一数据集的缓存副本。

## 4. 环境安装

下面所有命令都在项目根目录 `dry-bean-final-project` 中运行。

检查 Python：

```powershell
python --version
```

安装依赖：

```powershell
python -m pip install scikit-learn
```

本次结果使用 Python 3 和 scikit-learn 1.9.0。若安装失败，先运行 `python -m pip --version`；如果仍失败，保留完整报错并检查 Python 环境。

## 5. 第一次复现

先检查拿到的是预期真实数据：

```powershell
python main.py --check-data
```

关键预期输出：

```text
真实数据检查通过
行数：13611
列数：17
缺失值：0
```

再运行测试：

```powershell
python -m unittest discover -s tests -v
```

关键预期输出：

```text
Ran 8 tests
OK
```

测试只证明已检查的程序行为符合预期，不证明模型在新环境中一定正确。

## 6. 现场输入一粒干豆

### 方式 A：逐项输入

运行：

```powershell
python main.py --predict
```

程序会依次询问 16 个特征。每输入一个数字按一次 Enter。输入顺序必须与下表一致。

| 顺序 | 特征名 | 简单含义 |
| ---: | --- | --- |
| 1 | Area | 豆子区域面积 |
| 2 | Perimeter | 边界周长 |
| 3 | MajorAxisLength | 长轴长度 |
| 4 | MinorAxisLength | 短轴长度 |
| 5 | AspectRation | 长短轴比例；数据集原字段使用这个拼写 |
| 6 | Eccentricity | 形状偏离圆形的程度 |
| 7 | ConvexArea | 凸包面积 |
| 8 | EquivDiameter | 等效圆直径 |
| 9 | Extent | 区域在外接框中的占比 |
| 10 | Solidity | 区域相对凸包的充实程度 |
| 11 | roundness | 圆度 |
| 12 | Compactness | 紧致度 |
| 13 | ShapeFactor1 | 形状因子 1 |
| 14 | ShapeFactor2 | 形状因子 2 |
| 15 | ShapeFactor3 | 形状因子 3 |
| 16 | ShapeFactor4 | 形状因子 4 |

### 方式 B：复制一整行真实样本

下面使用真实数据第 12,804 条记录，真实类别为 DERMASON：

```powershell
python main.py --predict --values 36002 700.889 257.8889857 178.2686494 1.446631175 0.7226049788 36368 214.1008409 0.7973864895 0.9899362077 0.9209541377 0.8302054481 0.007163184981 0.002099076398 0.6892410861 0.9970776423
```

关键输出包括：

```text
单粒干豆预测完成
预测品种：DERMASON
范围检查：16项数值都在训练数据范围内
使用边界：这只是辅助建议，必须由人复核
```

程序还会显示模型置信度。这里的置信度只是模型对本次输出的内部评分，不能证明预测一定正确。

如果任一输入超出训练数据中观察到的最小值或最大值，程序会显示：

```text
必须人工复核
```

这时不要把结果当成可靠分类，应交给人工复核。

### 方式 C：先展示异常，再展示真实样本

先运行全零异常输入：

```powershell
python main.py --predict --values 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

关键预期输出是“必须人工复核”。`HOROZ` 和 `1.0000` 不能采用，因为16项输入全部超出训练范围。随后再运行方式 B 的真实样本命令，对比“范围检查通过”。

## 7. 基线、候选与错误分析

运行多数类基线：

```powershell
python main.py --baseline
```

关键结果：Accuracy 0.2604，Macro F1 0.0590。

运行逻辑回归候选：

```powershell
python main.py --candidate
```

关键结果：Accuracy 0.9214，Macro F1 0.9335。

运行随机森林比较：

```powershell
python main.py --random-forest
```

关键结果：Accuracy 0.9218，Macro F1 0.9342，共 213 条测试错误。

显示逻辑回归的类别结果和真实失败案例：

```powershell
python main.py --analyze-errors
```

其中一个真实失败是数据第 13,592 条：真实类别 DERMASON，模型输出 SIRA。该错误说明总体指标较高时仍会出现具体错分。

## 8. 程序流程

```text
真实 ARFF 数据
→ 检查行数、列数、类别和缺失值
→ 读取 16 个特征和 Class
→ 固定分层划分
→ 在相同测试集比较基线与候选
→ 保存指标和失败案例
→ 用全部批准数据训练演示模型
→ 对现场输入输出候选类别
→ 范围异常或重要结果交给人工复核
```

评估命令使用固定留出测试集产生报告数字；`--predict` 为现场使用而在全部批准数据上重新训练逻辑回归。这两个目的不同，不能把现场预测当成新的测试成绩。

## 9. 十分钟现场演示建议

1. 45 秒：说明使用者、16 项输入、7 类输出和人工边界。
2. 60 秒：运行 `python main.py --check-data`，证明数据结构。
3. 75 秒：解释数据进入检查、划分、模型和输出的流程。
4. 90 秒：运行基线和逻辑回归，比较同条件 Macro F1。
5. 60 秒：说明随机森林只轻微提高，避免夸大。
6. 90 秒：运行 `--analyze-errors`，解释真实失败和影响。
7. 90 秒：运行 `--predict` 或复制真实样本命令，展示单粒预测。
8. 60 秒：说明数据限制、智能体贡献和人工复核。
9. 剩余时间：切换 PPT 或回答教师追问。

现场演示前必须由每名成员亲自运行这些命令，并能解释一条输入如何经过 `load_dataset`、标准化、逻辑回归成为输出。

## 10. 文件说明

- `main.py`：数据检查、评估、错误分析和单粒预测入口。
- `tests/test_data_check.py`：数据读取、边界输入和模型配置测试。
- `report.md`：完整项目报告与结果解释。
- `presentation.pptx`：十分钟答辩幻灯片。
- `.gitignore`：排除原始数据、环境、缓存和模型文件。

## 11. 常见错误恢复

出现“找不到数据文件”：确认当前文件夹是项目根目录，并检查 `data/raw/Dry_Bean_Dataset.arff` 是否存在。

出现 `No module named sklearn`：运行：

```powershell
python -m pip install scikit-learn
```

输入数量错误：`--values` 后必须正好有 16 个数字；也可以改用 `python main.py --predict` 逐项输入。

中文显示乱码：在 PowerShell 运行：

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## 12. 提交前检查

- [ ] 在新终端按本 README 完整运行；
- [ ] 数据检查通过；
- [ ] 8 项测试通过；
- [ ] 基线、候选和报告数字一致；
- [ ] 单粒真实样本预测成功；
- [ ] 能解释一个真实失败案例；
- [ ] PPT 完成十分钟计时；
- [ ] 每名成员能解释关键函数、Macro F1 和人工边界；
- [ ] 补齐并检查 `submission.json`；
- [ ] GitHub 中没有 `data/raw/`、密钥、虚拟环境或缓存；
- [ ] 学生本人完成 GitHub 推送、权限检查和邮件提交。




