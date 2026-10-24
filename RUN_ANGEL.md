# Angel 单类别无条件训练

## 本地验收结果

已在 RTX 4060 Ti 16 GB / dp 环境完成：512 条训练样本、512 条独立验证样本，先训练 2 轮（32 步），再从 `last.ckpt` 恢复到第 3 轮（48 步）。验证损失依次为 0.794512、0.349072、0.289777。

独立进程加载第 2 轮 checkpoint 生成 16 个样例，与训练期间相同种子的生成结果逐元素完全一致，坐标均为有限值。恢复训练日志确认优化器及训练状态恢复成功。`pip check` 无依赖冲突。训练输入图已人工查看，笔画方向和断笔正常；短程生成仍为无结构线条，符合未充分训练的状态。

当前最终 checkpoint 为 `runs/angel_smoke/checkpoints/best-epoch=002.ckpt`（epoch 文件名从 0 开始），最终样例为 `runs/angel_smoke/generated_best.png`。`runs/angel_reload/generated.png` 保留独立加载第 2 轮模型的采样结果。

推荐入口是 `train_angel.py`。它复用原项目的 BiGRU、扩散损失、EMA 和 DDPM 采样器，补充 Sketch-RNN NPZ 数据适配、本地 CSV 日志、checkpoint 和图片输出。原始 `config.yml` 属于 Lightning 1.5 配置，不适用于当前环境；新入口不读取它。

## 环境

本地已使用 `D:\miniconda\envs\dp\python.exe`，Python 3.11、PyTorch 2.4.1+cu121、Lightning 2.4.0、NumPy 2.2.6。在服务器先安装与显卡驱动匹配的 PyTorch，再运行：

```bash
python -m pip install -r requirements-dp.txt
```

不需要 W&B 登录，训练日志写入本地。原项目中仍保留 W&B logger 的可选实现。

## 本地短程验证

在项目目录、已激活 dp 的终端中执行：

```powershell
python train_angel.py --data D:\document\DataSet\quickdraw\raw\angel.full.npz --output runs/angel_smoke --limit 512 --batch-size 32 --epochs 2 --preview-every 1
```

`--limit` 分别限制 train 和 valid 的样本数；0 表示全部。默认只跑 512 条样本、2 个 epoch。train/valid/test 划分保持独立，短程验证不读取 test。数据量不足一个 batch 时也会训练，不丢弃尾批。

输出：

- `training_examples.png`：转换后的真实训练数据，用于核对笔画和坐标方向。
- `checkpoints/last.ckpt`：最近一轮的模型、EMA、优化器和训练状态。
- `checkpoints/best-epoch=NNN.ckpt`：验证损失最低的模型。
- `samples_epoch_NNNN.png/.npz`：固定随机种子的 16 个生成样例。
- `generated_best.png`：训练结束后，重新加载最佳 checkpoint 生成的样例。
- `metrics/version_N/metrics.csv`：训练和验证损失。
- `run_config.json`、`status.json`：启动参数和最新进度。

少量训练后的样例只能用于验证采样流程，不能用来判断最终生成质量。

单独重新加载并生成（无需训练）：

```powershell
python train_angel.py --sample --resume runs/angel_smoke/checkpoints/last.ckpt --output runs/angel_reload --limit 512
```

恢复优化器、EMA 和训练进度，再训练到第 3 轮：

```powershell
python train_angel.py --resume runs/angel_smoke/checkpoints/last.ckpt --output runs/angel_smoke --limit 512 --batch-size 32 --epochs 3 --preview-every 1
```

`--epochs` 是总轮数，不是追加轮数。恢复训练时保持数据、limit、batch size 一致。

## 服务器正式训练

```bash
python train_angel.py --data /path/to/angel.full.npz --output runs/angel_full --limit 0 --batch-size 128 --epochs 200 --preview-every 5
```

这是正式训练的起始配置，不保证 200 轮已达到目标质量。根据验证损失和固定种子的生成图决定是否继续训练；断点续训增加总 epoch 数即可。当前入口使用单卡、FP32、num_workers=0；优先保证可复现的运行闭环，服务器性能调优另行进行。

## 数据适配说明

输入应包含 `train`、`valid`、`test`，每个样本是 `[dx, dy, pen_up]`。对位移累加得到绝对坐标，以第一个点为原点，最大空间半径缩放至 10，保留断笔标志，补上 [0,1] 时间列；与原项目 `ThreePointAbs` 一致，丢弃初始原点。过滤非有限值、退化样本，以及原序列长度不在 5–300 的样本并打印数量。

这条路径直接使用 NPZ 已有的采样点，不执行原始 NDJSON 路径中的曲线重采样或 Perlin 扰动；因此是针对该文件的训练适配，不声称严格复现论文预处理。无条件采样只从训练集抽取长度分布，初始坐标由纯高斯噪声产生，不使用真实样本坐标作为生成条件。
