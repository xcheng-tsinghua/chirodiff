# 单类别无条件生成

所有参数从命令行读取，未填写时使用 train.py 中的默认值。激活 dp 后运行：

```powershell
python train.py --npz_path "D:\document\DataSet\quickdraw\raw\angel.full.npz" --batch_size 128 --random_seed 5555 --epoch 200 --num_images 16
```

使用 `python train.py --help` 查看参数说明。带下划线的参数也支持对应的短横线写法，例如 `--batch-size`。服务器上修改数据路径，并使用已配置好 `requirements-dp.txt` 和 CUDA PyTorch 的环境。

只读取 NPZ 内的 `train_split`；没有该键时读取 `train`（当前 angel 文件的实际键名）。不会读取 valid/test，也不从训练划分另切验证集。使用全部符合预处理条件的训练样本；过滤规则沿用已验证的数据适配：长度 5–300、有限坐标、非退化轨迹。输入为 Sketch-RNN 的 `[dx, dy, pen_up]`。

训练达到指定总轮数后，保存最后一轮的完整 checkpoint（包含模型、EMA、优化器），然后重新加载该文件进行无条件 DDPM 采样。生成数量为 PNG 文件数，每张图片一个样例，采样长度取自训练集的长度分布。

所有输出均相对于脚本所在的项目根目录，与启动终端的工作目录无关：

```text
weights/angel.ckpt
imgs_gen/angel/2026-10-24_15-30-45/时间_000001.png
imgs_gen/angel/2026-10-24_15-30-45/时间_000002.png
imgs_gen/angel/2026-10-24_15-30-45/时间_config.json
logs/angel/version_N/...
```

`angel.full.npz` 自动提取类型名 `angel`；普通 `cat.npz` 提取 `cat`。目录时间按北京时间计算，采用生成开始时的年月日时分秒，格式为 YYYY-MM-DD_HH-MM-SS。同类再次训练会覆盖 `weights/{类型名}.ckpt`；图片带微秒时间前缀，同一天的不同运行不会覆盖彼此。

epoch > 0 时从零训练，不自动续训；epoch=0 时只加载已有权重生成，不训练也不覆盖权重；不在本地默认启动正式训练。已用 33 条真实样本、batch_size=16、epoch=1、random_seed=42 测试，共 3 次参数更新（包含最后的单样本 batch），成功保存/加载 checkpoint 并生成 3 张独立 PNG。测试产物类别为 `smoke_angel`，不会占用 `angel.ckpt`。


## 仅生成与图片样式

`python train.py --npz_path "D:\document\DataSet\quickdraw\raw\angel.full.npz" --batch_size 32 --random_seed 1 --epoch 0 --num_images 100`

epoch=0 自动读取项目根目录的 weights/angel.ckpt，文件不存在时报错。仍需提供数据文件，用训练划分的序列长度分布确定生成长度，但不执行训练。图片使用 Matplotlib 绘制：白底、无坐标轴、等比例、圆润粗线，不同笔画按蓝/橙/绿/红等颜色循环，与参考图风格一致。

