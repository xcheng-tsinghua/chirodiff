# 单类别无条件生成

编辑 `train_single_category.py` 顶部的 `CONFIG`：

```python
CONFIG = {
    'npz_path': r'D:\document\DataSet\quickdraw\raw\angel.full.npz',
    'batch_size': 128,
    'random_seed': 5555,
    'epoch': 200,
    'num_images': 16,
}
```

激活 dp 后运行 `python train_single_category.py`。服务器上修改路径，并使用已配置好 `requirements-dp.txt` 和 CUDA PyTorch 的环境。

只读取 NPZ 内的 `train_split`；没有该键时读取 `train`（当前 angel 文件的实际键名）。不会读取 valid/test，也不从训练划分另切验证集。使用全部符合预处理条件的训练样本；过滤规则沿用已验证的数据适配：长度 5–300、有限坐标、非退化轨迹。输入为 Sketch-RNN 的 `[dx, dy, pen_up]`。

训练达到指定总轮数后，保存最后一轮的完整 checkpoint（包含模型、EMA、优化器），然后重新加载该文件进行无条件 DDPM 采样。生成数量为 PNG 文件数，每张图片一个样例，采样长度取自训练集的长度分布。

所有输出均相对于脚本所在的项目根目录，与启动终端的工作目录无关：

```text
weights/angel.ckpt
imgs_gen/angel/2026-10-24/时间_000001.png
imgs_gen/angel/2026-10-24/时间_000002.png
imgs_gen/angel/2026-10-24/时间_config.json
logs/angel/version_N/...
```

`angel.full.npz` 自动提取类型名 `angel`；普通 `cat.npz` 提取 `cat`。日期按北京时间计算，采用生成开始时的日期。同类再次训练会覆盖 `weights/{类型名}.ckpt`；图片带微秒时间前缀，同一天的不同运行不会覆盖彼此。

本入口每次从零训练，不自动续训；不在本地默认启动正式训练。已用 33 条真实样本、batch_size=16、epoch=1、random_seed=42 测试，共 3 次参数更新（包含最后的单样本 batch），成功保存/加载 checkpoint 并生成 3 张独立 PNG。测试产物类别为 `smoke_angel`，不会占用 `angel.ckpt`。
