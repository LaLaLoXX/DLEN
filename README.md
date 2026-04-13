# DLEN — 低光图像增强

**GitHub 仓库**：[https://github.com/LaLaLoXX/DLEN](https://github.com/LaLaLoXX/DLEN)

```bash
git clone https://github.com/LaLaLoXX/DLEN.git
```

基于 [BasicSR](https://github.com/xinntao/BasicSR) 改写的低光增强训练与测试代码库。网络主体为 **DLEN**（含光照估计、去噪与幅度分支等模块），在本仓库内独立实现与命名。

## 功能概览

- **训练 / 验证**：`ImageCleanModel` + 成对数据 `Dataset_PairedImage`，支持多 GPU（`torchrun`）。
- **测试**：标准 BasicSR 测试入口，按 YAML 配置数据集与指标。
- **预置配置**：LOL-v1、LOL-v2（Synthetic / Real）等 YAML 位于 `Options/`。

## 环境要求

- Python 3.8+（建议与 PyTorch 官方说明一致）
- PyTorch（需与 CUDA 版本匹配）
- 常见依赖见下方「依赖安装」

## 依赖安装

本仓库的 `setup.py` 未维护完整的 `install_requires`，建议先安装 PyTorch，再安装其余包，最后在项目根目录以可编辑方式安装本包：

```bash
pip install einops opencv-python numpy pyyaml tqdm scikit-image scipy natsort tensorboard wandb
pip install -e . --no_cuda_ext
```

说明：

- **`--no_cuda_ext`**：当前仓库未包含 BasicSR 原版的 CUDA 算子扩展源码；若你自行补全 `basicsr/models/ops` 并具备编译环境，可去掉该参数尝试编译扩展。
- 若使用 TensorBoard / WandB 日志，请确保已安装对应包并在 YAML 中按需开启。

## 数据准备

按所选 YAML 中的路径放置数据。例如 `Options/DLEN_LOL_v1.yml` 使用：

- `dataroot_gt`：正常光照图像根目录  
- `dataroot_lq`：低光图像根目录  

请将 YAML 内路径改为你本机数据目录，或保持相对路径并在项目根下建立 `data/` 结构。

## 训练

在项目根目录执行（单卡示例）：

```bash
python basicsr/train.py --opt Options/DLEN_LOL_v1.yml
```

默认配置见 `basicsr/train.py` 中的 `--opt` 默认值，也可显式指定其他 YAML。

### 多卡训练

可参考 `train_multigpu.sh`（Linux）。示例：

```bash
bash train_multigpu.sh Options/DLEN_LOL_v1.yml 0,1 4321
```

其中参数依次为：配置文件路径、可见 GPU 列表、`master_port`（多任务时请错开端口）。

Windows 下可改用等价命令，例如：

```bash
set CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=4321 basicsr/train.py --opt Options/DLEN_LOL_v1.yml --launcher pytorch
```

## 测试

使用 BasicSR 测试脚本，需准备与 YAML 中 `datasets` 一致的测试集路径：

```bash
python basicsr/test.py --opt Options/DLEN_LOL_v1.yml
```

测试前可在对应 YAML 的 `path` 中设置 `pretrain_network_g` 指向你的权重文件。

## 配置文件说明

| 文件 | 说明 |
|------|------|
| `Options/DLEN_LOL_v1.yml` | LOL-v1 成对数据训练/验证 |
| `Options/DLEN_LOL_v2_synthetic.yml` | LOL-v2 Synthetic |
| `Options/DLEN_LOL_v2_real.yml` | LOL-v2 Real |

网络类型在 YAML 的 `network_g.type` 中应为 **`DLEN`**，与 `basicsr/models/archs/DLEN_arch.py` 中类名一致。

## 目录结构（节选）

```
├── basicsr/              # 训练、测试、数据、模型与工具
│   ├── train.py
│   ├── test.py
│   ├── data/
│   └── models/
│       └── archs/        # DLEN、LWN、SEB 等结构定义
├── Options/              # 实验 YAML
├── Enhancement/          # 增强相关脚本与工具（如数据集测试流程）
├── train_multigpu.sh     # 多卡启动示例
├── setup.py
├── VERSION
└── LICENSE.txt
```

## 致谢

- [BasicSR](https://github.com/xinntao/BasicSR) 提供的训练框架与工程组织方式。  
- 低光增强领域公开数据集与相关先行工作(详见引用)对本项目的启发。

## 许可证

见仓库根目录 [LICENSE.txt](LICENSE.txt)。

---

如需在 README 中补充论文引用、预训练权重下载链接或实验表格，可将对应信息发给我以便更新本节。
