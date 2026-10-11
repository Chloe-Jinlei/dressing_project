# Dressing Project

- [主项目与 GPT 实验说明](bunnylab/README.md)
- [Genesis 场景与环境说明](genesis-local/README_CN.md)

克隆时请使用 git clone --recurse-submodules。

本仓库包含项目代码、场景 assets 和配置。
虚拟环境需要在本机重新安装。

## 本机安装与启动

在仓库根目录执行（需要已安装 `uv`）：

```bash
uv sync --locked --project bunnylab
bash genesis-local/install.sh
```

控制端使用 `bunnylab/.venv`；Genesis 使用 `genesis-local/.venv`，安装脚本会为新克隆创建包含 C 头文件的 Python 3.12 环境，并验证 CPU、CUDA 和离屏渲染。完整安装验证需要可用的 NVIDIA GPU。

启动 GPT / Genesis 实验：

```bash
cd bunnylab
uv run dressing doctor
uv run dressing route-a --simulator genesis --open-browser
```

网页输入一个完整目标后会自主循环，默认最多 600 个动作或 30 分钟；结束后人工判定。“新实验：全部归位”重建原场景并创建新 GPT 会话，保留旧记录。刷新页面不会重置任务。

仅打开 Genesis 场景（不调用 GPT），在仓库根目录执行：

```bash
bash genesis-local/run-dressing.sh
```

路线 A 使用本机已登录的独立 Codex CLI，以及 `bwrap` 和 `ffmpeg`；不需要设置单次 API 接口的 `OPENAI_API_KEY`。模型配置见 `bunnylab/config.toml` 的 `[route_a]`。省略 `--simulator genesis` 会使用 MuJoCo。

下级 README 中的 `/home/chloe/`、`~/Desktop/genesis-local` 和 `~/Desktop/bunnylab` 是原机器的历史路径；本仓库请使用上述相对路径。历史运行报告未随仓库全部提供，不能当作本机验证结果。

本机验证（2026-10-10）：RTX 4070 SUPER；Genesis 1.4.3、Python 3.12.14、PyTorch 2.14.0+cu130。控制端 33 项测试通过，Genesis 控制规则 4 项通过（独立 CPU 摩擦积分测试未运行）；bubblewrap 隔离检查通过；CPU/CUDA 各 100 步基础验证与 Franka 离屏渲染通过。实际 CUDA dressing 场景完成 200 步 / 2 秒，五视角正常，状态有限、无桌面穿透且源资产未改动。报告见 `genesis-local/runs/setup-20261010-cuda/report.json`，测试日志见 `bunnylab/runs/setup-20261010/`。这些本机输出不纳入版本控制。本次未调用真实 GPT，也未验证穿衣成功。

GPT + MPPI 对照实验（复用网页、软体和三相机）：

```bash
cd bunnylab
uv run dressing route-a --simulator genesis --controller gpt-mppi --open-browser
```

新模式默认端口 8766，原模式为 8765。当前 MPPI 使用末端运动学预测，
不预测软体变形或接触；配置、记录和实验边界见 [GPT + MPPI 说明](bunnylab/README.md#gpt--mppi-对照实验路线-b)。
