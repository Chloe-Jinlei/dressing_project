# Genesis 本地环境

安装目录：`/home/chloe/Desktop/genesis-local`。

已验证：Genesis 1.4.3、Python 3.12.15、PyTorch 2.14.0+cu130、CUDA 13.0。
CPU 和 NVIDIA GeForce RTX 5070 Laptop GPU 的 100 步仿真均通过；Franka 模型加载和 320×240 离屏渲染通过。
渲染预览：[franka-preview.png](franka-preview.png)。

Python 环境为 `.venv`，使用项目专用、包含 C 开发头文件的 Python 3.12.15，供 Triton 编译 GPU 内核。
Genesis 源码位于 `genesis-world/`，使用可编辑安装。
PyTorch 固定为 `2.14.0`，其 Linux wheel 使用 CUDA 13.0；这样可以复用本机已有缓存。

## 启动

激活环境：

```bash
source ~/Desktop/genesis-local/.venv/bin/activate
```

在 IDE 中选择解释器：`/home/chloe/Desktop/genesis-local/.venv/bin/python`。

运行 Franka 窗口示例（默认使用 NVIDIA GPU）：

```bash
bash ~/Desktop/genesis-local/run-franka.sh
```

需要在 CPU 上运行时：

```bash
bash ~/Desktop/genesis-local/run-franka.sh --backend cpu
```

## 验证

在安装目录运行：

```bash
cd ~/Desktop/genesis-local
export GS_CACHE_FILE_PATH="$PWD/.cache/genesis"
export XDG_CACHE_HOME="$PWD/.cache"
.venv/bin/python verify_genesis.py --backend cpu
.venv/bin/python verify_genesis.py --backend cuda --franka --render
```

GPU 验证要求实际使用 CUDA，加载 Franka，执行 100 步仿真，并输出 `franka-preview.png`。
首次运行需要编译内核，耗时会比之后长。

## 树懒与衣服场景

场景入口为 `dressing_scene.py`，通过 `run-dressing.sh` 使用本目录已有环境。
默认从相邻的 `../bunnylab/config.toml` 读取资产和布局，原项目和原资产不会被修改。
包含双 Franka、0.74 m 桌面、树懒、绿色 T 恤，以及 overview、overhead、detail、左手腕和 table_side 五个相机。
`table_side` 从桌前水平看向树懒与衣服，镜头高出桌面 6 cm，视线与桌面平行；
位置 `[0,-0.90,0.80]` m、目标 `[0,-0.08,0.80]` m、视场角 45°，统一在 `../bunnylab/config.toml` 设置。
路线 A 网页同时显示三个策略视角；新增 `table_side.mp4` 和三视角合成 `triple.mp4`，原双视角录像也保留。
机器人保持原配置的 home 关节角度，夹爪张开到 0.04 m。

打开交互窗口（默认 CUDA）：

```bash
bash ~/Desktop/genesis-local/run-dressing.sh
```

CPU 运行或无窗口验证：

```bash
bash ~/Desktop/genesis-local/run-dressing.sh --backend cpu
bash ~/Desktop/genesis-local/run-dressing.sh --backend cpu --headless --steps 200 --width 640 --height 480
```

默认静置 400 步，每步 10 个 0.001 s 子步，共 4 s 仿真。
GPT 路线 A 每个动作执行 1 s，对应 100 个控制步／1000 个求解器子步，每轮输出恰好 5 个 action chunk。
动作期间每 0.01 s 更新一次插值后的关节目标。五个动作顺序执行完毕，共 5 s 仿真，取得新图像后再请求 GPT 决策；
等待模型回复时仿真暂停，因此两轮决策的现实时间间隔还包含模型延迟和仿真耗时。
窗口在静置完成后保留最终状态，关闭窗口或 Ctrl+C 退出。
首次运行会编译 Genesis 内核，CPU 速度明显慢于 GPU。
如果 CUDA 不可用，程序会明确报错；需要手动指定 `--backend cpu` 才会使用 CPU。

输出位于 `runs/<时间>-dressing-<进程号>/`，包括转换后的 OBJ、原四面体数据、
`initial-*.png`、`settled-*.png`、粒子最终状态和 `report.json`。
报告记录源资产 SHA256、配置、网格数量、实际求解器、质量、时间步长和有限状态检查。
仅检查资产时可使用 `--prepare-only`；更换源目录使用 `--source-project /path/to/bunnylab`。

历史场景验证（2026-10-07，子步 0.0005 s）：CPU 后端完成 400 步 / 2 s，四个相机的初始和结束图像均通过，
节点初始坐标、单元连接关系和质量一致性检查通过，源资产 SHA256 未变，结束时粒子没有穿过桌面。
结果：[总览图](runs/20261007-genesis-dressing-verified/settled-overview.png)、
[报告](runs/20261007-genesis-dressing-verified/report.json)。
用户随后在本机用 CUDA 完成 400 步 / 2 s（当时子步为 0.0005 s），报告见
[CUDA 场景验证](runs/20261007-141028-dressing-125861/report.json)。
当前工具执行环境仍无法访问 NVIDIA 设备或桌面显示，因此工具内的抓取任务验证使用 CPU 无窗口运行。

历史 1 ms 子步验证（当时动作时长为 0.1 s）：CPU 路线 A 完成 2 s 静置、两个各 0.1 s 的平滑往返动作，
再静置 2 s，共 4.2 s 仿真；粒子位置／速度有限，树懒与衣服均无桌面穿透。
源资产未改动，双视角图像与三份录像保存通过。模型回复使用离线夹具，此项不是实际 GPT 调用。
结果：[验证报告](../bunnylab/runs/20261007-183927-genesis-1ms-check/validation.json)、
[最终画面](../bunnylab/runs/20261007-183927-genesis-1ms-check/final-overhead.png)。

### 几何与物理对应

树懒使用原 `sloth.npz` 的 551 个节点和 2549 个四面体，保留原 UV 和 PNG 纹理。
衣服按 MuJoCo 相同的简化、尺寸和旋转规则处理：621 个节点、1199 个三角形、四个开口。
两者位置、机械臂底座位置与朝向、桌子尺寸和相机位姿沿用原配置；质量分别为 0.18 kg 和 0.055 kg。

`genesis_mesh_adapter.py` 针对 Genesis 1.4.3 的导入接口做了小范围适配，
跳过 PBD 默认重新网格化，直接建立原网格上的约束。树懒原四面体外表面有一条非流形边，
因此不能直接让 TetGen 重建；适配器保留该原始几何，只统一四面体的顶点朝向。
适配仅在添加该实体时生效，随后恢复 Genesis 的原导入类；没有改动已安装的 Genesis 源码。
更换 Genesis 版本时需要重新审查该适配器。

当前树懒为 `PBD.Elastic`，衣服为 `PBD.Cloth`，不是原 MuJoCo 的弹性模型。
质量保持一致，但 PBD 的 compliance 参数尚未标定，不能直接当作原 Young's modulus、Poisson ratio 和 damping。
树懒已调为更柔软的弹性体：`stretch_compliance` 从 `1e-7` 增至 `0.005`，
`volume_compliance` 从 `0` 改为 `1e-12`，允许更容易弯曲和受压形变。
这两个参数及松弛系数统一放在 `../bunnylab/config.toml` 的
`[scene.genesis.sloth_material]` 中，场景启动和 GPT 路线 A 的 Genesis worker 都读取此处。
保留原来的 551 个节点、2549 个四面体、纹理和 0.18 kg 质量。
重新启动场景后生效；旧的抬升录像与报告使用之前较硬的参数，不能视为当前软体参数的抬升验证。
这组软体参数曾在 0.5 ms 子步、每步 20 子步下完成 CPU 静置验证：150 步、1.5 秒，状态有限且无桌面穿透；
结果见 [验证报告](runs/20261007-sloth-soft-material/report.json) 和
[最终画面](runs/20261007-sloth-soft-material/settled-detail.png)。
PBD 共用的接触粒子直径为 4 mm，与原来树懒/衣服的两个碰撞半径和衣服厚度不同。
该版本原生的 legacy rigid/PBD 接触通过 SDF 投影粒子，未实现这个分支的摩擦。
当前场景已用 `genesis_mesh_adapter.py` 中的版本检查适配器补上静／动摩擦：
在每个 0.001 s 子步内，仅对发生接触的自由粒子按法向投影量限制切向修正，
相对速度考虑接触刚体的运动，并将反作用力传回刚体。系数取粒子与刚体的较小值；
桌面对树懒使用 0.7、对衣服使用 0.6。算法采用
[PBD 接触摩擦论文](https://mmacklin.com/uppfrta_preprint.pdf) 的位置约束形式。
适配器仅作用于当前场景，不修改安装的 Genesis 源码；物体没有被固定到桌面或逐步重置位置／速度。
PBD 粒子之间仍有摩擦和粒子级自接触，也不能保证衣服层之间完全不穿透。
报告现在记录物体中心、中心速度、粒子 RMS／最大速度和初始静置的分段记录；
有限状态与无桌面穿透检查本身不代表物体已经静止。

2026-10-08 静置专项检查：原资产、1 ms 子步、双臂保持初始关节目标，无 GPT 动作或抓取约束。
先自然静置 2 s，再观察 5 s。修复前后物体中心的水平漂移如下：

| 物体 | 原生无摩擦接触 | 当前摩擦适配 |
| --- | ---: | ---: |
| 树懒 | 2.03 mm | < 0.001 mm |
| 衣服 | 1.96 mm | 约 0.007 mm |

衣服局部仍有低幅抖动：观察期间最大节点位移约 0.68 mm，不能声称所有节点完全静止。
两个场景均无桌面穿透，机械臂关节位置保持不变。见
[原生对照](runs/20261007-rest-friction-check/native/report.json)、
[最终代码静置检查](runs/20261007-rest-friction-check/patched-final/report.json)。
另有实际 CPU 接触检查：零摩擦维持滑动、静摩擦消除微小切向速度、
大切向速度减速后停下、非接触粒子正常自由落体，见
[接触报告](runs/20261008-contact-friction-test/report.json)。
修复后的实际 Genesis 路线 A 子进程完成 2 s 静置和五个各 1.000 s 的有界动作，
三视角正常出图，场景没有非有限状态或桌面穿透；见
[流程报告](../bunnylab/runs/20261008-genesis-friction-bridge/report.json)。该流程检查未调用 GPT。
这些是 CPU 物理检查，不代表 CUDA、真实材料参数或穿衣任务已经验证。

MJCF 夹爪 tendon 在 Genesis 中近似为两个关节驱动，并用位置控制保持张开。

默认模式验证场景导入、静置和渲染。`--task lift-sloth` 提供下面的双臂抬升任务。
GPT 路线 A 已通过下面的统一入口接入；脚本抬升本身仍不调用 GPT，穿衣成功需人工判定。
后续若需要与原材料参数更接近的薄壳 FEM，可考虑 Genesis FEM.Cloth + IPC，
但它需要额外的 `pyuipc` 依赖和独立验证，当前环境未安装该依赖。
求解器说明：[Genesis 官方软体求解器文档](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/soft_solvers.html)。

## GPT 穿衣路线 A

材料参数可以先用当前默认值，标定不是接入 GPT 的前提。
复用 GitHub 项目的 Codex 会话、三视角网页、有界动作和录像流程，只把物理执行切换到 Genesis：

```bash
cd ~/Desktop/bunnylab
uv run dressing route-a --simulator genesis --open-browser
```

CPU 模式显式加 `--genesis-device cpu`。真实模型与会话记忆检查加 `--smoke-test`。
这些命令需要独立版 Codex CLI、自己的登录／模型权限、bubblewrap 和 ffmpeg，与原路线 A 相同。
同一次实验持续同一个 GPT 对话：默认观察俯视／腕部／桌面侧视图像和机械臂状态，每轮输出五个小幅平移／旋转／开合动作，
经过验证后由 Genesis 顺序执行，再反馈新图像。默认每个 chunk 在 2 cm 内、0.1 rad 内，执行 1 s 仿真。
每个增量相对该 chunk 开始时的实际机器人姿态；等待则输出 `status="wait", actions=null`。
单个 chunk 验证失败、补充指令或 Esc 会停止余下动作，下一轮反馈已执行及未完成的 chunk。
修改时长统一使用 `../bunnylab/config.toml` 的 `[control].duration_seconds`；求解器子步仍为 0.001 s。
GPT worker 的双臂启用重力补偿，与脚本抬升使用相同设置，避免 1 s 动作的静态下沉累计。
网页的“本任务成功 / 本任务失败”只记录当前指令结果并暂停，保留当前姿态、抓取和同一个 GPT 会话；
可以继续发送“保持树懒的抓取，把衣服提起来”等下一条指令。
逐条人工判定保存在 `task_results.json`，下一轮也提供给模型。
“结束整场实验并保存视频”、Esc 或关闭页面才终止整场实验；整场人工结果独立记录。
若侧视只供人工观察，将 `[observation].cameras` 改为 `["overhead", "wrist"]`；侧视仍保留在网页和录像中。

Genesis 继续使用本目录自己的环境；统一入口用进程间接口传递允许的观测和动作，
不把物体网格、真实材质或接触诊断发给 GPT。结果统一写入 `bunnylab/runs/<run-id>/`。
闭爪双指垫实际接触树懒或衣服时，使用局部无滑动 PBD 约束近似抓取，张爪释放。
这使流程可尝试抓取，但纯摩擦夹持、材料真实性和穿衣成功仍需后续验证。
完整流程与限制见 [bunnylab README](../bunnylab/README.md)。

本次检查已通过真实 CPU Genesis 的小幅往返、旋转 IK、动作拒绝和三份录像，
另有 24 项项目测试和 4 项本目录接触／释放测试通过，包括五动作顺序执行、部分拒绝、补充指令／退出取消和不同步长下的平滑动作。
五个 1 s chunk 的真实 CPU 流程完成 7 s 仿真（含 2 s 静置），反馈与三份录像保存通过；
[批次报告](../bunnylab/runs/20261007-185024-genesis-five-chunks/validation.json)使用离线模型回复和 320×240／2 fps，记录的是重力补偿修正前的流程。
随后[重力补偿检查](../bunnylab/runs/20261007-185800-genesis-gravity-hold/validation.json)通过：
1 s 水平动作的垂直偏移约 0.14 mm，两臂静止保持 1 s 的末端漂移小于 0.002 mm；粒子有限且无桌面穿透。
独立 Codex 调用因工具环境无法连接服务而失败；实际网页 TCP 监听也受工具环境限制。
这两项在本机联网终端验证，默认材料参数不阻挡接入。
测试接触规则可运行 `.venv/bin/python test_genesis_route.py -v`。

新增侧视相机的 CPU 检查通过：三个真实 640×480 视角、水平视线、高度和五份录像输出均通过；
29 项项目测试通过。本次没有重复物理基准或调用真实 GPT。
见[侧视预览](../bunnylab/runs/20261007-205343-table-side-camera/preview-table_side.png)和
[验证报告](../bunnylab/runs/20261007-205343-table-side-camera/validation.json)。

## 双臂抓取与抬升

使用同一个场景和启动入口：

```bash
bash ~/Desktop/genesis-local/run-dressing.sh --task lift-sloth --substeps 20
```

CPU 无窗口录制：

```bash
bash ~/Desktop/genesis-local/run-dressing.sh --task lift-sloth --substeps 20 --backend cpu --headless --width 640 --height 480
```

动作包括静置、左右臂分别接近、双臂下降、闭爪、同步抬升和空中保持。
抬升动作复用场景的 `[scene.genesis.sloth_material]` 软体参数，
本次材料设置同时记录在 `report.json` 的 `sloth_soft_body` 和 `lift_task.sloth_material` 中。
默认抬升 0.22 m、保持 1.5 s，可通过 `--lift-height` 和 `--hold-seconds` 调整。
机械臂通过 IK 和关节位置控制运动，并启用机械臂重力补偿以减少腕部静态下沉。
抓取点根据当前两侧表面节点配对选择，指垫中心距桌面至少 17 mm，以留出指尖间隙。
下降后重新根据表面对齐；一侧未通过近接触检查时，张开该侧夹爪、重新对齐并重试，最多三次。
树懒始终由 Genesis 仿真推进，没有直接设置物体的位置。
脚本用仿真几何选择两侧肢体的抓取点，因此这是脚本控制实验，不是视觉或 GPT 策略。

抓取方式为**接触触发的局部 PBD 约束**：闭爪时，必须先检查每只夹爪的两个指垫距树懒
表面节点不超过 4 mm，再把每个指垫附近最多四个节点约束到对应的手指 link。
这里使用几何距离作为近接触判据，没有测量指垫对树懒的夹持力。
其他节点继续运行原软体仿真。该约束近似无滑动抓取；当前版本的 rigid/PBD 摩擦未实现，
且 link 附着约束不会把负载反作用传给机械臂，因此该演示不验证纯摩擦夹持或夹持力。

每次运行的输出目录包含 `lift-process.mp4`（总览与近景、中文阶段标注）、
`stage-*.png`、`trajectory.jsonl` 和 `report.json`。报告记录接触检查、附着节点数、
整个树懒的离桌高度、质心抬升、保持期间的漂移和网格拉伸；只有全部任务检查通过才打印 PASS。
上述命令每步采用 20 个 0.001 s 子步，控制步长为 0.02 s；`--substeps` 可显式调整。

2026-10-07 CPU 完整验证通过：1032 步 / 10.32 s，双臂近接触检查通过，
空中保持 1.5 s；保持阶段树懒最低点距桌面约 20.05 cm，质心抬升约 21.51 cm，
记录到的质心漂移约 0.31 mm。每步检测均无两臂互碰，最大网格边长增加约 5.01%，
源资产一致性检查通过。仅 6 / 551 个树懒表面节点附着于手指，其他节点继续软体仿真。
成果：[完整视频](runs/20261007-sloth-lift-contact-03/lift-process.mp4)、
[动画预览](runs/20261007-sloth-lift-contact-03/lift-preview.gif)、
[最终状态](runs/20261007-sloth-lift-contact-03/stage-lifted.png)、
[验证报告](runs/20261007-sloth-lift-contact-03/report.json)。

2026-10-07 新软体参数的完整抬升验证也已通过：`stretch_compliance = 0.005`、
`volume_compliance = 1e-12`，CPU 共 1028 步 / 10.28 s，子步为 0.0005 s。
双臂接触检查通过，6 / 551 个表面节点附着于手指，其余节点继续软体仿真。
空中保持 1.5 s；最低点距桌面至少 12.80 cm，中心抬升 19.70 cm，
保持期间中心漂移 1.45 mm，最大网格边长增加 13.67%，两臂没有互碰。
源网格、质量和资产一致性检查全部通过。新的结果：
[完整视频](runs/20261007-sloth-soft-lift/simulation/lift-process.mp4)、
[动画预览](runs/20261007-sloth-soft-lift/simulation/lift-preview.gif)、
[最终状态](runs/20261007-sloth-soft-lift/simulation/stage-lifted.png)、
[验证报告](runs/20261007-sloth-soft-lift/simulation/report.json)。

## 双臂穿衣（开发验证中）

同一入口新增 `--task dress-sloth`，先完成双臂抬升，再移动、支撑树懒，
重新抓持后脚、抓取衣服前片下摆、撑开开口、尝试穿入并释放。树懒继续使用配置中的
`PBD.Elastic` 软体参数，衣服继续使用原来的 `PBD.Cloth`、0.055 kg 质量和四个开口。

```bash
bash ~/Desktop/genesis-local/run-dressing.sh --task dress-sloth --substeps 20
```

也可以从已有、材料及源资产完全一致的成功抬升记录开始后续动作：

```bash
bash ~/Desktop/genesis-local/run-dressing.sh --task dress-sloth --substeps 20 --continue-lift ~/Desktop/genesis-local/runs/20261007-sloth-soft-lift/simulation
```

续跑仅在初始化时恢复保存的粒子、关节位置并将速度置零，随后重新建立已记录的局部抓取；
初始化与连续动力学复现有区别，会在报告中明确记录。控制器不把物体设置到目标姿态。
穿衣模式额外用衣服顶点、边中点和面中心与树懒三角形表面建立采样接触约束：
按节点逆质量施加双方碰撞修正，被夹爪固定的节点不参与位移修正。
接触约束在每个物理子步与原有伸长、体积约束交替求解，位置修正按相邻最短边限制，
避免只在外层控制步末修正而造成布片过度拉伸。这不是连续三角形碰撞，也不是把衣服移动到目标姿态。
实现使用 Genesis 1.4.3 的求解器字段及内核，未修改安装的 Genesis 源码；报告记录额外迭代和接触缓存。
0.6 秒布料落下碰撞检查通过：无超过 3 mm 的穿透采样点，布料最大边长比 1.238，
树懒体积比 0.993–1.000；见 [接触检查报告](runs/20261007-sloth-dressing/substep-contact-check/report.json)。
该检查验证碰撞稳定性，不代表穿衣成功。
穿衣模式、默认静置和 GPT 路线 A 都使用上面的子步级刚体／PBD 摩擦适配器。
穿衣模式原先按重力估算法向支撑冲量的外层速度修正已移除，避免重复施加摩擦。
该适配仍未验证真实接触压力或纯摩擦夹持负载；局部无滑动抓取约束继续用于抓取。
带抓取元数据的阶段快照可用 `--continue-dressing <state-阶段.npz>` 续跑；
它保存节点和关节状态、抓取节点及手部目标朝向，不等同于完整求解器检查点。
`dressing-process.mp4` 录制后续动作，`state-*.npz` 保存各阶段的粒子及关节状态。
穿衣完成要求躯干包覆比例达到 80%、头穿过领口、两侧肢体分别穿过袖口，
并通过深度穿透、网格拉伸、释放和两臂碰撞检查；衣服覆盖画面不能单独作为成功判据。
当前流程仍在物理验证，未通过的运行会保存失败诊断并报错，不会标为穿衣成功。

## 继续安装

```bash
bash ~/Desktop/genesis-local/install.sh
```

该脚本复用已有环境和仓库，不会再次触发 `.venv` 覆盖确认。
安装使用本机的 uv 下载缓存和硬链接，减少 CUDA 库的重复磁盘占用。
依赖快照、源码提交和验证日志分别保存在 `installed-packages.txt`、`installed-commit.txt`、`verify-cpu.log` 和 `verify-cuda.log`。

官方说明：https://github.com/Genesis-Embodied-AI/genesis-world#quick-installation
