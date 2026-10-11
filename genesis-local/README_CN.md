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
两者位置、机械臂底座位置与朝向、桌子尺寸和相机位姿沿用原配置。树懒质量为 0.18 kg；
Genesis 衣服按棉布面密度与网格面积计算，当前约 0.02050 kg，原 0.055 kg 仅保留用于 MuJoCo。

`genesis_mesh_adapter.py` 针对 Genesis 1.4.3 的导入接口做了小范围适配，
跳过 PBD 默认重新网格化，直接建立原网格上的约束。树懒原四面体外表面有一条非流形边，
因此不能直接让 TetGen 重建；适配器保留该原始几何，只统一四面体的顶点朝向。
适配仅在添加该实体时生效，随后恢复 Genesis 的原导入类；没有改动已安装的 Genesis 源码。
更换 Genesis 版本时需要重新审查该适配器。

当前树懒为 `PBD.Elastic`，衣服为 `PBD.Cloth`，不是原 MuJoCo 的弹性模型。
不能把 PBD 的 compliance 直接当作原 Young's modulus、Poisson ratio 和 damping。
当前树懒 `stretch_compliance=0.02`、`volume_compliance=1e-14`，
拉伸松弛系数 0.3、体积松弛系数 0.05；它是可变形填充玩具的近似，未用实物标定。
这两个参数及松弛系数统一放在 `../bunnylab/config.toml` 的
`[scene.genesis.sloth_material]` 中，场景启动和 GPT 路线 A 的 Genesis worker 都读取此处。
保留原来的 551 个节点、2549 个四面体、纹理和 0.18 kg 质量。
重新启动场景后生效；旧的抬升录像与报告使用之前较硬的参数，不能视为当前软体参数的抬升验证。
旧版软体参数曾在 0.5 ms 子步、每步 20 子步下完成 CPU 静置验证：150 步、1.5 秒，状态有限且无桌面穿透；
结果见 [验证报告](runs/20261007-sloth-soft-material/report.json) 和
[最终画面](runs/20261007-sloth-soft-material/settled-detail.png)。
PBD 共用的接触粒子直径为 4 mm，与原来树懒/衣服的两个碰撞半径和衣服厚度不同。

2026-10-10 棉布与软体修复：`[scene.genesis.garment_material]` 使用
[Shibata 与 Hirai，Robotics 2016，Table 3 Cotton A](https://doi.org/10.3390/robotics5020010)
的面密度 0.0211 g/cm²（211 g/m²）及 KES 弯曲刚度 0.0567 gf·cm²/cm
（5.56036655e-6 N·m）。这是特定棉布样本，不能代表所有棉布或针织 T 恤。
原网格面积约 0.0971593 m²，质量约 20.50 g；顶点质量按三角形面积分配。
拉伸刚度 600 N/m 是独立的各向同性膜近似，**不是该论文实测拉伸数据**；
摩擦、空气阻力和树懒的填充材料同样尚未实测标定，不能声称与特定实物完全一致。
默认不模拟织物经纬各向异性、非线性拉伸或弯曲滞回。

实际网格上的边弹簧柔顺度为 `3*l²/(4*S*A_adj)`；铰链柔顺度为
`2*A_adj/(3*B*l²)`（B 使用 N·m）。后者来自相邻三角形重心距离
`d=(h1+h2)/3` 下的薄壳能量 `B*l*theta²/(2*d)`。
这些是离散近似，并非把杨氏模量的倒数直接赋给所有网格边。
场景适配器修正角度梯度的归一化，加入每子步内累计的 XPBD 乘子；
衣服拉伸迭代 32 次，树懒拉伸保持 8 次，体积迭代 32 次。还修正原体积位移中重复乘梯度平方的问题，
避免软化树懒后体积约束几乎不生效。已安装的 Genesis 源码未修改。
刚体接触投影后，增加接触方向受限的体积不等式约束（目标最小体积为原单元的 10%），
防止少量贴桌面薄四面体被投影压平或翻转；它允许节点离开接触面或沿面移动，
不把物体重置到原形，也不指定动画。该稳定化未验证附加的机器人反作用力。
`report.json` 的 `cotton_material` 记录实际逐边参数范围，`genesis.materials`
中的布料标量 compliance 仅是分配字段前的初始值。

验证输出在 `runs/cotton-material-validation/`：`verified/deformation.mp4`
使用原始衣服和树懒，仅固定局部小片，其余顶点受重力自由运动，不逐帧指定形变。
3 秒时扣除最佳刚体变换后的 RMS 形变为衣服约 47 mm、树懒约 5.05 mm；
树懒体积比约 1.00011，采样帧无四面体翻转，衣服边伸长比的 99 分位数约 1.068。
这是材料悬挂测试，不是机器人抓取或穿衣成功验证。
`verified-hinge/report.json`、`verified-volume/report.json` 的弯曲和体积单步结果
与独立数值能量梯度相符。`verified-scene/report.json` 的完整桌面场景
完成 5 秒 CUDA 仿真，采样检查及最终状态无倒置或压平单元、无桌面穿透，五视角正常。
控制端 51 项测试、5 项控制与材料单元测试、3 项实际 GPU 材料测试通过。
重启网页服务后才加载新代码及参数，已有视频和已启动进程保持原行为。

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
同一次实验持续同一个 GPT 对话，只输入一个完整目标，由模型自主规划并循环操作。
每轮观察俯视／腕部／桌面侧视图像和机械臂状态，输出五个有界动作；默认每个 chunk 在 2 cm 内、
0.1 rad 内，执行 1 s 仿真。动作验证失败会向同一会话反馈有限的拒绝原因，自动重试。
`wait` 现在表示短暂推进仿真后自动重新观察；`completed` 和 `unable` 结束本场操作并保存录像。
模型完成声明与人工成功判定分开记录；本版本未添加自动几何成功判定。
默认预算为 600 个动作（短暂等待也占名额）、30 分钟现实时间、连续 5 次错误或 8 轮无明显可见变化。
配置及准确行为见 [自主任务网页说明](../bunnylab/README.md#路线-a-自主任务网页)。
网页的人工成功／失败按钮记录最终结果并停止任务；“新实验：全部归位”在保存后重建原场景、
建立空白 GPT 会话和新的运行目录，旧录像与记录保留。刷新／关闭页面不会重置后台实验；
点击“保存并退出服务”或终端 Ctrl-C 后，再次运行启动命令会创建新实验。
修改动作时长仍统一使用 `../bunnylab/config.toml` 的 `[control].duration_seconds`；
求解器子步仍为 0.001 s。GPT worker 的双臂重力补偿、场景和相机配置保持不变。
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
衣服与树懒现在在网页路线 A、默认静置和脚本穿衣中共用面接触求解器：
每个物理子步检查双向顶点–三角面和边–边接触，用扫掠包围盒和前一子步的所在侧检查跨面运动，
按逆质量给两者施加相反修正；固定的抓取节点保持固定。默认表面间隙 1 mm，
参数位于 `config.toml` 的 `[scene.genesis.surface_contact]`。
可视网格逐点绑定对应的物理节点，避免前后布片被邻点蒙皮混合。
接触缓存溢出或迭代后仍有严重未解决接触时会报错停止，不会静默丢弃约束。
旧的仅脚本穿衣使用的控制步采样接触已移除，避免重复求解。
实现使用 Genesis 1.4.3 的求解器字段及内核，未修改安装的 Genesis 源码。
这不是任意速度、任意变形下有数学保证的连续碰撞检测；布料自身仍使用原生粒子自碰撞。
衣服原始网格已有缝合的前后两片，内部中空，保留下摆、领口和两个袖口共四个开口。

碰撞回归使用 5 m/s 布片撞击大三角面，以及原始衣服落到原始软体树懒上的 1.5 s 仿真。
后者同时检查布点/面中心的内部深度、双向网格边与三角面的相交、树懒体积和桌面接触。
运行：

```bash
cd /home/icebear/Desktop/dressing_project/genesis-local
GENESIS_SURFACE_TEST_RUN_DIR="$PWD/runs/surface-contact-check" .venv/bin/python -m unittest test_genesis_route.SurfaceCollisionIntegrationTests
```

每次使用新的输出目录。验证产物见 `runs/surface-contact-validation/verified/`。
这些检查验证接触行为，不代表穿衣任务成功。
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

## 独立衣服开口诊断

`check_garment_opening.py` 是独立的环境能力检查，不修改 GPT 提示词或网页策略。
脚本用原始网格选取下摆抓取点，通过网页同一个 `GenesisController` 的
动作检查、关节控制和接触抓取尝试提起一片、再抓另一片并分离。
不直接设置物体位置、不增加额外的布料固定点、不改变碰撞或材料。
它使用仿真几何和更频繁的反馈，因此结果不能作为 GPT 自主规划的成绩。

```bash
cd /home/icebear/Desktop/dressing_project/genesis-local
GS_CACHE_FILE_PATH="$PWD/.cache/genesis" XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .venv/bin/python check_garment_opening.py --grasp-x .098 --output runs/my-opening-check
```

每次指定一个新的输出目录。`opening.mp4` 包含俯视和侧视，`report.json` 记录
失败原因、抓取是否混合前后片及开口指标，`measurements.jsonl` 保存逐帧数据，
`script.py` 保存当次脚本版本。动作接触使用原有无滑动绑定近似。
投影开口面积和内切圆半径只作为筛查指标；投影自交时不判定开口成功，
投影非自交也不能单独证明三维通道畅通或衣服自碰撞正确。
一次失败只能说明该次操作未成功，不能证明所有可行策略都无法打开衣服。
