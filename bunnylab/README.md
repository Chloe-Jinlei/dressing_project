# Dressing Lab

穿衣路线 A/B 的独立 uv 开发项目。只有一个 CLI 和一个配置文件。
当前用户明确指定的场景是：双 Franka Panda、桌面、DeformGen/PhysTwin 树懒软体和带袖 T 恤。
兔子任务仍未验证；树懒场景不会记作兔子穿衣成功。路线 A 已接通视觉决策与末端控制，路线 B 规划器尚未接通。

## 路线 A 三视角实验

```bash
uv run dressing route-a --open-browser
uv run dressing route-a --smoke-test
```

第一条完成 GPU 沉降检查后打开本地三视角界面。输入自然语言任务后开始决策；
可以追加指令。“本任务成功 / 本任务失败”记录当前指令的人工结果，并暂停剩余动作；
保留场景、夹爪抓取和同一个 GPT 会话，随后直接输入下一条指令即可继续。
例如先“把树懒抓起来”，点击“本任务成功”，再发送“保持树懒的抓取，把衣服提起来”。
Esc、“结束整场实验并保存视频”或关闭页面才结束整场实验。
整场结果可在结束前单独选择；某条指令成功不会自动标记整场实验成功。
等待指令与模型思考期间暂停物理，录像按墙钟时间继续记录。
结束后保留下载页面，关闭页面退出服务。终端 Ctrl-C 也会保存录像。

每次实验通过已登录 ChatGPT 的本机 Codex CLI 创建一个全新独立对话，默认模型为
`gpt-6-astra`，不继承开发对话。实验内所有步骤通过同一个 thread ID 继续对话，
追加批准的图像、机器人状态、指令和执行反馈；下一次实验使用新的隔离会话目录。
`policy/` 保存本次会话历史和 ID，不能把旧版本每步新建对话的实验中途转换成新版。需要本机 `codex`、`bwrap` 和 `ffmpeg`。
隔离进程不挂载工程、场景、物性或其他对话；禁用工具，仅提供俯视/腕部/桌面侧视 RGB、
相机标定、机器人末端位置与姿态、夹爪开合、用户指令、上次动作及其执行前图像，以及上一组动作的执行反馈。
当前图像就是上次执行后图像。腕部相机安装位置固定，会随机械臂运动。
桌面侧视相机 `table_side` 位于桌前 `[0,-0.90,0.80]` m，对准 `[0,-0.08,0.80]` m，
视线与桌面平行，镜头高于 0.74 m 桌面 6 cm，垂直视场角 45°。
相机位置、目标和视场角在 `[scene.cameras.table_side]` 中调整；保持位置和目标的 Z 相同即可保持水平视线。
默认 GPT 同时使用三个视角。若只希望显示侧视画面，可把 `[observation].cameras` 改回 `["overhead", "wrist"]`；网页和录像仍显示侧视。
此入口不使用项目的 `OPENAI_API_KEY`，单次 API 接口仍单独保留。

模型每轮输出 `status="act"` 和恰好包含 5 个动作的 `actions` 数组，或 `status="wait", actions=null`。
每个 action chunk 执行 1 秒仿真，一轮完整执行 5 秒后再向 GPT 提交新图像。
默认每个 chunk 平移不超过 2 cm、旋转不超过 0.1 rad；位移和旋转相对该 chunk 开始时的实际机械臂姿态。
先检查整组动作的数值限制，再逐个检查工作空间、IK 和关节增量，然后顺序执行。
某个 chunk 被拒绝时取消后续动作；判定当前任务、补充指令或 Esc 在下一个控制更新点停止当前及后续动作。
完成与部分执行分别反馈，逐个记录实际执行时间。每条指令最多 100 个 chunk，即最多 20 个完整批次。
视频是 15 fps 墙钟录像，不能当作实时物理速度。

每次实验保存到 `runs/<run-id>/`：`triple.mp4`、`dual.mp4`、`overhead.mp4`、`wrist.mp4`、`table_side.mp4`、
`session.json`、`task_results.json` 中逐条任务的人工判定、逐帧时间线、指令和 `steps/` 下的批准观测、独立对话 ID、决策与执行记录。
每轮 `steps/<轮次>/execution.json` 汇总五个 chunk，`chunks/<序号>/execution.json` 保存各自的执行记录。
人工成功/失败/未判定单独记录，不用模型的文字声明替代任务评估。
下一条指令的模型输入会包含已有指令的人工判定，按每条指令分别说明结果。

`--smoke-test` 使用真实独立模型调用要求保持不动，再由脚本测试左手往返各 1 cm，
检查 GPU 控制和五个视频，并在同一对话追加一轮记忆验证；这不是模型完成穿衣任务。
当前软体接触与自碰撞仍有限制，见下文；尚无穿衣成功验证。

### 在 Genesis 中运行同一条路线 A

材料参数标定不是接入 GPT 的前提。可以先用当前默认参数完成
“三视角观察 → 同一实验对话决策 → 有界动作验证 → Genesis 执行 → 再观察”的闭环。
模型、Codex 登录、会话延续、任务追加、网页和录像逻辑与上面的 MuJoCo 路线 A 共用。

在本项目目录运行（默认 Genesis CUDA）：

```bash
uv run dressing route-a --simulator genesis --open-browser
```

明确使用 CPU，或验证真实模型接口：

```bash
uv run dressing route-a --simulator genesis --genesis-device cpu --open-browser
uv run dressing route-a --simulator genesis --smoke-test
```

Genesis 使用相邻 `../genesis-local/` 已安装的独立 `.venv`，不需要把 Genesis 和 PyTorch
装进本项目环境。路径、设备、子步和粒子直径在唯一的 `config.toml` 的 `[scene.genesis]` 中设置。
MuJoCo 仍是未指定 `--simulator` 时的默认后端；`--genesis-device` 只用于 Genesis。
原 `genesis-local/run-dressing.sh` 保留场景／脚本抬升演示，GPT 实验使用这里的统一入口。

Genesis 末端位置采用原 Panda 手指尖抓取点，支持世界坐标的小幅平移、旋转和夹爪开合。
动作经过同样的平移／旋转范数、工作空间、IK 收敛和关节增量检查，
通过关节位置控制和物理步进执行；IK 规划不会重置机器人或物体位置。
当前求解器子步为 1 ms，Genesis 每个控制步包含 10 个子步，耗时 0.01 s 仿真时间。
每个 1 s 动作分 100 次更新关节目标，保留平滑插值，包含 1000 个求解器子步。
五个动作顺序执行完毕后取得新图像，再请求 GPT 决策；中间图像仍更新网页与录像。
Genesis GPT worker 的双臂开启重力补偿，避免每个长动作开始重新规划时累计静态下沉。
等待模型回复时仿真暂停，两轮决策的现实时间间隔取决于模型延迟和仿真速度。
GPT 仅收到俯视／腕部／桌面侧视 RGB、相机标定、机械臂末端状态、指令、上一动作／图像和各 chunk 的执行反馈。
物体节点、材质和接触诊断留在 `runs/<run-id>/private/genesis/`，不挂载给模型进程。

Genesis 1.4.3 原生 legacy rigid/PBD 分支没有实现摩擦；当前场景已添加版本检查的场景级
静／动摩擦适配器，每个 1 ms 子步对实际接触的自由节点施加以法向投影量为限的切向修正，
考虑刚体相对运动并保留反作用力。桌面对树懒的系数为 0.7、对衣服为 0.6，
不会把物体固定在桌面，也不会逐步重置粒子位置／速度。安装的 Genesis 源码未修改。
闭爪并且两个指垫实际接近同一个软体的表面节点时，仍会激活局部 PBD 抓取约束，
每个手指最多附着四个接触附近节点；张爪解除约束。该规则同时适用于树懒和衣服，
不为 GPT 选择抓取目标或生成动作。这是无滑动抓取近似，不是纯摩擦夹持或负载验证。
材料、接触和自碰撞的标定影响穿衣效果与真实性，但不阻挡 GPT 闭环运行。

2026-10-08 CPU 静置对照：保持双臂不动，先静置 2 s 再观察 5 s；
树懒中心的水平漂移从约 2.03 mm 降至 < 0.001 mm，衣服从约 1.96 mm 降至约 0.007 mm。
衣服局部仍有约 0.68 mm 的最大节点位移，不能视为所有节点完全静止。
报告保存中心位移和 RMS／最大粒子速度，无桌面穿透检查不再被描述为静止验证。
详见 [Genesis 静置记录与接触限制](../genesis-local/README_CN.md)。
修改适配器后需要重启 Genesis 场景／路线 A；已经运行的子进程保留原来的代码。
修复后的实际 CPU Genesis 子进程也完成 2 s 静置和五个各 1.000 s 的有界动作，
三视角均正常出图，无非有限状态或桌面穿透；见
[流程检查报告](runs/20261008-genesis-friction-bridge/report.json)。
该检查没有调用 GPT，也不判定穿衣任务成功；项目 29 项测试和 Genesis 5 项接触／控制测试通过。

Genesis 子进程的日志、场景报告和抓取事件位于 `private/genesis/`；
网页、五个 MP4、`session.json` 和逐步决策记录仍使用路线 A 原有的目录结构。
接口通过、真实模型调用、控制运行和人工穿衣结果分别记录，不以模型文字声明成功。
IK 使用 Genesis 的[官方世界坐标／局部抓取点接口](https://genesis-world.readthedocs.io/en/latest/_modules/genesis/engine/entities/rigid_entity/rigid_entity.html)。

2026-10-07 本地接入检查：17 项项目测试和 4 项 Genesis 接触／释放测试通过；
真实 CPU Genesis 执行左手小幅往返、旋转 IK 规划、越界拒绝和三份录像输出。
网页响应逻辑使用真实相机图像检查通过，但本次工具环境禁止创建 TCP 套接字，未验证实际网页监听。
独立 Codex 真实调用已尝试，服务连接失败（`Connection failed: error sending request`），
因此真实 GPT 回复、CUDA 执行和实际网页需要在本机正常联网的终端继续验证，不能将离线夹具检查当作真实 GPT 通过。
检查记录：[validation.json](runs/20261007-genesis-route-validation/transport-check/validation.json)。

历史 1 ms 子步与平滑控制检查（当时动作时长为 0.1 s）：19 项项目测试、4 项 Genesis 接触／释放测试通过。
真实 CPU Genesis 完成 4.2 s 仿真（2 s 静置、两次各 0.1 s 往返、2 s 保持），
状态有限、无桌面穿透、源资产未变，三份录像保存通过；模型回复使用离线夹具。
记录：[1 ms 验证](runs/20261007-183927-genesis-1ms-check/validation.json)。

五个 1 s chunk 的新流程检查：24 项项目测试、4 项 Genesis 接触／释放测试通过。
真实 CPU Genesis 完成 2 s 静置和完整的 5 s 动作批次，各 chunk 实际执行 1.000 s，
下一轮收到完整执行反馈，三份录像保存通过，状态有限且无桌面穿透。
此检查使用 320×240、2 fps 降低 CPU 渲染开销，模型回复使用离线夹具；
[批次验证](runs/20261007-185024-genesis-five-chunks/validation.json)记录的是启用机器人重力补偿之前的流程。
随后补偿后的独立检查通过：1 s 水平动作的垂直偏移约 0.14 mm，1 s 静止保持的末端漂移小于 0.002 mm；
见[重力补偿验证](runs/20261007-185800-genesis-gravity-hold/validation.json)。真实 GPT 回复和 CUDA 尚需本机验证。

桌面侧视相机检查：29 项项目测试通过，包括两／三视角策略输入、水平相机方向、侧视网页响应和五个录像输出。
真实 CPU Genesis 渲染三个 640×480 视角并保存五个 MP4，通过相机高度与水平视线检查；
这次为相机／渲染检查，没有重复物理基准或调用真实 GPT。
见[侧视预览](runs/20261007-205343-table-side-camera/preview-table_side.png)和[相机验证](runs/20261007-205343-table-side-camera/validation.json)。

## GPU 场景与窗口

```bash
uv run dressing scene --view
uv run dressing scene --view --play
uv run dressing scene
uv run dressing scene --build-only
```

第一条在 GPU 沉降验证后打开交互窗口，初始暂停；空格运行/暂停，鼠标旋转、平移和缩放。
加 `--play` 会在验证后从初始状态重放并自动运行，能看到软体落到桌面。
不加 `--view` 时在 NVIDIA EGL 上离屏渲染并验证；`--build-only` 只编译场景，不执行物理，也不证明 GPU 可用。
默认运行 2 秒仿真，可用 `--seconds` 调整。第一次 CUDA 内核编译较慢。
配置固定 `backend="warp"`、`device="cuda:0"`，不可用时直接报错，不静默回退 CPU。
这里直接使用 MuJoCo Warp 物理后端，尚未引入 mjlab 的训练环境/manager 层。

每次运行将生成场景 XML、配置快照、初始/最终状态、五视角 PNG、沉降动画 `settling.webp`、
数值轨迹和 `report.json`，全部位于 `runs/<run-id>/`。CUDA 编译缓存复用
`runs/gpu-runtime/cache/`。报告分别记录物理 GPU 和 OpenGL 渲染设备，以及警告、速度、
桌面穿透检查和实测仿真/墙钟时间比例。单场景 GPU 不保证比 CPU 更快。

相机包括全景 `overview`、俯视 `overhead`、桌面近景 `detail`、左手腕 `wrist` 和水平侧视 `table_side`。
场景状态和调试视角不自动进入 A/B 观测；实时路线 A 默认接收 overhead/wrist/table_side RGB 和机器人状态。

软体参数集中在 `config.toml`，均为未经实物标定的仿真先验：

| 对象 | 模型 | 质量 | 杨氏模量 | 泊松比 | Rayleigh 阻尼 |
| --- | --- | --- | --- | --- | --- |
| 树懒 | 551 顶点、2549 四面体体积弹性 | 180 g | 8000 Pa | 0.3 | 0.001 s |
| T 恤 | 约 1200 三角面，拉伸与弯曲壳弹性 | 55 g | 30000 Pa | 0.3 | 0.001 s |

T 恤物理厚度 1 mm、宽 30 cm；网格来自原始 Green T-shirt，简化时检查连通性和四个边界开口。
没有用 `cloth3` 布片替代衣服。MuJoCo 3.15 的软体使用 `discrete` 积分器，当前共享求解器时间步为 1 ms。
当前关闭 flex 自碰撞，因此衣服前后层仍可能相互穿透；沉降检查不能代替穿袖与抓取验证。
桌面和机械臂接触使用各软体顶点上的小球碰撞体，这不是连续表面接触；树懒与衣服之间
保留 flex 面接触。该近似适合当前场景支撑检查，不足以证明穿袖时不会穿透。
2026-10-06 的 RTX 4070 SUPER 历史实测（当时子步为 0.5 ms）：2 秒 GPU 仿真（含 50 帧渲染）约 18 秒墙钟时间，
最低顶点高度 0.7393 m（桌面 0.74 m）、最终最大广义速度约 0.00090，
无 MuJoCo 数值警告或 GPU 求解/缓冲区溢出。当前高自由度场景尚未达到实时速度。
树懒导入目录的 README 引用了本项目没有的旧转换脚本和完整源快照，当前只使用已提供的
NPZ/PNG；可用来源及内容哈希记录在 `assets/manifest.json` 和运行报告中。

## 环境与检查

```bash
uv sync --locked
uv run dressing doctor
uv run pytest --basetemp=runs/pytest
```

Python 3.12；依赖由 `uv.lock` 固定，`.venv` 独立于旧项目。
另包含 MuJoCo Warp、Warp 和网格简化依赖，使用当前工程 `.venv`，不导入旧项目代码或环境。

路线 A 还需要 bubblewrap 的用户命名空间权限。`doctor` 会实际检查隔离环境能否启动，
`route-a` 会在 GPU 初始化和打开页面前检查，不调用模型。Ubuntu 24.04 默认的 AppArmor
用户命名空间限制可能导致 `bwrap: setting up uid map: Permission denied`。遇到此错误时，
在项目目录安装并加载随项目提供的配置（需要管理员权限）：

```bash
sudo install -m 0644 system/apparmor.d/dressing-bwrap /etc/apparmor.d/dressing-bwrap
sudo apparmor_parser -r /etc/apparmor.d/dressing-bwrap
uv run dressing doctor
uv run dressing route-a --smoke-test
uv run dressing route-a --open-browser
```

该配置仅允许系统 `/usr/bin/bwrap` 创建用户命名空间，全系统的 AppArmor 限制保持启用。
模型进程仍使用文件系统隔离；每次实验使用独立会话，同一次实验内继续对话。已打开的旧实验需要退出后重新启动。
这份配置只适用于 `/usr/bin/bwrap`；其他安装路径需由管理员调整对应路径。
参见 [Ubuntu 24.04 官方说明](https://documentation.ubuntu.com/release-notes/24.04/#unprivileged-user-namespace-restrictions)。

## 目录约定

```text
config.toml                唯一实验配置
src/dressing_lab/          实现；一个 dressing 命令入口
assets/robot/              Panda XML、所需网格、许可证
assets/garment/            原始 T 恤网格与许可证说明
assets/bunny/              用户提供的 deformgen 树懒/布片；不是兔子资产
assets/manifest.json       资产来源、状态和内容哈希
tests/                    本地测试
runs/<run-id>/             所有生成结果；按需创建
```

整理要求写在 `AGENTS.md`，后续修改应扩展现有模块，避免复制入口和产生零散文件。
当前衣服尺寸为树懒场景配置，未来兔子任务需要独立确认尺度和接触效果。

## API 设置

仅在本地 shell 设置 `OPENAI_API_KEY` 和 `OPENAI_MODEL`。
也可在本机创建被忽略的 `.env`，再通过 `uv run --env-file .env ...` 加载。
不要将密钥发送到聊天、写进代码或提交到版本控制。模型必须支持图像输入和
结构化输出；项目不猜测你的账户可用模型。

API 使用官方 Responses 的图像输入与 Pydantic 结构化解析：
https://developers.openai.com/api/docs/guides/images-vision
https://developers.openai.com/api/docs/guides/structured-outputs

## A/B 决策接口

两条路线只接收俯视／腕部 RGB、可选的 table_side RGB、机器人末端位置/夹爪开合、可选上次动作及执行前图片。
严格拒绝额外的物体真值字段。下面是单次 API 接口；上面的实时路线 A 另外提供姿态、相机标定与有界旋转，并已接通 IK。
图片路径相对于 observation.json；图片内容会发送至 OpenAI API。

观测格式（这是格式说明，不是实验数据）：

```json
{
  "rgb": {"overhead": "overhead.png", "wrist": "wrist.png"},
  "robot": {
    "left_position_m": [0.3, 0.0, 0.5],
    "right_position_m": [0.3, 0.5, 0.5],
    "left_gripper": 1.0,
    "right_gripper": 1.0
  },
  "previous_action": null,
  "previous_rgb": null
}
```

有真实观测后可调用一次 API（会产生 API 费用）：

```bash
uv run dressing decide --route a --observation runs/session/observation.json
uv run dressing decide --route b --observation runs/session/observation.json
```

- A 输出两只夹爪的短程平移与开合，每个平移向量默认不超过 2 cm；拒绝非法动作。
- B 输出物性区间假设与末端目标，**不执行、不修改物理场景、不假装已经完成 MPPI**。
  GPT 提出的杨氏模量/摩擦只是尚未验证的先验，需在选定材料模型下用交互数据验证。
- API 拒绝、不完整响应、非有限数值会终止决策；不会用静默回退动作伪装成功。
- 一次调用的决策、用量和延迟只写到 `runs/<run-id>/decision.json`；记录
  `executed=false`、`task_success=null`，不把接口通过计为任务成功。

## 等兔子资产到位后的工作

1. 确认尺度、姿态、单位、表面/体网格、材质与用途；选择软体建模方法。
2. 构建机器人＋兔子＋衣服的场景，验证相机、控制、接触、速度与稳定性。
3. 在兔子资产上验证 A 的真实闭环，保留有限动作次数与耗时记录。
4. 加入独立的预测模型与规划器；模型只根据允许的观测重建状态，禁止复制环境真值。
5. 比较 A、固定模型＋规划器 B0、更新模型＋同一规划器 B1；定义袖口任务成功指标。

本地测试使用合成图片和 mock API，只验证传输/数据边界，不代表真实 GPT 调用，
不代表模型辨识或机器人任务成功。
