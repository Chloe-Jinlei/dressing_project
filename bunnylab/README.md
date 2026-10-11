# Dressing Lab

穿衣路线 A/B 的独立 uv 开发项目。只有一个 CLI 和一个配置文件。
当前用户明确指定的场景是：双 Franka Panda、桌面、DeformGen/PhysTwin 树懒软体和带袖 T 恤。
兔子任务仍未验证；树懒场景不会记作兔子穿衣成功。路线 A 已接通视觉决策与末端控制；路线 B 新增 GPT 局部目标 + 运动学 MPPI 控制，软体动力学预测尚未实现。

## GPT + MPPI 对照实验（路线 B）

两组实验共用同一个网页、软体场景、三相机、动作安全检查、录像和人工判定。
在 `bunnylab/` 中启动新模式：

```bash
uv run dressing route-a --simulator genesis --controller gpt-mppi --open-browser
```

原实验仍使用 `uv run dressing route-a --simulator genesis --open-browser`。
保留统一 CLI 的 `route-a` 网页入口，通过 `--controller` 选择控制器；
`direct` 是路线 A（默认），`gpt-mppi` 是路线 B。A 默认端口 8765，B 默认端口 8766。
建议正式对照实验依次运行，避免同时使用 GPU 影响耗时比较。

B 的闭环为：固定用户目标 + 三视角 RGB → GPT 的结构化局部目标 → MPPI →
已有 IK / 关节限幅控制器 → 真实软体仿真 → 新观测。
GPT 输出双臂的绝对世界坐标、单位四元数（wxyz）及夹爪开度，不能输出代码或物理模型。
每轮目标距当前末端最多 10 cm / 0.5 rad，并须在工作空间内。
MPPI 每次采样 512 条、预测 8 步、迭代 3 次，使用指数代价加权和采样修正，
只执行优化序列的首个动作。每个动作仍为 1 秒，最多 2 cm / 0.1 rad；
读取实际机器人末端状态后重新优化下一个动作。五个动作后再发送三视角图像给 GPT。
夹爪直接采用 GPT 目标开度，不进行夹爪时序优化；接近、闭合和搬运应分轮规划。
停止、超时或 IK 拒绝会中止剩余动作。完成声明仍需人工判断。

**预测模型边界：** 当前为 `cartesian_kinematic_v1`，只用批准的机器人观测做
局部笛卡尔积分和旋转切空间近似。代价包含目标误差、终端误差、动作幅度和动作变化。
它不预测衣服/树懒变形、接触或碰撞，不读取粒子、网格、物体真值和真实材料参数，
也不会把 GPT 的材料猜测当作已验证模型。实际执行场景仍使用原有软体求解器。
因此这是 **GPT 局部目标 + 运动学 MPPI 基线**，不是软体动力学 MPPI，不能据此声称
已实现变形预测或验证穿衣成功。算法参考 [Williams 等人的 MPPI 论文](https://arxiv.org/abs/1707.02342)。

参数在同一份 `config.toml` 的 `[mppi]` 中；模型、画面、动作时长和任务预算仍与
`[route_a]` / `[control]` 共用，以便对照。`seed` 控制 MPPI 采样；接近目标时降低探索噪声。
每场重置重新创建仿真、GPT 会话和规划器。`session.json` 保存控制模式、预测模型、
完整 MPPI 参数与局限，`steps/<轮>/decision.json` 保存 GPT 目标，
`steps/<轮>/chunks/<动作>/mppi.json` 保存实测状态、目标、采样诊断和实际生成的动作。
视频和所有输出仍在 `runs/<run-id>/`，目录名区分 `direct` / `gpt-mppi`。

可用以下命令检查真实模型连接和 MPPI 小幅往返运动（不会运行完整穿衣任务）：

```bash
uv run dressing route-a --simulator genesis --controller gpt-mppi --smoke-test
```

这个检查的 GPT 仅被要求保持不动；往返目标由测试脚本提供。完整 GPT 目标到 MPPI
任务的有效性仍须通过实际实验评价。原 `decide --route b` 的离线物性提案接口保持原含义，
不等于此处的实时控制模式。

2026-10-11 验证：77 项单元与回归测试通过，覆盖原模式、目标格式与观测边界、
限幅、收敛、停止、IK 拒绝反馈和录像。真实 GPT + CPU Genesis 的接口检查完成两次
小幅运动并保存五份视频；该物理检查在最后的采样初值优化前完成。最终控制器另外通过
真实三视角 → GPT 目标 → MPPI 动作的只规划检查。GPU 检查因 PyTorch 无法访问 NVIDIA
设备而未通过，完整自主穿衣未验证。详见本机 [验证报告](runs/mppi-development/validation.json)。

## 路线 A 自主任务网页

```bash
uv run dressing route-a --simulator genesis --open-browser
```

输入一个完整目标（例如“帮助树懒将衣服穿好”），点击“开始自主任务”。一场实验只接受一个固定目标。
GPT 在同一独立 Codex 会话里持续观察三视角图像、自主规划、执行与修正；不需要逐阶段追加指令。
它只收到批准的 RGB、相机标定、机器人末端状态、上一动作/图像、执行反馈及有限枚举的控制器拒绝原因。
不提供物体真实几何、接触状态或材质，不将原始异常或私有路径发送给模型。

每轮 `act` 仍输出五个动作，各执行 1 秒仿真；原平移、旋转、IK、关节增量与工作空间限制保持。
`wait` 表示最多推进 1 秒仿真后自动重新观察，不再等待用户输入。
`completed` 是模型对整个目标的完成声明；`unable` 表示无法继续。两者结束操作并保存视频。
模型同时输出阶段、观察到的进展与说明，在网页显示。模型声明不能自动标记任务成功。

动作验证失败会取消剩余 chunk，把有限的拒绝原因反馈给同一个会话自动重试。
调用失败或超时也有限次重试，不悄悄替换会话。等待模型回复时物理暂停。
默认预算集中在 `config.toml` 的 `[route_a]`：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `max_steps` | 600 | 最多执行的动作数，短暂等待也占一个名额；完整动作批次需剩余至少 5 个名额 |
| `max_wall_seconds` | 1800 | 从提交目标开始，最多 30 分钟现实时间 |
| `decision_timeout_seconds` | 180 | 单次模型调用超时 |
| `max_consecutive_failures` | 5 | 连续调用或控制拒绝次数上限 |
| `max_no_progress_rounds` | 8 | 连续没有明显观测变化的轮数上限 |
| `wait_seconds` | 1.0 | 每次 `wait` 推进的仿真时间 |

无变化检查仅比较批准的机器人状态和 RGB 变化，是停滞启发式，不是穿衣进展或成功评估器。
现实时间上限会在模型等待和动作控制更新时检查；正在进行的单次仿真/渲染调用及视频保存可能额外耗时。
Esc 或“停止并保存视频”随时结束。结束后网页保留最终画面、模型声明和视频下载，点击人工“成功/失败”记录最终判定。
也可以执行中人工判定并停止。`session.json` 分开记录 `model_claim`、`automatic_success=null` 与人工 `task_success`。
本版本没有添加自动几何成功评估，也不预设脚本穿衣动作。

“新实验：全部归位”在保存完成后销毁旧仿真、重新加载原始场景，建立新的输出目录和空白 GPT 会话。
页面自动重连，旧实验视频、日志、判定保留在原目录。刷新或关闭浏览器标签不会中断后台任务或重置场景；
重新打开同一服务 URL 仍是当前实验。要退出后台服务，请点击“保存并退出服务”或在终端 Ctrl-C。
重新运行启动命令总会从初始场景与新会话开始。

模型使用本机已登录的独立 Codex CLI（配置 `[route_a].model`），需要 `bwrap`、`ffmpeg`；
不使用单次 API 接口的 `OPENAI_API_KEY`。隔离进程不挂载工程、资产、真实物性或其他会话，并禁用工具。
`--smoke-test` 仍是独立的真实模型保持不动/会话记忆检查加脚本往返检查，不是完整任务试验。
省略 `--simulator genesis` 时仍使用 MuJoCo，物理场景与相机未因自主任务改动。

每次实验输出到 `runs/<run-id>/`：五个 MP4、`session.json`、人工 `task_results.json`、
模型结束声明 `model_result.json`、`policy/` 会话记录，以及 `steps/` 内逐轮观测、动作、执行与预算记录。
视频按墙钟时间录制，含模型思考等待，不能当作实时仿真速度。

本版本机检查（2026-10-10）：单元与流程测试覆盖自主重试、短暂等待、预算、操作中停止、人工判定和新实验。
另用离线模型回复连接真实 CUDA Genesis 与 HTTP 网页跑完两场：各五个动作、一次等待和结束声明，
各自保存五份录像；新实验目录和测试会话不同，重建后的俯视图与初始图一致（平均像素差约 0.008/255）。
记录见 `runs/autonomous-development/integration.json`。该检查未调用真实 GPT，未验证穿衣成功；
当前工具没有可连接的浏览器，网页验证覆盖 HTTP 接口和 JavaScript 语法，未做浏览器内视觉验收。

2026-10-10 输出格式修复：新增的 `stage` / `progress` 有本地默认值，原先直接导出 Pydantic schema，
导致真实服务拒绝 `invalid_json_schema`。现在发送给 Codex 的 schema 将所有对象字段列为必填并移除默认值，
本地验证和动作边界保持。已补充严格 schema 回归及不可重试错误检查；这类配置错误立即停止并在网页显示原因，
不会再连续调用五次。真实模型保持不动、同会话记忆和 CUDA 脚本往返检查通过：
`runs/20261010-192652-route-a-genesis-78c548/session.json`。此外真实 GPT 生成的五个左手 1 mm 小动作均通过验证并在 CUDA 执行，记录见 `runs/schema-fix/real-actions-192823/session.json`；48 项测试通过。这仍不是完整穿衣成功验证。

### 在 Genesis 中运行同一条路线 A

材料参数标定不是接入 GPT 的前提。可以先用当前默认参数完成
“三视角观察 → 同一实验对话决策 → 有界动作验证 → Genesis 执行 → 再观察”的闭环。
模型、Codex 登录、单目标自主任务、网页和录像逻辑与 MuJoCo 路线 A 共用。

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
Genesis 棉布参数位于 `[scene.genesis.garment_material]`：采用文献实测 211 g/m² 和
5.56036655e-6 N·m 弯曲刚度，当前衣服质量按面积计算为约 20.50 g。
树懒保持 180 g，使用可变形四面体及修正后的体积约束。
棉布拉伸、摩擦与树懒填充物仍为近似，详见
[数据来源、离散方法和形变验证](../genesis-local/README_CN.md#几何与物理对应)。
下文 MuJoCo 的 55 g、杨氏模量和厚度不是 Genesis 当前布料参数。修改后须重启服务。
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
