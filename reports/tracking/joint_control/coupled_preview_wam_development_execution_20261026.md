# WAM预览开发：实际准备与启动

正确参考18/4907控制资格通过后，新12案例开发已完成实际输入/源码/CLI准入并启动；目前还没有新完整WAM任务结果或fresh成绩。

## 已执行的准入

- 原生Python3.11完整12案例输入preflight通过，0模型/物理/array decode，记录`outputs/coupled_preview_wam_preparation_20261026/input_preflight_py311/result.json`。
- 新候选相关77项CPU测试通过（15.65秒），实际QA记录与命令存档。
- 首次prepare使用`.venv`因缺少h5py，在创建新namespace前退出1；失败日志`outputs/commit_preview_20261026/wam_prepare.log`保留。随后显式使用已有原生3.11解释器执行同一脚本，未安装或改动共享环境，未重做已通过QA/预检。
- 新prepared与1572份源码快照完整生成，独立metadata实际退出0。主目录`outputs/current_coupled_wam_development_20261026`，prepared SHA `beb26bb9cb6e7822039ab56314a33bb14487bd7ae29541748e5d333cded106e5`。
- 新registration生成，owner、模型解释器的service dry及12个原生采集dry共14个真实CLI全部return0，0模型初始化/物理周期。registration SHA `60bbb0b892ad8442067a3d0783cc4e2ae1a2586f2a8ecf32e0e9dcaf2c6bd8ee`，详见`successful_cli_dry.json`。

## 实际启动与监督

启动前通过驱动查询允许UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`：可见GPU2、设备minor6、显存0、无compute进程、利用率0。只使用该UUID。

外层durable owned_stage直接持有新owner进程，使用同一冻结source roster和`supervisor/launch.json`完成owner的实际birth/ready握手，最终负责真实wait/reap及异常清理记录。owner内部再负责模型服务与12个采集器的真实退出。没有重复启动或自动重试。

启动快照经/proc核实：owner PID1127722/start_ticks595284179，模型服务PID1127755/start_ticks595285080，均同身份存活。模型服务实际CUDA_VISIBLE_DEVICES为允许UUID、LD为空、MASTER_PORT=31366。快照时`service_readiness.json`尚未产生，不能将进程存在称为模型初始化或请求成功。

源文件从准备后冻结，保持固定12案例、raw/terminal058统一预览控制器、900请求/31500节点/7200周期及原开发门槛。只有完整审查后terminal稳定≥5/6且各任务≥1/2才允许准备最终fresh；当前不消费留出新初态。总目标未完成。

## 服务ready与首个真实案例（后续快照）

`development/orchestration/service_readiness.json`已产生，真实模型初始化1次，ready时请求/节点计数均0。允许UUID上驱动观察到35591MiB显存；`nvidia-smi`显示的是主机PID，不以它代替容器/proc身份链。

首个raw案例`task0_init0_s42_raw`的采集worker PID1128886/start_ticks595294569已经启动，与owner1127722及服务1127755均经/proc核实同身份存活。最近progress为60/600物理步、8个chunk、尚未ever成功；这是前缀状态，不是完整任务结果或失败判定。

对前3个真实当前图像请求做独立查询检查实际退出0，含120个生成端点、当前PNG像素SHA、请求/响应case/seed/分支和checkpoint归一化反解。证据`outputs/coupled_preview_wam_live_first3_20261026/verified.json`绑定捕获时的不可变前三chunk副本及完整原请求/响应/图像SHA；运行中的chunks.json继续追加，不将快照伪称最终完整文件。审查未发起额外模型请求或物理步。

该前缀仅验证查询链路，全量独立机械/31500节点及600步任务成绩仍须等待正式case和开发结束。未变更冻结源码或运行配置，无自动重试，fresh仍未消费。
