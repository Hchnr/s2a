# 240 全记录矩阵进程级环境准入修正

239首条164周期原生JSON/ODE/KKT/native-goal/history准入及actual wait0/reap完成。扩展18case时，case00再次完整通过，但同一进程第二次调用OpenCV“导入前准入”被正确拒绝；原failure.json、源码、case00证明、actual wait1/reap保持。未在case01开始六维求解，无新物理。

checked入口只把严格CV2准入移到进程main一次，在任何MuJoCo/robosuite/OpenCV导入之前执行；各case仍调用strict_native_environment并核对CONTROL SHA，分别加载原XML/raw states、重置两套重建状态。数学模型、优化、native decoder、独立ODE/数值梯度与KKT、6项损坏拒绝、4907周期和原控制门槛均不变。不存在安装或修改外部环境、跳过guard、自动重试或覆盖原产物。

新namespace `outputs/full_coupled_all_records_checked_20261026`，新持久监督启动前必须核实旧worker实际wait1/reap和进程消失。新完整18/4907仍只属记录状态与重建控制器cache证据；无反事实物理控制、模型或fresh资格。
