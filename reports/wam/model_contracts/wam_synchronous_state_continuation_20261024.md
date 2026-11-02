# 055：同步状态采样有限接续

2026-10-24 用户明确要求恢复任务后，仓库小文件创建、flush/fsync及删除检查通过；共享挂载 df 显示约155 TB可用。此前ENOSPC的具体原因未确认，不声称已定位根因。

055已有prepared及独立metadata保留。注册时重新核验全部source、snapshot及input SHA，通过后生成registration.json；实际模型Python执行CLI dry成功，0模型初始化、0请求。仓库及/opt Python缺少h5py，准备入口在导入阶段退出，未生成注册或启动模型；改用已有模型Python完成注册，未安装或修改外部环境。

产物根目录：`outputs/synchronous_state_probe052_continuation_20261024/`。

- prepared.json SHA256：`8d9ae112f3acfd2fa006cce4c8a387c723cdcd880eccfaec0cbd45d6d58a7ce4`。
- metadata_verified.json SHA256：`53666cc93e2f0215bae388cf61541513d93492dd04cc019d7705ca04b0166271`。
- registration.json SHA256：`b54ed60369ca0c61136448b34f8d63044af1b41987bdb4082d073a66bdf16947`。
- supervisor/launcher_result.json SHA256：`a3fc9dd393310e824acd8d4e528a6733245ac7908d22376a226743cb873af606`。

持久化owner PID2808200/start_ticks579017061已ready；模型子进程PID2808219/start_ticks579017262，PPID2808200。启动后实际ps观察二者存活，模型处于初始化阶段，尚无完整采样或性能结论。只使用允许UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`；启动前和owner内检查空闲及设备身份，持有UUID排他锁。进度与终态以实际PID、command.log及owner收据为准，不因观察超时重新启动。

范围固定原indices4..98：95请求/3325节点，额外最多一次初始化；054前4请求/140节点及原失败完整保留，累计初始化2次。新数值审查通过且正常wait/reap/GPU释放后才执行CPU评分。没有新的fit、物理任务或fresh；最终24条稳定≥20/24、每任务≥4/8门槛及总目标不变，当前不能声明目标达成。

后续实际观察：同一owner/model PID仍存活，日志已完成模型加载并进入首probe052的UniPC35步采样，观察到32/35步；已越过054的采样前分派错误点。此为运行进度证据，尚不代表完整95请求或数值/性能资格通过。

进一步观察已闭合原index4..9共6请求/210节点；同一两个PID仍存活。额外CPU只读调用`collect_wam_clean_training.same_initial`，将054实际raw index3的initial_ref与055 index4的initial.json比较，完整初始数组/条件逐位配对检查通过。此检查未打开053未来状态标签，不能代替最终全节点数值审查。

## 完整终态：数值通过，性能门槛失败

全部95请求/3325节点完成，worker complete/passed=True；联合prefix后为99请求/3465节点，累计2次初始化。独立审查完整通过，48对initial核验通过、prefix105 zero节点保留、source/input未变，审查本身0未来档案读取。模型实际returncode0、reaped=True、signal_sent=False，GPU memory0/compute_apps空；数值审查及CPU评分均exit0。owner complete/passed=True；最终owner及模型PID均已消失。日志有NCCL未显式destroy警告，实际正常退出和GPU释放证据保留。

| 产物 | SHA256 |
| --- | --- |
| state_sampling/result.json | 2387db2d7131eb6349603e1abffe193e0acfd3bc580e4b1c2b983f40c1236d65 |
| numerical_verified.json | 57de38b05852976e391263adb0d921f7effecf86f4c4fa43cafff4b9648a97ba |
| state_score.json | 3cbbc580cdcdd94642ac767ac907d44c2e797cf6bea9ef05c6c5ddc54f80985b |

数值/生命周期通过后CPU评分才读取六条053未来轨迹。原固定state gate=False、eligible_for_development=False：短期联合改善、分量保护、至少5个episode、短期outlier及双seed保护均失败；仅严格胜过hold/constant_velocity检查通过，episode通过2/6。不得将评分脚本passed=True解读为性能门槛通过。

| 时域 | 位置RMSE raw→probe052 (m) | 旋转RMSE (rad) | 开度MAE |
| --- | --- | --- | --- |
| 8 | 0.022961→0.027137 | 0.098577→0.078373 | 0.063151→0.115072 |
| 16 | 0.037509→0.047629 | 0.179833→0.139385 | 0.046795→0.123004 |
| 40 | 0.067346→0.082709 | 0.457855→0.320060 | 0.097898→0.150933 |

结论：052训练误差改善没有转化为合格同步参考；旋转改善伴随位置及夹爪退化，不能推进该候选到开发任务或fresh。下一步应以现有训练来源和本次validation失败做分量退化诊断，再独立登记新候选；不根据最终留出初态调参，不改现有门槛。当前0新增物理/fresh，总目标未达成。
