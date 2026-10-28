# 原 task0 的原生运行时路径快照（CPU，执行前登记）

原总目标及18正确参考/12WAM开发/24最终留出门槛保持，GPU禁用。当前固定优化余量搜索4000响应已正常关闭但完整H12 witness为0；完整独立checker仍在原session28112运行。本轮仅准备后续原task0/actual368固定命令物理验证所需的文件路径，不新增搜索候选、模型加载、数值响应、环境构建或物理周期。

## 输入与身份边界

原manifest为`outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/manifest.json`，SHA`4a42c9647372bfce84f63bde93bb066d0ee06d7157cce9d8cef99271ee3539ab`。原runtime_config、79维初态及其SHA保持；历史声明中的`/share/project`只作provenance，不读取、写回或恢复。

安全只读根固定为`/share-mixed/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim`。原config JSON形式的yaml SHA`d7ece7fb95f0fa9462b60b97632e2b7ece297f074d9c66433c0002018e9a114e`，OSMesa SHA`40e28f0f15c4cd13ad3c353509eb2fad79d8041528c59dc56d6e68fae8d1677a`。完整LIBERO运行包中的45份历史Python源码必须逐一匹配原manifest；正式venv中的120份历史robosuite源码亦按原引用只读核验。

本任务bowl/plate BDDL、默认OSC JSON、未登记LIBERO文件及18个系统动态库只有当前身份。新快照不得把当前SHA追认为原物理运行时的历史身份，不宣称二进制或行为等价。具体当前指纹由`configs/native_runtime_path_snapshot_20261029.json`固定。默认OSC JSON仍由原venv的controller_factory定位；系统库实际装载次序、符号解析和额外dlopen依赖尚未检查。

## 有限文件迁移

新增纯stdlib模块及CLI；不导入LIBERO、torch、MuJoCo、OpenGL或ctypes，不执行setup.py、ldd、动态加载器、外部脚本或模型。命令cwd固定本仓库，空CUDA/LD，单线程，所有新增文件在仓库outputs内。

新独占目标为`outputs/native_runtime_path_snapshot_20261029`。复制安全根的`LIBERO/libero`完整非缓存文件树及LICENSE，保留双层namespace布局；排除`__pycache__`、`.pyc/.pyo`，不复制.git、训练数据集或外部路径。保留包中已有init_files作为文件快照，不将其计为固定manifest初态的实际输入。另复制单个OSMesa文件。最多10000个源文件、1GiB总源字节，执行前预检、流式复制、每个源和目标SHA一致、结束时重新枚举及核验；拒绝symlink、路径穿越、特殊文件和目标覆盖。发生中途失败保留独占目录，不自动重试。

保存原manifest及原config字节副本；原runtime对象只替换`libero_root/libero_config/osmesa_library`三项。config内`benchmark_root/bddl_files/init_states/assets/datasets`五键精确映射到快照根，其他参数不改。`datasets`只映射至明确未填充的空目录，不读取原data目录，也不宣称数据已复制。独立文件核验须反向比较原对象并检查精确inventory，防止重算manifest SHA掩盖非路径参数变化、文件遗漏或附加文件。

OSMesa传递系统库及默认OSC JSON在本轮仅登记只读当前引用，不修改LD_LIBRARY_PATH/RPATH、不预加载，也不宣称已激活这些副本。复制文件不等于实际导入来自快照。完整资产树中非本任务XML的历史绝对路径不改；本快照不授予其他任务或整个LIBERO的资产资格。

## 验证与后续门槛

合成文件测试验证精确3+5路径修改、原参数/初态字节保持、历史源码SHA、源/副本变化及重hash篡改拒绝、全量inventory、预算、路径安全、独占失败和不读取训练dataset；无仿真或GPU。CLI检查全部输入/当前依赖、source-bound QA、实际dry及执行后文件核验；正式原生Python运行，实际wait/reap与日志保存，单次文件准备总限900秒。完成可验证阶段及时本地提交。

产物只能标记`file_snapshot_verified=True`，且`runtime_loaded=False/bootstrap_verified=False/control_qualified=False`；不能据文件迁移申请物理资格。实际加载及bootstrap的单独资格方案尚未执行。只有原H12完整模型witness、全部独立响应核验和owner正常关闭均成功，才可另行登记实际物理pilot；本轮没有该witness，不运行物理。
