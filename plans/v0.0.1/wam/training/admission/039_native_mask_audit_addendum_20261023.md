# 039：037原生无效槽mask的只读审计补充

在任何修正检查器执行前登记。本项仅消费037已完整保存的7请求/245节点，不重新初始化模型、采样、训练或执行物理。037模型worker已actual wait0/reap、无信号退出，GPU已记录0MiB释放；原owner因独立checker拒绝而False，原结果、源码、日志和SHA全部保留。

## 已证实的检查器假设错误

037已登记新Cosmos原生action mask修订。冻结`omni_mot_model.py:2747–2755`先将无效action槽的condition reference清零，再将inactive118的flat condition mask置1，以便FixedStep重加噪后钳制为零。mask语义仍为1=condition、0=generation，不反转；有效10槽保持0。实际UniPC接口并不接收该flat mask作为额外未来输入。

原独立checker `verify_wam_clean_bridge.py:430–432`错误要求整个40×128 action mask都没有1，把已登记的无效槽零钳制视为特权action条件。旧合成测试的全零mask也没有覆盖真实原生合同，不能替代该合同验证。

独立只读诊断已核全部7条initial/point0：vision按`(1,48,11,60,40)`仅首latent帧为1，其余全0；action按`(40,128)`前10槽为0、后118槽为1；inactive初始值全零、当前vision首帧与保存first_condition_latent逐位相同，initial与实际point0逐位相同。完整证据为037目录`aux/initial_native_mask_diagnosis.json`，SHA `829b3d4eb321d1b17647e56dc8207eb69e268c47ac0d5b492f673b85e07ea6c2`。本项不依据任何未来state或任务成绩修订规则。

## 修正边界

新增`verify_wam_clean_bridge_native_mask.py`，显式复用原独立检查器其余全部数学、来源、预算与配对检查，仅替换initial检查为**完整原生mask精确重建**：

- vision只有当前首latent帧为1，任何future vision条件均拒绝；有效10 action槽任何条件均拒绝。
- 无效118槽必须精确为1且对应initial/实际sample为零，不能只删除原any判断或接受任意binary mask。
- FP32、完整layout、request/seed/variant/caption、initial与point0逐位一致及当前图像latent逐位绑定均保持。

原checker写出路径固定为prepared.root/review/verified.json。新wrapper每次先完整核原prepared与真实root/SHA，仅对该次检查的内存副本作输出路由到039新root；config、请求、source、qualification及原prepared引用不变。磁盘manifest及旧目录不写入。路由源码及配置进入新审计source closure。

新root为`outputs/wam_stability_20261021/clean_bridge_native_mask_audit_20261023/`，配置`configs/wam_clean_bridge_native_mask_audit_20261023.json`。新owner冻结自身、wrapper、定向测试、配置、计划及原执行来源，核037模型正常退出/GPU释放与原checker actual wait1、原owner False；CPU空CUDA/LD，仅启动一次独立审计child，1800秒上限、无自动重试、真实wait/reap。

首次039准备因新owner漏建preparation父目录而在写manifest前停止，未创建orchestration或启动独立checker。该部分源码快照已原字节迁移到同级`clean_bridge_native_mask_audit_20261023_preparation_failed/`，工具exit1及逐文件SHA保存于`outputs/clean_bridge_continuation_20261023/native_mask_preparation_failure.json`。修正新owner的父目录创建后，在空的原定root重新准备；没有重跑模型或已消费的独立审计，也不覆盖失败现场。

必要测试包括原生mask正控和有效action条件、未来vision条件、错误无效槽、非零无效initial、当前latent不匹配、来源/输出路由与完整原检查器复用拒错。新规则在读取下游训练或任务评分前固定，绝不据结果放宽。

## 完成与证据口径

新审计必须完整复算245节点、3raw/zero配对、3历史输出、非零实际velocity改变、完整来源/预算，且新CPUowner正常退出，才给补充的`bridge_qualified=True`。原037 owner False、旧checker错误与所有消费SHA保持原样；新True只对应补充审计及原完整模型证据的联合资格，不改写历史封套。

另以独立stdlib/NumPy peer复核保存数据和联合生命周期。新增预算为0模型初始化、0denoise、0fit、0物理、0fresh、一次独立CPU检查；peer不运行模型。若仍有其他来源/数学错误保留False，不补跑GPU或调门槛。

本项不证明任务收益，不自动启动训练。034数据资格加上037完整模型证据与本项通过的独立审计，可作为038准备阶段的桥接前提；最终预留24次稳定成功≥20/24且每任务≥4/8不变。
