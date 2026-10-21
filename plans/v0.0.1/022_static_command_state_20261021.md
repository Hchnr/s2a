# 022：当前8400的固定任务描述配对诊断

本项在任何新增文本条件推理前登记，独立于020/021。现有只读来源审计已发现一个明确尚未隔离的条件差异：当前在线短任务会正确传入renderer，但发布训练caption的command是较完整的抓取/搬运/释放描述。旧8800两种文本的state对照没有一致改善，不能据此断言当前8400的影响，也不预设文本差异就是失败原因。020/021所有参数、执行源、输入条件和结果保持不变。

仅用已有六个开发episode、24阶段样本/22不同anchor，不打开预留fresh。模型固定实际8400非EMA、active10 mask、H40/20Hz、35去噪步/g1/shift5/SDPA。每样本seed42/43，各48请求。先用完全相同task-only输入新增48次复现，完整raw及normalized128必须逐值精确对旧8400/g1缓存；任一不一致立即停止，不能继续解释为文本收益。然后同一服务仅新增48次固定expanded-command，共最多96次生成、0物理、0拟合。GPU7/端口10047；CPU显式空CUDA/LD，模型显式CUDA_VISIBLE_DEVICES=7，禁止0–3。服务不自动重启或重试。

每任务唯一静态长command，确定规则为原24样本manifest中该task首次出现样本的published H40 caption的command。任务选择和样本顺序原样，不根据误差或闭环成败挑描述。移除task_plan/current_subtask/其他示范阶段信息，task_title保留当前在线短任务原字节；相机、机器人、频率、时长等由现有服务按同一请求准备。所有该任务episode/阶段/seed都用完全相同描述，不读取当前对象、contact、成功谓词、未来state或真实阶段选择文字。该条件可作为用户完整任务说明，仍是推理条件干预，不能说原在线协议未变或认证原短任务成绩。

新增隔离准备/服务/编排/审计文件，复用未改的run_wam_state_matrix、几何解码、active mask和权重加载。服务在任何模型调用前把实际batch caption、请求名/seed、相机布局及有效mask另存持久账本；原生成目录清理不得丢失该账本。先核两种实际caption除command外字典全部一致、expanded与登记描述一致，raw/norm/图像/seed/权重身份完整绑定；不以请求参数取代实际内存caption证据。先通过CPU条件差分检查及独立source审查再启动模型。

全部48+48完成后从当前实测pose积分预测delta、绕过S2A，比较真实未来state；分别报告H8/H16/H40位置RMSE、SO(3) RMSE、opening MAE，以及任务/episode/阶段分组。保持与历史外推对照原样。样本重复和重叠窗口不视为独立episode，不把state误差直接当任务成功。

固定后续候选门槛：H8与H16六项均不劣于task-only，且每个窗口至少一项降低10%；H40三项均不超过task-only的1.02倍。用position/10mm、SO3/.05rad、opening/.1的逐endpoint平方和定义共同归一化分数；每task在H8/H16的分数均不劣，至少5/6episode的H8/H16分数均不超过原1.02倍。若未满足，不改描述/阈值/seed/模型后重跑或追加物理；若满足，另行冻结统一闭环协议才允许物理任务验证。本项没有物理或fresh预算，不能以通过state门槛宣布稳定成功。
