# 8400短重新观测WAM闭环 — 2026-10-21（Asia/Shanghai）

状态：完整005/006/009证据及007资格之后冻结，新增固定12条，不打开新初态。目标仍是多任务稳定成功，不把旧cream单例、ever或正确参考资格当达标。

005新v3夹爪24条完整仍stable0/24；006同8400十二条为stable1/12，仅alpha0/cream尾486端点连续成功，raw两组0/3。009新latest15600的全部九项短/长state指标均差于8400及15400，未给出替换候选证据。8400的H8/H16 state三指标最优，且旧v2/新v3两轮alpha0/cream均稳定，因此选择同一8400统一评价三个任务，不拼接按任务权重/seed/策略。

007的8/16步重观测、v3无gov/恢复gov共12条/7200步正确参考已完整独立通过；短gov可能不断重置桥和丢弃关闭后缀，旧WAM真实首8/16前缀已记录这种风险。本轮先去掉gov，直接评价短前缀的图像闭环，不引入新的桥策略或控制增益。相同response逆及v3参数保留。

| 独立配置profile | maxobs | governor | 模型参考 | 任务数/seed |
| --- | ---: | --- | --- | --- |
| interval8_alpha1 | 8 | 无 | raw | 3旧task/init0/s42 |
| interval8_alpha0 | 8 | 无 | 逐delta旋转置单位 | 同上 |
| interval16_alpha1 | 16 | 无 | raw | 同上 |
| interval16_alpha0 | 16 | 无 | 逐delta旋转置单位 | 同上 |

每profile的variant均名为position_hold，由独立config/output、明确model_rotation_alpha及model_reference_label识别。alpha0是模型参考干预，保持每query实测anchor朝向，仍会通过局部平移积分改变world路径；不是完整WAM或纯S2A。原始40×128输出完整保存。

H40/20Hz、execution horizon40、采样35步/guidance1/shift5、每条600实际步、最后40原生成功、OSC inverse.5/rate5、v3全部参数及三个旧初态均不变。仅每8/16物理步重query；没有gov，因此实际消费8/16端点，其余生成后缀丢弃。600步分别75/38次query，共12条/7200步、最多678个H40请求；不因中途成功或失败改变预算。

只在物理GPU4/9444和5/9445各持有一个模型服务。每卡两profile的模型/采样/conditioning完全相同，仅CPU端alpha不同，可共享同一私有服务，顺序一worker执行；server生成产物所在alpha1根与alpha0请求/物理日志根分别绑定，不能混作来源。CPU编排/OSMesa空CUDA与LD。旧服务必须完整release且自有PID退出、卡空闲和端口可用；不触碰其他用户进程。

启动前复核005/006完整审计及正常release、007资格、009完整state及新旧权重指纹stat；配置/实际依赖/预算提交SHA固定。复用同一冻结position执行函数和矩阵入口，不修改旧服务或物理实现。独立新编排绑定四config及单服务两job，一次失败保存停止，不自动重跑。每case首query与原8400 intent_hold的PNG、完整raw/normalized、初态/anchor精确回归。

独立核验可直接复用冻结006完整verify_case：四config都叫position_hold，其既有首raw来源为同8400旧intent_hold，alpha后处理另由实际字段独立重建。旧checker中first_query003_regression/006文字只是继承字段，010注册、分母和输出另列；不重算或计入既有oracle。须核每case75/38请求、query物理state/time不变、motor/qpos/开度连续、三值命令/arm速率界、有效参考与最后40谓词，全部输入SHA前后复验。所有12结果完整报告，不按成功seed挑选。

若出现一致多任务能力，再冻结候选到预留新初态；本预算不自动执行fresh验收、参考校正器或额外训练。更多采样步等其它推理候选也必须单独注册，不中途更换本轮35步。
