# 035：迁移环境下真实WAM clean残差桥接资格

本项在新模型初始化前登记，只检查真实8400单条件采样与clean残差的执行合同，不拟合、不评分未来state或任务成功。所有命令从用户指定的share-mixed S2A执行，输出仅在本仓库；模型只用当前可见2/3对应允许UUID，禁止沿用旧数字GPU6入口。

预检确认本地模型环境缺失，但可只读借用share-mixed Cosmos的Python3.13.14/torch2.10cu130。该Cosmos工作树被并发更新，不直接作为执行源码。本项将外部/share/project/eai_pwm/repos/cosmos-framework-pwm的clean已登记commit归档到本S2A新output/source/cosmos，并逐文件冻结实际执行源与导入路径；其26历史核心源中23项相同，三项原生action mask修订是新来源合同，不能冒称原26完全相同。外部仓库、环境与数据均只读。所有实际S2A模型帮助函数及新bridge/worker/checker也固定SHA/快照。

模型固定8400非EMA、SDPA、无compile、无small-action metadata重用、单rank、guidance1.0、shift5、35步、完整128维仅前10有效、H40。新源码原生mask与登记的S2A mask wrapper均显式记录。checkpoint163文件先逐stat核历史登记，三份小metadata重新SHA；本资格不重新读取32GB权重作完整SHA，不把stat一致表述为本次完整权重内容验证。真实加载及输出回归另给执行证据。

固定消费029原free_generations的0/16/32：三个开发task的第一个anchor、seed42，仅当前image+task；不传measured state/future/action，不打开训练EP或fresh。旧request只允许vision_path按安全resolver重定位，当前PNG SHA与原声明必须相同。保留历史请求和response引用，拒替换图像或任务。noise roster明列训练101/102与评测42/43，本项只执行eval42，不使用训练噪声或拟合。

固定预算一次模型初始化、7完整请求、245 denoise。先三个raw，再同三个zero clean head，最后第一个固定request加常数clean残差bias，各active维1e-5，其他系数0。它是合成非零资格，不是训练候选。每节点原head/denoise/velocity仅一次。35节点均读取实际FP32 sampler latent与实际sigma；packed网络时间另记，禁止由整数t推sigma或用BF16 packed action代替实际sample。clean C0按FP64 sample-sigma*原masked v，features固定[hidden1280,C0active10]，残差改变真正返回sampler的v=原v-residual/sigma，再按原输出dtype舍入。terminal sigma0不调用残差。

逐节点保存完整FP32 latent、masked原始velocity、真实deployed velocity、raw head、hidden、packed action/video/时间/condition/offset、C0及predicted residual，保证独立checker可重算。raw↔zero的全35节点、最终normalized/raw action与真实初始条件须逐位相同；vision和inactive118必须保持原bits；非零返回必须确实改变sampler消费的至少一项active值并符合独立FP64与舍入合同。原模型参数/RNG仅原生成路径消费，观察器不改其状态；实际参数版本前后核验。

三个raw输出还须与原028/029保存响应全有效及完整128维精确回归。source/预算/数值/ownership错误拒绝协议；历史输出不一致可以完整保存本轮证据但bridge_qualified=False，不重试、重选样本、调bias或宣称原始WAM等价。passed/complete只指证据完整，bridge_qualified另列。无自动重试/覆盖，只停止自己的child；actual wait0/reap/noSignal、GPU UUID/显存释放正常后独立CPU审计。

model_training_authorized与fresh_performance_authorized始终False。资格通过也不代表生成质量改善，下一真实训练采集/有限fit须另登记来源与预算；034新native数据资格与本项桥接资格各自完成后才考虑该登记。最终真实WAM→固定S2A的24条稳定成功≥20/24、各任务≥4/8、末40原生持续成功的总目标不变，当前尚未达成。
