# 034：完整三十条原生示范的固定收尾训练数据协议

本计划在新物理采集前登记。所有工作仅在用户指定的share-mixed仓库，CPU/OSMesa显式空CUDA/LD，seed42。原030三十条28/30稳定、资格False及031/032负结果完整保留，不修改旧数据、末40定义或判定阈值。

固定沿用原030三任务各十条、完整三十条名单/初始化/XML/native动作及顺序，不新增来源、不读取HDF/future RGB、不替换失败条目。每条执行全部N条native命令；在N端点读取实际controller SITE位置和旋转，后80步逐次向原OSC传该绝对goal，同时继续末条native夹爪命令。前40步是事先固定的释放/settling，后40端点检查原生成功谓词。无任务/接触/未来信息进入命令，无cache/qpos/ctrl写入。此新数据收尾不同于030零delta保持，不能倒推原030成功，也不改变最终WAM任务时长。

原前四anchor floor(j*(N-40)/3)保留，第五固定移到N+40；该40步训练标签完整覆盖已settling后的测量窗口。每个anchor只渲染当前两相机，20Hz端点、SITE/BODY观测、TCP外参、delta、command与opening分开保存。固定总预算30环境/6180步/150canvas，0WAM/fit/fresh。无自动重试、早停或产物覆盖。质量失败仍完成全名单；来源或数值合同失败立即停止。

迁移后的032准备及独立审计作为前置来源。所有历史引用保留原path/SHA，经只读RelocatedArtifacts解析；旧Cosmos缺失只用历史精确快照证明，不宣称旧live源码存在。冻结本阶段源码/配置/测试、当前cv2环境探针与运行解释器，运行前后逐SHA复核。独立checker不导入collector或控制算法，逐条比较原030完整N前缀，包括controller及ragged gripper cache；原前四raw相机逐值相同，当前canvas按明确像素合同检查。新第五anchor独立核其时刻、相机身份、渲染前后物理状态与重组canvas，历史canvas≤1 uint8例外单列，不宣称bitwise等价。

全部三十条末40原生成功才能获得eligible_for_model_training_registration；来源/标签审核通过与数据质量分开。model_training_authorized/fresh_performance_authorized仍False，真实模型采集和fit需另行固定来源、噪声、预算与验收。若未过，不继续无限修两条；须另登记有效native窗口的用途，不能追认此全三十条资格成功。

下一模型阶段方向为冻结8400非EMA、显式真实FP32 sample的C0 clean残差、训练噪声与seed42/43隔离、有限raw及两轮on-policy上下文。CPU合成bridge不代表真实WAM资格；当前外部Cosmos源有变动，先固定实际执行源码并完成真实zero-head精确复现。最终目标不变：仅当前图像/任务生成参考，经固定S2A，在预留12新初态×seed42/43的24条中至少20条稳定、各任务至少4/8、末40个20Hz端点持续满足原生条件；等待推理暂停物理，不考核实时。
