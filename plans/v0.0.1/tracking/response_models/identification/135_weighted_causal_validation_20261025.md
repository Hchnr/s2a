# 135 加权因果残差原性能验证

冻结116唯一prior及134固定20周期/单位ridge/七组核。六holdout逐步预测后observe，仅已完成反馈进入历史；六局部anchor仅原完整前缀observe，七分支共享同一实例只读predict，不更新probe。每次查询局部系数用独立增广最小二乘复核。原24holdout/24局部门槛不变，全部必须通过；在线求解明确计数，无全局fit/物理/世界模型/fresh。冻结来源及owned_stage实际wait/reap。失败不进入控制。
