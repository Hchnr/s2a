# 153 合格forward受限逆控制绑定

保留原083半旋转补偿与081/079夹爪策略、原正确参考18条/4907步/seed42。替换平移为149合格146模型的motion+dt*linear_velocity有界最小二乘，速率5/s。以实际当前source_from_SITE刚体变换将desired source及已生成reference转到SITE，匹配模型当前SITE/world twist，禁止混用参考点速度。

在线151当前状态、当前wrench、150仅完成历史；每步先夹爪决策/固定旋转命令，再针对这两命令求平移逆解。实际执行后刷新端点并更新history，记录实际模型输入、完整响应预测、物理Jacobian/offset、目标SITE/速度及命令边界。所有候选仅查询当前可得信息。逆解实际command同步converter上周期命令，避免错误速率历史。

当前阶段仅执行器集成，不自动启动物理。后续必须配置/真实CLI dry/独立命令重算及因果准入、冻结参数/来源、持久owner完整18条回归及独立检查。位置RMSE<=.01m、SO(3)RMSE<=.05rad及稳定18/18保持；WAM开发/最终24新初态门槛保持。不得将148离线合格当控制成功。
