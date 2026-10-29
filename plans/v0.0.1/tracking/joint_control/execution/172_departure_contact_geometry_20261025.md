# 172 已保存端点接触几何核验

对085/162/168已独立审查case03的endpoint50–61原sim_states，用各自实际model.xml重建qpos/qvel/time，仅mj_forward不积分，控制SITE矩阵与原采集数组1e-9绑定。列出机器人涉及接触的body pair、距离和位置；不重建历史力、warmstart或控制器，不调用mj_step。

endpoint k对应完成k个周期、即step k-1之后，避免与171逐step错一位。本证据用于检查54–58物体分离窗口的实际几何约束消失时间，不能宣称唯一动力学因果或已完成新控制策略。原回归/门槛不变。
