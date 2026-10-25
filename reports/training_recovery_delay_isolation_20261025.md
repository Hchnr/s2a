# 064 wine参考等待隔离：执行审查通过，性能失败

固定同task1/demo_1训练初态、seed42、原S2A/控制器/夹爪，直接执行254步已生成正确参考，去掉063额外56步；1初始化、254物理步、0模型/拟合/fresh。所有输入及执行源码冻结，实际CPU/OSMesa空CUDA/LD，owner3412994/birth582633757、client3413007/birth582633788正常结束，无重试。

实际独立`verify_training_recovery_delay_isolation.py`正常exit0：完整254步控制/参考/夹爪回放、255实际端点标签/状态hash/时间、3当前图像及实际wait0/reap/no signal、全来源指纹复核通过。结果`outputs/training_recovery_delay_isolation_20261025/independent_verified.json` complete=True、mechanical_labels_images_passed=True、performance_qualified=False。

独立原生ever=True、terminal/stable=False，成功15步、尾连续0，与063/wine零偏移的最终计数相同；不能仅据计数声称轨迹逐值相同。删除额外56步没有消除失败，因此额外等待不是该失败的必要因素，也未证明它完全无影响。保持063失败及原门槛，不扩展恢复训练。

下一步比较原native与当前S2A在接触/放置前后的实际TCP、旋转、开度及motor cache；低平均误差仍可能在关键时刻产生差异，测得开度也不等于原夹爪控制意图。只做离线定位，不把专家action或物体真值接入WAM在线策略。原061候选0/6及fresh关闭不变，总目标未达成。

## 关键释放阶段的实测差异

实际对齐原native与064端点，保存case03/native_release_comparison.json及输入前后SHA。endpoint168两者TCP位置差约0.10mm，但native实测开度0.8714、S2A仅0.4438；原motor target已[1,-1]，S2A仍[0.25,-0.25]。S2A step163才触发reference_rise_and_measured_clearance_confirmed，目标开度0.5321时实际约0.3772。endpoint174两者motor均已打开，但原native继续成功而S2A开始失稳；后期即使位姿/开度接近，物体接触状态也不会自动回到原轨迹。该时序差异是可检验的释放滞后假设，尚非因果证明。

下一步应依据训练轨迹辨识夹爪目标到实测开度的前向响应，并用当前状态/历史及已生成开度参考做逆响应补偿；不能直接把历史native动作作为在线输出，也不以强制释放阈值掩盖该问题。只调末段阈值或假定实测开度就是无延迟motor command均缺少依据。
