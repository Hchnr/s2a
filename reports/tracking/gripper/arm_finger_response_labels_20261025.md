# 179 手指间距响应标签准备

`outputs/arm_finger_response_labels_20261025/result.json`与`labels.npz`已生成，保留146实际方程顺序：24fit4865自然周期、207arm成对干预导数方程；6holdout1315标签独立保存。标签是实测两手指间距变化delta(q_left-q_right)，单位米，不是motor缓存或期望开度。

自然完整n+1实测fingerq与087每行当前fingerq1e-12核对，成对干预前端点手指q1e-12一致、用原实际normalized命令差分归一化；全部来源指纹保存。未使用176/178失败输入或标签训练，未拟合/改模型，无新物理、WAM或fresh。仍需独立从raw sim_state/关节索引复核标签，再进行可辨识性与原划分验证；当前不宣称控制资格。

178case03已独立机械通过而任务失败。路由从step38进入native_close_load且学习权重确为0；相对085成功基线step53 source位置差约5.6微米，但step55开度差已.0958，不能认为单纯末端精度即可保证接触。此观察支持检查arm与手指测量状态耦合，不能直接定唯一因果。
