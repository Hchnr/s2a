# 172 已保存状态的接触几何证据

`outputs/departure_contact_geometry_20261025/verified.json`完成085成功、162与168失败case03各endpoint50–61共36个保存状态的几何核验。各自model.xml及原sim_states来源绑定，重建SITE位置/朝向与采集数组1e-9一致，未推进仿真，也不报告历史接触力。

三组endpoint50–54均存在bowl与左右手指接触。162/168从endpoint55起已无任何机器人接触，085保持左右手指接触至endpoint59，endpoint60后才无接触。endpoint55为step54完成后，接触丢失发生在171物体/开度大分离的step55之前。原日志grasp_contact=false没有捕获此真实几何差异，不以该flag定因。

窗口52–56三组夹爪命令均0、电机缓存均[-1,+1]、路由均native_close_load，但实测手指位置在失败组快速收拢；不能把电机缓存视作实测开度或把同命令等同相同接触状态。该证据支持继续研究负载下接触维持与反馈逆，而非单纯压平均位姿误差；仍不能仅凭端点几何断言唯一原因。

162已完整4907周期、36采集/独立检查stage及owner实际wait0/reap与来源核验，性能13/18，原门槛失败。168最近16条性能13/16，剩余继续。WAM/fresh保持关闭，总目标未完成。
