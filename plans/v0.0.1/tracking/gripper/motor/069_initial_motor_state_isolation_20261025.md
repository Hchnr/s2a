# 069 当前夹爪电机目标保持的初态隔离

068失败wine/demo_11的第0步：真实复位cache=[0]，执行器将其对齐成[0.702806,-0.702806]；原native实际命令-1/open，S2A却+1/close。实测开度均0.849522，而下一真实期望开度0.768007来自机械响应，不代表闭合控制意图。需要隔离“把measured opening当motor target”这一初态干预。

固定唯一CPU对照：同训练task1/demo_11初态和228步正确世界参考，seed42、原position_hold、原OSC、无额外等待、command preview0。只改变初始化：保留当前真实可读Panda current_action目标；若reset是长度1则按该接口原broadcast语义复制为长度2，不按实测开度重设目标。不得读取expert/native未来命令作为在线输入，不恢复历史专家cache。原baseline case15和候选case16失败收据保留。

1初始化、228物理步，0模型/fit/fresh。全部实测端点、原desired与命令、真实cache及normal收据记录。独立核验初始目标保持（而非旧开度对齐公式）、完整控制/夹爪/参考、标签、图像与时间；报告原native末40稳定及10mm/.05rad门槛。若失败不调阈值重跑，若通过须另登记广泛正确参考回归再推进模型。

这只检验一个初态控制干预，不预设它解释全部接触失败或适用于任何机器人。当前仅登记，尚未运行，原最终目标与门槛不变。
