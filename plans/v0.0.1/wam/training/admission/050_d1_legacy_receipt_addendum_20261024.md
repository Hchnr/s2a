# 050：D1完整数值结果的旧格式退出收据补充核验

046实际D1已完成300请求/10500节点，原独立capture审查与原D1 owner均True；模型、审查及CPU wrapper实际wait0/reap，GPU已释放。046额外supervisor在post-wait `check_worker`误读旧release不存在的`actual_wait`字段，持久化False和KeyError。此处是接续代码错误，不改写原失败，也不将False改成True。

原519源/快照包括错误代码保持字节不变。旧owned_command格式真实wait来自exit（三字段），reap来自release（七字段、无actual_wait）；新durable格式两文件都有actual_wait。新增明确format的只读收据核验，不合成字段，不转换JSON/schema，不泛化接受任意错误。

050只核验并发布独立补充结果：完整D1原producer/audit及引用、两条legacy实际Popen收据、GPU释放、原source闭包、全部已消费输入SHA；CPU wrapper的durable exit0/reap/no-signal与**唯一post-check KeyError('actual_wait')**，所有记录后代已reaped；保留原046 owner/supervisor False。必须证明失败只在后置收据读取，不是模型/数值/退出失败。没有原数值资格、进程非零退出、signal、未reap或其他异常均拒绝。

预算0模型/denoise/fit/QR/physics/fresh、0数组decode、不重采、不重跑独立数学；只读完整已生成证据和文件SHA。050源码、配置、QA与快照单独冻结，原数据和519源不变。原W1失败与累计尝试继续保留。

随后047显式绑定050结果，才可复用这份D1并进行固定W2；047/048尚未执行的准备代码可修正为明确legacy/durable分支，并保留修订前源码/SHA安全副本及旧QA。不得修改运行过的046代码或向旧收据补字段。实际W2/STATE/任务门及最终20/24、各任务4/8均不变。
