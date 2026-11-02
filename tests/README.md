# 测试与共享 fixture

| 领域 | 当前实际入口与审核模块数（不含 __init__） |
| --- | ---: |
| [research](research/index.md) | 346 |
| [support](support/index.md) | 2 |
| [unit](unit/index.md) | 123 |

从仓库根运行 `CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m pytest`。临时文件位于 `outputs/`，并发 worker 使用各自独占目录。

保留原测试函数、参数和节点身份；重定位测试文件不删除失败。集成项需显式只读输入与环境，未执行/deselected 不算通过。软件测试、保存数据复算、真实物理/模型和最终任务验收分别报告。

当前读取用显式 current identity；冻结声明及历史 fixture 保留原 path/SHA，通过历史 locator 核字节。不能以全局 Path/open hook、裸模块 alias 或修改旧 JSON 消除迁移错误。详见[来源协议](../docs/migrations/README.md)及[旧测试说明](../docs/history/tests_README_20261029.md)。
