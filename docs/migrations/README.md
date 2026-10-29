# 结构迁移与来源身份

[全局设计](../architecture/repository_design.md)规定顺序及验收。
[基线清单](baseline_20261029.json)只记录迁移前的3276项身份，不预判新路径；
基线commit为`7b2712850dea54090d61e564380f88825ad8b8f6`，清单SHA256为
`1a7d3737696e5c6cb1c4d22bc572257dd5994493a01bcd991228df9f9eb23897`。

## D1：路径及历史字节定位

`s2a.infrastructure.repository_paths.repository_root()`从明确anchor查找项目标志，
不依赖固定目录深度；`repo_path()`校验仓库相对路径。

`s2a.infrastructure.source_identity.SourceIdentityLocator`接收明确root、baseline
commit、catalog_ref、migration_refs及outputs内cache目录。`resolve(declared_ref)`
首先确认原路径/SHA已登记，再返回declared_ref、storage_ref、migration_ref、
catalog_ref、原Git来源及resolution_kind。调用方最后执行`finish()`核验整批。

后续迁移清单绑定相同catalog_ref，moves每项包含：

```json
{
  "old_path": "src/s2a/example.py",
  "original_sha256": "<baseline SHA256>",
  "new_path": "src/s2a/research/example.py",
  "current_sha256": "<current storage SHA256>"
}
```

路径在迁移后改字节时，原身份定位原Git blob，当前源码另有指纹。已登记current
目标漂移会失败，不通过fallback掩盖。未知版本、错误blob、缓存损坏、路径越界
及旧根访问均拒绝；不会自动导入找到的源码。结果scope固定为`byte_identity_only`，
不授予历史代码执行、runtime或控制资格。旧JSON与旧helper目前保持原样。

D1专项36项CPU测试通过（最终session28118实际exit0，35.12s），新文件Ruff及
format检查通过。真实3276项catalog接线、active源码定位及本地Git收尾通过。

## 测试发现基线

`outputs/repository_baseline_collection_20261029/`保存迁移前实际collect-only收据、
日志与nodeids。session17648/child3375668实际wait/reap；收集到4976节点，5项既有
collection错误，退出码1。错误均为CPU环境PyArrow要求NumPy≥2.0，而当前用于
数值回归的解释器为NumPy1.26.4。没有执行测试用例；不能将此记录称为全仓通过。

后续按旧→新文件映射比较测试node集合，并分别报告新增测试和这些既有错误。
收集器中的模型/任务零计数字段没有独立运行时探针；本基线只用作测试发现与
源码未变的证据，不据此授予任何模型或仿真执行资格。
