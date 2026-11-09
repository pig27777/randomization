# 本机资料归档

导入的聊天记录与原压缩包放在 `conversations/<日期>/`。该目录由根 `.gitignore` 排除，只在本机保留；不要提交原始会话内容。项目事实与实验结论继续维护在各线文档和报告中。

## 2026-11-08 聊天备份

| 原根目录路径 | 归档路径 |
| --- | --- |
| `claude-history-random-20261108(2).zip` | `conversations/2026-11-08/export.zip` |
| `claude-history-random-20261108(2)/` | `conversations/2026-11-08/extracted/` |

原压缩包与已解压的 58 个文件均完整保留。压缩包 SHA256：

```text
ca5c55aa1dd6f946357e18b49f2327ac5cb0282bf8f8665f6ff643a5de5f2a55
```

本机恢复说明位于 `conversations/2026-11-08/extracted/claude-history-random-20261108/README.md`。归档仅调整外层路径，内部目录和文件内容保持原样；新克隆的仓库不包含这些本机备份。

同目录的 `SHA256SUMS` 记录 ZIP 和全部解压文件的 SHA256。之后可在仓库根目录重新校验：

```sh
(cd archive/conversations/2026-11-08 && sha256sum -c SHA256SUMS)
```
