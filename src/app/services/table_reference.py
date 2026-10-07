"""表格指代查询检测（search 与 agent_executor 共享的单一出口）。

背景（2026-08-12 锁定回归 R17/R18/R20/R22）："第二张表""展开第一张"
等中文指代查询无法从自身词元匹配英文表格文本（``_table_block_matches_query``
要求 ≥2 token 交集），若被全灭过滤则检索拿不到任何表格证据。指代检测
是检索放宽与会话锚点注入的共同前提，search.QueryService 与
AgentExecutor 两处都要用。

两处曾各自维护一份正则与检测方法（复制粘贴防漂移失败，code review
2026-08-12 收敛为共享模块）；本模块只依赖 ``re``，无任何业务依赖，
不会引入循环导入。
"""

from __future__ import annotations

import re

# 表格指代模式：序数指代（第N张）、指示代词（这张/那张/这个表/那个表）、
# 时间/位置回溯（刚才/前面/上面/之前/上一…表）、展开指令（展开…表）。
TABLE_REFERENCE_RE = re.compile(
    r"第[一二两三四五六七八九十百\d]+张|"
    r"这张|那张|这个表|那个表|"
    r"(?:刚才|前面|上面|之前|上一).{0,12}(?:表|两列|两项成绩|各列成绩|数值)|"
    r"展开.{0,10}表"
)


def is_table_reference_query(question: str | None) -> bool:
    """判断 *question* 是否为表格指代查询（无法从词元检索到表格内容）。

    Example: "现在展开第二张" → True；"论文中有哪些表格" → False。
    """
    return bool(TABLE_REFERENCE_RE.search(question or ""))
