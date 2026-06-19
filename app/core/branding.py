from __future__ import annotations


PRODUCT_NAME = "LandScout Agent"
PRODUCT_NAME_CN = "地脉智能体"
PRODUCT_DISPLAY_NAME = f"{PRODUCT_NAME}（{PRODUCT_NAME_CN}）"
PRODUCT_TAGLINE = "面向中国房地产开发商的住宅拿地早期信号智能体工作台"

LANDSCOUT_AGENT_WORKFLOW: tuple[tuple[str, str], ...] = (
    ("Source Scout", "发现和筛选政府公开数据源"),
    ("Crawler Agent", "抓取公开页面、附件和接口"),
    ("Evidence Analyst", "抽取可追溯的政策、产业、交通和供地信号"),
    ("Geo Analyst", "把文档信号定位到上海候选片区"),
    ("Site Ranker", "按人口流入前置信号和抢先拿地窗口评分"),
    ("Visualization Agent", "生成报告、地图和可视化摘要"),
)


def landscout_agent_workflow_payload() -> list[dict[str, str]]:
    return [{"agent": name, "role": role} for name, role in LANDSCOUT_AGENT_WORKFLOW]


# Compatibility aliases for earlier local code/imports.
AGENT_WORKFLOW = LANDSCOUT_AGENT_WORKFLOW


def agent_workflow_payload() -> list[dict[str, str]]:
    return landscout_agent_workflow_payload()
