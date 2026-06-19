# LandScout Agent（地脉智能体）

`LandScout Agent（地脉智能体）` 是一个 Python MVP：从上海政府公开页面、公开附件和公开接口抓取信息，解析 HTML / Excel / CSV / PDF / Word / JSON，通过 LLM 结构化抽取政府信号，地理编码后计算房地产开发选址机会分，并输出报告、地图、CSV 和证据包。

产品名采用 agentic / multi-agent 叙事，但当前实现仍是单进程编排；代码中已经把工作流拆成 Source Scout、Crawler Agent、Evidence Analyst、Geo Analyst、Site Ranker 和 Visualization Agent 六个职责层，后续可以逐步拆成真正的多智能体执行。

## 业务目标

服务对象是中国房地产开发商。核心问题不是判断“哪里现在已经成熟宜居”，而是在人口真正开始流入之前，从政策支持、产业导入、重大项目、轨道交通、规划批复、供地节奏等早期公开信号里，提前发现未来住房需求可能增长的区域，帮助开发商更早做投拓判断和拿地准备。

住宅推荐模型因此优先关注：

- 未来人口流入潜力：产业项目、招商签约、重大工程和就业导入信号。
- 人口流入前置信号：规划、批复、专项政策、交通和公共服务建设前期信号。
- 抢先拿地窗口：早期供地、控规/批复、尚未形成充分住宅供应和竞品暴露的机会。
- 风险约束：证据不足、定位不准、已有住宅供应压力、区域已经成熟导致竞争加剧。

当前先聚焦上海，后续应扩展 AI 辅助的数据源发现、文档检索、爬虫生成、MCP/接口接入、文档分析和可视化决策工作流。

## 安装

```bash
make install
```

如果当前系统没有 `make`：

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Playwright 仅用于渲染公开页面和发现公开 XHR，不用于绕过访问限制。首次使用 Playwright live 源时可安装浏览器：

```bash
python -m playwright install chromium
```

## `.env` 配置

复制 `.env.example` 为 `.env` 后配置：

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_FAST_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_PROXY=
AMAP_KEY=...
APP_NAME=LandScout Agent（地脉智能体）
DATABASE_URL=postgresql+psycopg://gov:gov@localhost:5432/landscout_agent
```

- `OPENAI_API_KEY`：live demo 必需。为空时，`--live` 会立即失败并提示，不会用假数据冒充 LLM 结果。
- `OPENAI_MODEL` / `OPENAI_FAST_MODEL` / `OPENAI_EMBEDDING_MODEL`：分别用于深度抽取与 memo、轻量 triage / source scout，以及 embedding index。
- `OPENAI_PROXY`：可选。只给 OpenAI SDK 使用，不影响政府网站抓取。常见本地代理示例：`http://127.0.0.1:7890`。
- `AMAP_KEY`：可选。为空时使用内置上海 gazetteer fallback，并记录较低 `geo_confidence`。
- `DATABASE_URL`：默认 docker-compose 使用 Postgres + PostGIS；未配置时可使用 SQLite fallback。

本地运行时项目会优先读取当前仓库的 `.env`。如果终端里残留了旧的 `OPENAI_API_KEY`，重启服务后也会以 `.env` 中的新值为准。

国内公开网站抓取建议使用直连或 VPN 规则模式。理想配置是：`.gov.cn`、`.sh.gov.cn`、区政府站和上海公开平台直连；OpenAI 通过 `OPENAI_PROXY` 单独走本地代理。不要把全局 VPN 强制套在所有国内政府站上，否则抓取会明显变慢。

## 一条命令 Demo

```bash
make landscout-demo
```

等价于：

```bash
python -m app.cli landscout-demo --live --days 540 --top-k 8
```

其他命令：

```bash
python -m app.cli recommend-residential --live --days 540 --top-k 8 --source-limit 12
python -m app.cli discover-web-sources --max-sources 8 --query-limit 6
python -m app.cli recommend-residential --live --days 540 --top-k 8 --source-limit 12 --discover-sources --dynamic-source-limit 5
python -m app.cli landscout-demo --live --days 540 --top-k 8
python -m app.cli discover-source --source-id sh_land_market
python -m app.cli fetch-source --source-id sh_fgw_major_projects --pages 2
python -m app.cli score --run-id <run_id>
python -m app.cli render --run-id <run_id>
python -m app.cli submit-batch --batch-file outputs/shanghai/<run_id>/batch_requests.jsonl
```

住宅开发推荐命令会抓取/读取上海公开信号，解析文档，抽取事件，计算住宅专项评分，并输出住宅开发机会报告：

```bash
python -m app.cli recommend-residential --days 540 --top-k 8
```

加 `--live` 时会抓取 live 官方公开源并要求配置 `OPENAI_API_KEY`；不加 `--live` 时使用 fixture 数据跑通完整链路。

无 live key 时可运行 fixture 全链路测试：

```bash
python -m app.cli landscout-demo --days 540 --top-k 8
```

## 网页工作台

启动本地网页：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

然后打开：

```text
http://127.0.0.1:8000
```

网页不会自动抓取或分析。需要在页面里选择城市、数据模式、内置源数量、回看天数、Top K，可选填写自定义源，然后点击“搜索并分析”。当前城市选择只开放“上海”，其他城市等后续扩展。

页面右侧包含机会地图。运行后会用候选区域的中心经纬度和半径画半透明覆盖圆：

- 填入“高德地图 API Key”后，页面会把该 Key 传给后端地理编码，并在前端加载高德地图，用 `AMap.Circle` 绘制覆盖圆。
- 如果高德控制台要求安全密钥，可填“高德安全密钥”。
- 不填高德 Key 时，页面会显示上海坐标示意图，仍然标出候选区域中心坐标和覆盖范围。

自定义源支持两种格式：

```text
浦东规划 | https://www.pudong.gov.cn/ | 住宅,地块,规划
临港公示 | https://www.lingang.gov.cn/ | 产业,招商,公示
```

也可以粘贴 JSON：

```json
{
  "sources": [
    {
      "name": "浦东规划",
      "base_urls": ["https://www.pudong.gov.cn/"],
      "keywords": ["住宅", "地块", "规划"],
      "max_pages": 3
    }
  ]
}
```

选择“公开网站抓取”时仍需要 OpenAI API Key；可以在网页左侧临时填写，也可以由服务端 `.env` / Render 环境变量提供。选择“演示数据”时不需要。

## Render 公开部署

仓库根目录已经包含 `render.yaml`，可在 Render 里作为 Blueprint 或 Python Web Service 部署。默认公开网页可以打开并运行“演示数据”。如果要在 Render 上跑“公开网站抓取”，用户可以在网页左侧填写 OpenAI API Key；也可以在 Render 服务的 Environment 里配置服务端默认 Key：

- `OPENAI_API_KEY`：可选。配置后作为服务端默认 Key；不配置时，用户仍可在网页里临时填写。
- `OPENAI_MODEL` / `OPENAI_FAST_MODEL` / `OPENAI_EMBEDDING_MODEL`：可选，不填时使用默认模型。
- `AMAP_KEY`：可选；不填时仍可在网页输入高德 Key，或使用内置上海 gazetteer fallback。
- `OPENAI_PROXY`：Render 海外环境通常不需要配置。

Render 配置要点：

```yaml
buildCommand: |
  python -m pip install --upgrade pip
  python -m pip install -e .
  python -m playwright install chromium
startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
healthCheckPath: /health
```

Render 的公网服务适合展示网页和跑轻量演示；真实公开网站抓取可能耗时较长，免费实例还可能休眠，首次访问和首次抓取会慢一些。生产化时建议把抓取改成后台任务队列，前端只轮询任务状态。

## 数据源

配置文件位于 `app/sources/configs/shanghai_sources.yml`，包含：

- 上海土地市场
- 上海市规划和自然资源局
- 上海市发改委重大建设项目清单
- 上海市发改委项目批复结果
- 上海市建设工程交易服务中心
- 上海市交通委员会
- 上海住建委工程招投标
- 上海市公共数据开放平台
- 上海市统计局统计年鉴
- 上海市经济和信息化委员会
- 上海投资促进 / 商务招商动态
- 临港新片区管委会

每个源配置 `access_mode`、`priority`、`max_pages`、`delay`、`keywords` 和 `attachment_types`。住宅推荐 live 默认使用 `--source-limit 12`，覆盖当前配置的全部 12 个源；想快速试跑时可以降低该值。发改委重大建设项目源会抓取页面并下载公开 Excel 附件，解析所有 sheet。

数据源选取优先级：先选官方公开、可追溯、能产生人口流入前置信号的渠道，再看是否有结构化附件、公开接口或可稳定发现的详情页。`priority` 越小越早抓取，当前排序优先覆盖土地/规划、重大项目、批复、建设工程、交通、公共数据、统计、产业和重点片区。

## 合规说明

- 仅采集公开页面、公开附件和公开接口。
- 抓取前检查 `robots.txt`。
- 按域名串行低频访问，并使用源级 `delay`。
- 遇到验证码、登录、403、429 或明确访问限制时记录错误并停止该 URL，不做绕过。
- 不实现验证码识别、IP 池、账号登录绕过、cookie 池、指纹伪装或鼠标轨迹伪装。
- Playwright 仅用于公开页面渲染和公开 XHR 发现。
- 保存原始 URL、抓取时间、`content_hash`、原文路径和证据片段。

## 输出文件

成功运行后生成：

- `outputs/shanghai/latest/recommendation.md`
- `outputs/shanghai/latest/opportunity_map.html`
- `outputs/shanghai/latest/visual_summary.html`
- `outputs/shanghai/latest/events.csv`
- `outputs/shanghai/latest/signals.json`
- `outputs/shanghai/latest/evidence_pack.json`
- `outputs/shanghai/latest/pipeline.log`
- `outputs/shanghai/latest/investment_memo.md`
- `outputs/shanghai/latest/monitoring_queries.json`
- `outputs/shanghai/latest/embedding_index.json`
- `outputs/shanghai/latest/batch_requests.jsonl`
- `outputs/shanghai/latest/quality_review.json`
- `outputs/shanghai/latest/crawler_hints.json`

其中 `visual_summary.html` 是离线可打开的可视化摘要，包含 Agent 工作流、数据管线统计、Top 区域评分拆解、关键指标条形图、证据事件类型分布和月份趋势；`opportunity_map.html` 用于查看候选区和证据点的空间分布。

终端会按 `--top-k` 输出候选区域，包括区域名、分数、置信度、动作摘要、关键理由、主要风险和输出文件路径。

## 常见失败原因

- 源站返回 403 / 429：系统会记录错误，不绕过。
- 页面要求登录或出现验证码：系统会停止该 URL。
- 附件无法下载：记录在 `pipeline.log`。
- `OPENAI_API_KEY` 缺失或无效：live demo 在 LLM extraction 阶段失败并提示；无效 Key 会显示 OpenAI 返回的鉴权错误。
- `AMAP_KEY` 缺失：不会失败，使用上海 gazetteer fallback，但坐标置信度较低。
- Playwright 浏览器未安装：`playwright_with_network_discovery` 源会记录错误，可运行 `python -m playwright install chromium`。

## 测试

```bash
make test
```

或：

```bash
python -m pytest
```

测试覆盖 source registry、HTML parser、Excel/CSV/JSON/PDF parser、evidence quote validation、unit conversion、geo fallback、scoring 和 fixture pipeline。
