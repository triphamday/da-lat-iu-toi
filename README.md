# zvideo-creator-agent

GreenNode AgentBase agent for Zalo Video KAM workflows: creator performance analysis, content insights, handbook generation, policy RAG, and KAM copilot Q&A.

## What This Scaffold Includes

- LangChain Agent built with `create_agent` and domain tools for Creator analysis, ranking, insights, handbook generation, KAM action plans, and policy RAG.
- CSV/Excel creator analysis with cleaning, scoring, category ranking, Top 5 per category, and Top 30 selection.
- Policy/document assistant that returns cited excerpts from PDF, DOCX, TXT, or Markdown files.
- Handbook Markdown generation and optional PDF export.
- Simple local web chat client in `web/index.html`.

## Agent Tools

- `analyze_creator_performance_tool` - Creator Performance Analysis.
- `top_creator_report_tool` - Top 5 Creator per category and Top 30 Creator report.
- `category_insight_report_tool` - category strengths, trends, growth formulas, viewer behaviors.
- `content_growth_recommendation_tool` - data-backed content growth recommendations.
- `creator_handbook_tool` - Creator Handbook by category or all categories.
- `kam_action_plan_tool` - action plan for KAM follow-up.
- `answer_policy_question_tool` - RAG answers from guideline/reject/restrict documents with citations.

## Setup

Install Python 3.10+ first. On Windows, make sure `python` is available in PowerShell.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If pip cannot find `greennode-agentbase`, configure the GreenNode SDK package source used by your organization, then rerun `pip install -r requirements.txt`. The AgentBase skills define this SDK as the runtime server dependency, but it is not available from the default PyPI index in this environment.

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Configure IAM credentials in `.greennode.json` or `.env`. For LLM-backed answers, configure an OpenAI-compatible provider:

```env
LLM_API_KEY=
LLM_BASE_URL=https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1
LLM_MODEL=
```

GreenNode AIP is recommended for AgentBase projects because it is OpenAI-compatible and integrated with the platform. Use `/agentbase-llm` to create or list API keys and browse enabled models.

## Data Layout

Put creator data in `data/`, for example `data/creators.csv`, with these columns:

```text
creator_id,creator_name,category,followers,videos_this_month,follower_growth_rate,avg_views_per_video,engagement_rate,reject_rate,tier
```

Put policy docs in `knowledge/`:

- Huong dan xay dung noi dung tren Zalo Video
- Cac noi dung reject
- Cac noi dung restrict

Supported formats: PDF, DOCX, TXT, MD.

## Run Locally

```powershell
python main.py
```

Health check:

```powershell
curl http://127.0.0.1:8080/health
```

Creator analysis:

```powershell
curl -X POST http://127.0.0.1:8080/invocations `
  -H "Content-Type: application/json" `
  -d "{ \"message\": \"phan tich top creator\", \"creator_data_path\": \"data/creators.csv\" }"
```

Policy RAG:

```powershell
curl -X POST http://127.0.0.1:8080/invocations `
  -H "Content-Type: application/json" `
  -d "{ \"message\": \"Noi dung nao bi reject?\", \"knowledge_paths\": [\"knowledge\"] }"
```

Handbook PDF:

```powershell
curl -X POST http://127.0.0.1:8080/invocations `
  -H "Content-Type: application/json" `
  -d "{ \"message\": \"generate handbook\", \"creator_data_path\": \"data/creators.csv\", \"category\": \"Entertainment\", \"generate_handbook\": true, \"generate_pdf\": true }"
```

Open `http://127.0.0.1:8080/` for the web chat UI. Use `http://127.0.0.1:8080/health` only for health checks.

## Expected Outputs

- Top Creator Report
- Creator Performance Analysis
- Category Insight Report
- Content Growth Recommendation
- Creator Handbook by content group
- KAM Action Plan

## Deploy

After local validation, use `/agentbase-deploy` to build, push, and deploy this Custom Agent to AgentBase Runtime.
