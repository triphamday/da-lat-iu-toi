import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from starlette.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from greennode_agentbase import (
    GreenNodeAgentBaseApp,
    PingStatus,
    RequestContext,
)

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

app = GreenNodeAgentBaseApp()

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / os.environ.get("KNOWLEDGE_DIR", "knowledge")
DEFAULT_REPORT_DIR = PROJECT_ROOT / os.environ.get("REPORT_DIR", "reports")

REQUIRED_CREATOR_COLUMNS = [
    "creator_id",
    "creator_name",
    "category",
    "followers",
    "videos_this_month",
    "follower_growth_rate",
    "avg_views_per_video",
    "engagement_rate",
    "reject_rate",
    "tier",
]

RATE_COLUMNS = ["follower_growth_rate", "engagement_rate", "reject_rate"]

COLUMN_ALIASES = {
    "id": "creator_id",
    "creator": "creator_name",
    "name": "creator_name",
    "content_category": "category",
    "followers_count": "followers",
    "videos": "videos_this_month",
    "monthly_videos": "videos_this_month",
    "growth": "follower_growth_rate",
    "growth_rate": "follower_growth_rate",
    "views": "avg_views_per_video",
    "avg_views": "avg_views_per_video",
    "engagement": "engagement_rate",
    "reject": "reject_rate",
}

TIER_BONUS = {
    "diamond": 0.05,
    "platinum": 0.04,
    "gold": 0.03,
    "silver": 0.015,
    "bronze": 0.0,
}

TIER_RANK = {
    "Tier 1": 1,
    "Tier 2": 2,
    "Tier 3": 3,
    "Mass": 4,
}

AGENT_SYSTEM_PROMPT = """You are ZVideo Creator Agent, a Vietnamese KAM copilot for Zalo Video.

Your job:
- Answer KAM and creator questions through chat.
- Analyze uploaded Creator CSV/Excel data.
- Produce Top Creator Report, Creator Performance Analysis, Category Insight Report, Content Growth Recommendation, Creator Handbook, and KAM Action Plan.
- Answer platform policy questions using uploaded knowledge documents about content guidelines, reject content, and restrict content.

Rules:
- Answer in the same language the user uses. Default to clear Vietnamese when the language is mixed or unclear.
- Communicate warmly, naturally, and gently like a helpful teammate. Use "mình" and "bạn" by default in Vietnamese.
- Be concise first, then offer structured detail when the user asks for analysis or reports.
- If the user greets you or chats casually, respond conversationally before asking what they want to do.
- Use LangChain tools whenever the user asks about data, rankings, reports, handbook, action plan, reject/restrict, or policy.
- When the user asks for a short list such as "top 5 Creator", "list 5 creator", or "Creator nào tốt nhất", use `top_5_creator_list_tool` and return only the concise list/table.
- When the user asks about decline/risk/suy giảm, use `declining_creator_list_tool`.
- When the user asks for creators with videos_this_month/video count below 10, low posting, "số video <10", or outreach to low-output creators, use `low_video_creator_outreach_tool`.
- When the user asks about Creator tier classification, tier rules, "phân loại Tier", "xếp Tier", or how to assign Tier, use `tier_classification_tool`.
- When the user asks which category is effective, use `effective_category_list_tool` or `category_insight_report_tool` if they name a specific category.
- When the user asks which category has the most active creators, creator active, or "hoạt động tích cực", use `active_category_ranking_tool`.
- When the user asks who fits a campaign, use `campaign_creator_fit_tool`.
- When the user asks about daily promotion, monthly promotion, traffic push, or "đẩy traffic", use `daily_promotion_plan_tool`.
- For daily promotion or traffic-push answers, always group results by category, show category counts, and include full creator_id plus creator_name in the output.
- For active category answers, start with "Category - N Creator đang hoạt động tích cực", rank categories from highest to lowest, list creator_id and creator_name, and include KAM actions for the highest and lowest categories.
- If a data-driven answer needs creator data and no path is available, ask the user to upload CSV/Excel first.
- If a policy answer needs documents and no relevant passage is found, say the documents do not contain enough information.
- For policy/RAG answers, do not infer beyond the retrieved document excerpts and include citations from the tool output.
- For low-video outreach answers, filter strictly `videos_this_month < 10`, include creator_id, creator_name, category, a message under 50 words, predicted causes, Creator Insight, Category Insight, and next KAM action.
- For tier classification answers, use this rule: Tier 1 requires followers >100K and >=48 videos/month; Tier 2 requires followers from 50K to 100K and >=28 videos/month; Tier 3 requires followers from 10K to 50K and >=16 videos/month; Mass is <10K followers. If a Creator reaches the follower threshold but misses the video threshold, downgrade exactly one tier.
- Keep recommendations practical for the KAM team and tie them to metrics when available.
"""

_agent_cache = None


def chat_ui(_request):
    return FileResponse(PROJECT_ROOT / "web" / "index.html", media_type="text/html")


app.add_route("/", chat_ui, methods=["GET"])
app.mount("/assets", StaticFiles(directory=PROJECT_ROOT / "web" / "assets"), name="assets")


def unique_upload_path(target_dir: Path, filename: str) -> Path:
    target_path = target_dir / filename
    if not target_path.exists():
        return target_path

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for index in range(1, 1000):
        candidate = target_dir / f"{stem}-{timestamp}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    return target_dir / f"{stem}-{timestamp}{suffix}"


async def upload_file(request):
    form = await request.form()
    upload = form.get("file")
    kind = str(form.get("kind", "creator"))
    if upload is None or not getattr(upload, "filename", ""):
        return JSONResponse({"status": "error", "message": "No file uploaded."}, status_code=400)

    filename = Path(upload.filename).name
    suffix = Path(filename).suffix.lower()
    if kind == "knowledge":
        if suffix not in {".pdf", ".docx", ".txt", ".md"}:
            return JSONResponse(
                {"status": "error", "message": "Knowledge files must be PDF, DOCX, TXT, or MD."},
                status_code=400,
            )
        target_dir = DEFAULT_KNOWLEDGE_DIR
    else:
        if suffix not in {".csv", ".xlsx", ".xls"}:
            return JSONResponse(
                {"status": "error", "message": "Creator data must be CSV or Excel."},
                status_code=400,
            )
        target_dir = PROJECT_ROOT / "data"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = unique_upload_path(target_dir, filename)
    content = await upload.read()
    try:
        target_path.write_bytes(content)
    except PermissionError:
        retry_path = unique_upload_path(target_dir, f"{Path(filename).stem}-copy{Path(filename).suffix}")
        retry_path.write_bytes(content)
        target_path = retry_path
    except OSError as error:
        return JSONResponse(
            {"status": "error", "message": f"Could not save uploaded file: {error}"},
            status_code=500,
        )
    return JSONResponse(
        {
            "status": "success",
            "path": str(target_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "filename": target_path.name,
            "kind": kind,
        }
    )


app.add_route("/upload", upload_file, methods=["POST"])


def normalize_column(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_")


def as_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return without_marks.lower()


def casual_chat_reply(message: str) -> str:
    lowered = fold_text(message)
    if any(keyword in lowered for keyword in ["may tuoi", "bao nhieu tuoi", "tuoi cua ban"]):
        return (
            "Mình không có tuổi như con người đâu. Nếu tính theo vai trò thì mình vừa mới sinh ra để làm copilot cho team KAM Zalo Video: "
            "đọc policy, phân tích Creator CSV/Excel và gợi ý chiến lược nội dung dựa trên dữ liệu."
        )
    if any(keyword in lowered for keyword in ["de thuong", "cute", "dang yeu", "xinh qua", "thich ban"]):
        return (
            "Cảm ơn bạn nha. Mình sẽ nhận lời khen đó và đổi lại bằng một việc hữu ích: "
            "bạn có thể hỏi mình chọn Creator, phân tích category, kiểm tra policy reject/restrict hoặc tạo handbook."
        )
    if any(keyword in lowered for keyword in ["cam on", "thanks", "thank you", "tks"]):
        return "Không có gì, mình ở đây để hỗ trợ team KAM nhanh hơn và đỡ cực hơn."
    if any(keyword in lowered for keyword in ["ban la ai", "ban ten gi", "ten ban", "lam duoc gi", "giup duoc gi"]):
        return (
            "Mình là ZVideo Creator Agent. Mình có thể tra cứu guideline/reject/restrict từ tài liệu đã nạp sẵn, "
            "và khi có CSV/Excel Creator thì mình phân tích top Creator, risk suy giảm, category hiệu quả, campaign fit, handbook và action plan."
        )
    greeting_words = ["chao", "hello", "hi", "hey", "xin chao", "ban oi"]
    if any(word in lowered for word in greeting_words):
        return (
            "Chào bạn, mình đây. Bạn có thể hỏi policy Zalo Video ngay, hoặc attach file CSV/Excel để mình phân tích Creator theo dữ liệu."
        )
    return ""


def get_llm() -> ChatOpenAI:
    model = os.environ.get("LLM_MODEL", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    if not model or not base_url or not api_key:
        raise RuntimeError(
            "LLM_MODEL, LLM_BASE_URL, and LLM_API_KEY are required. Configure GreenNode AIP or another OpenAI-compatible provider."
        )
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key)


def get_agent():
    global _agent_cache
    if _agent_cache is None:
        _agent_cache = create_agent(
            model=get_llm(),
            tools=[
                analyze_creator_performance_tool,
                top_5_creator_list_tool,
                top_creator_report_tool,
                declining_creator_list_tool,
                low_video_creator_outreach_tool,
                tier_classification_tool,
                effective_category_list_tool,
                active_category_ranking_tool,
                campaign_creator_fit_tool,
                daily_promotion_plan_tool,
                category_insight_report_tool,
                content_growth_recommendation_tool,
                creator_handbook_tool,
                kam_action_plan_tool,
                answer_policy_question_tool,
            ],
            system_prompt=AGENT_SYSTEM_PROMPT,
            name="zvideo_creator_agent",
        )
    return _agent_cache


def read_creator_file(path_value: str) -> pd.DataFrame:
    path = as_project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Creator data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Creator data must be a CSV or Excel file.")


def clean_creator_data(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        normalized = normalize_column(column)
        renamed[column] = COLUMN_ALIASES.get(normalized, normalized)

    df = frame.rename(columns=renamed).copy()
    missing = [column for column in REQUIRED_CREATOR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required creator columns: {', '.join(missing)}")

    df = df[REQUIRED_CREATOR_COLUMNS].copy()
    df["creator_id"] = df["creator_id"].astype(str).str.strip()
    df["creator_name"] = df["creator_name"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().replace("", "unknown")
    df["tier"] = df["tier"].astype(str).str.strip().str.lower()

    for column in ["followers", "videos_this_month", "avg_views_per_video"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(lower=0)

    for column in RATE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
        if df[column].abs().max() > 1.5:
            df[column] = df[column] / 100

    df["reject_rate"] = df["reject_rate"].clip(lower=0)
    df = df.drop_duplicates(subset=["creator_id"], keep="last")
    return df


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if values.nunique(dropna=False) <= 1:
        ranked = pd.Series(0.5, index=values.index)
    else:
        ranked = values.rank(pct=True)
    return ranked if higher_is_better else 1 - ranked


def score_creators(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    base_score = (
        0.12 * percentile(scored["followers"])
        + 0.12 * percentile(scored["videos_this_month"])
        + 0.20 * percentile(scored["follower_growth_rate"])
        + 0.20 * percentile(scored["avg_views_per_video"])
        + 0.24 * percentile(scored["engagement_rate"])
        + 0.12 * percentile(scored["reject_rate"], higher_is_better=False)
    )
    tier_bonus = scored["tier"].map(TIER_BONUS).fillna(0)
    scored["performance_score"] = ((base_score + tier_bonus).clip(0, 1) * 100).round(2)
    scored = scored.sort_values(
        ["category", "performance_score", "engagement_rate"],
        ascending=[True, False, False],
    )
    scored["category_rank"] = (
        scored.groupby("category")["performance_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return scored


def frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def analyze_creator_performance(path_value: str) -> dict[str, Any]:
    raw = read_creator_file(path_value)
    cleaned = clean_creator_data(raw)
    scored = score_creators(cleaned)
    top_by_category = scored[scored["category_rank"] <= 5].copy()
    top_30 = top_by_category.sort_values("performance_score", ascending=False).head(30)

    return {
        "summary": {
            "source": str(as_project_path(path_value)),
            "total_creators": int(scored.shape[0]),
            "categories": int(scored["category"].nunique()),
            "selected_top_creators": int(top_30.shape[0]),
        },
        "scored": scored,
        "top_by_category": top_by_category,
        "top_30": top_30,
    }


def render_top_creator_report(analysis: dict[str, Any]) -> str:
    summary = analysis["summary"]
    top_30 = analysis["top_30"]
    lines = [
        "# Top Creator Report",
        "",
        f"- Nguồn dữ liệu: `{summary['source']}`",
        f"- Tổng Creator: {summary['total_creators']}",
        f"- Số category: {summary['categories']}",
        f"- Số Creator tiêu biểu được chọn: {summary['selected_top_creators']}",
        "",
        "| Rank | Creator ID | Creator name | Category | Tier | Score | Growth | Views/video | Engagement | Reject |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(frame_to_records(top_30), start=1):
        lines.append(
            "| {rank} | {creator_id} | {name} | {category} | {tier} | {score:.2f} | {growth:.2%} | {views:,.0f} | {engagement:.2%} | {reject:.2%} |".format(
                rank=index,
                creator_id=row["creator_id"],
                name=row["creator_name"],
                category=row["category"],
                tier=row["tier"],
                score=float(row["performance_score"]),
                growth=float(row["follower_growth_rate"]),
                views=float(row["avg_views_per_video"]),
                engagement=float(row["engagement_rate"]),
                reject=float(row["reject_rate"]),
            )
        )
    return "\n".join(lines)


def render_top_5_creator_list(analysis: dict[str, Any], sort_by: str = "performance_score", category: str = "") -> str:
    rows = analysis["scored"].copy()
    if category:
        rows = rows[rows["category"].str.lower() == category.lower()]
    if rows.empty:
        return f"Không có Creator phù hợp cho category `{category}`."

    sort_column = sort_by if sort_by in rows.columns else "performance_score"
    sort_columns = [sort_column] if sort_column == "performance_score" else [sort_column, "performance_score"]
    rows = rows.sort_values(sort_columns, ascending=[False] * len(sort_columns)).head(5)
    title = "Top 5 Creator"
    if sort_column == "follower_growth_rate":
        title = "Top 5 Creator tăng trưởng tốt nhất"
    elif sort_column == "avg_views_per_video":
        title = "Top 5 Creator theo views/video"
    elif sort_column == "engagement_rate":
        title = "Top 5 Creator theo engagement"
    if category:
        title = f"{title}: {category}"

    lines = [
        f"# {title}",
        "",
        "| Rank | Creator ID | Creator name | Category | Tier | Score | Growth | Views/video | Engagement | Reject |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(frame_to_records(rows), start=1):
        lines.append(
            "| {rank} | {creator_id} | {name} | {category} | {tier} | {score:.2f} | {growth:.2%} | {views:,.0f} | {engagement:.2%} | {reject:.2%} |".format(
                rank=index,
                creator_id=row["creator_id"],
                name=row["creator_name"],
                category=row["category"],
                tier=row["tier"],
                score=float(row["performance_score"]),
                growth=float(row["follower_growth_rate"]),
                views=float(row["avg_views_per_video"]),
                engagement=float(row["engagement_rate"]),
                reject=float(row["reject_rate"]),
            )
        )
    lines.extend(["", "## Cách chọn"])
    if sort_column == "follower_growth_rate":
        lines.append("- Ưu tiên growth cao, sau đó dùng performance score để tie-break nếu growth bằng nhau.")
    elif sort_column == "avg_views_per_video":
        lines.append("- Ưu tiên reach qua views/video, sau đó kiểm tra score và reject rate để giảm rủi ro.")
    elif sort_column == "engagement_rate":
        lines.append("- Ưu tiên engagement để chọn Creator có khả năng kích hoạt người xem.")
    else:
        lines.append("- Ưu tiên performance score tổng hợp, đồng thời nhìn views/video và reject rate để chọn Creator an toàn hơn.")

    lines.extend(["", "## Lý do shortlist"])
    for row in frame_to_records(rows):
        reason = f"score {float(row['performance_score']):.2f}, views/video {float(row['avg_views_per_video']):,.0f}, reject {float(row['reject_rate']):.2%}"
        if float(row["reject_rate"]) <= rows["reject_rate"].median():
            reason += ", rủi ro reject thấp so với nhóm top"
        lines.append(f"- {row['creator_id']} - {row['creator_name']}: {reason}.")
    return "\n".join(lines)


def resolve_category_from_message(message: str, analysis: dict[str, Any]) -> str:
    folded_message = fold_text(message)
    categories = sorted(
        [str(item) for item in analysis["scored"]["category"].dropna().unique()],
        key=len,
        reverse=True,
    )
    for category in categories:
        if fold_text(category) in folded_message:
            return category
    return ""


def is_tier_classification_question(lowered: str) -> bool:
    if "tier" not in lowered:
        return False
    return any(
        keyword in lowered
        for keyword in [
            "phan loai",
            "xep",
            "cach",
            "dieu kien",
            "quy dinh",
            "rule",
            "ranking",
            "rank",
            "creator",
            "follower",
            "follow",
        ]
    )


def tier_by_followers(followers: float) -> str:
    if followers > 100_000:
        return "Tier 1"
    if 50_000 <= followers <= 100_000:
        return "Tier 2"
    if 10_000 <= followers < 50_000:
        return "Tier 3"
    return "Mass"


def tier_video_requirement(tier: str) -> int:
    return {
        "Tier 1": 48,
        "Tier 2": 28,
        "Tier 3": 16,
    }.get(tier, 0)


def downgrade_one_tier(tier: str) -> str:
    return {
        "Tier 1": "Tier 2",
        "Tier 2": "Tier 3",
        "Tier 3": "Mass",
        "Mass": "Mass",
    }.get(tier, "Mass")


def classify_creator_tier(followers: float, videos_this_month: float) -> dict[str, Any]:
    follower_tier = tier_by_followers(followers)
    required_videos = tier_video_requirement(follower_tier)
    if follower_tier == "Mass":
        return {
            "follower_tier": follower_tier,
            "final_tier": "Mass",
            "required_videos": 0,
            "is_downgraded": False,
            "reason": "Followers <10K nên xếp Mass.",
        }
    if videos_this_month >= required_videos:
        return {
            "follower_tier": follower_tier,
            "final_tier": follower_tier,
            "required_videos": required_videos,
            "is_downgraded": False,
            "reason": f"Đủ followers và đạt {videos_this_month:.0f}/{required_videos} video/tháng để giữ {follower_tier}.",
        }
    final_tier = downgrade_one_tier(follower_tier)
    return {
        "follower_tier": follower_tier,
        "final_tier": final_tier,
        "required_videos": required_videos,
        "is_downgraded": True,
        "reason": (
            f"Đủ followers cho {follower_tier} nhưng chỉ có {videos_this_month:.0f}/{required_videos} "
            f"video/tháng nên hạ 1 tier xuống {final_tier}."
        ),
    }


def render_tier_classification(analysis: dict[str, Any] | None = None, limit: int = 30) -> str:
    lines = [
        "# Quy định phân loại Tier Creator",
        "",
        "Tier được xác định theo 2 nhóm điều kiện: **quy mô followers** và **sản lượng video trong tháng**. Followers là điều kiện nền để xác định tier ban đầu; video/tháng là điều kiện duy trì tier.",
        "",
        "## Điều kiện xếp Tier",
        "| Tier | Điều kiện followers | Điều kiện sản lượng | Kết luận |",
        "|---|---:|---:|---|",
        "| Tier 1 | >100,000 followers | >=48 video/tháng | Creator quy mô lớn, duy trì sản lượng cao. |",
        "| Tier 2 | 50,000-100,000 followers | >=28 video/tháng | Creator quy mô trung-cao, có nhịp đăng ổn định. |",
        "| Tier 3 | 10,000-50,000 followers | >=16 video/tháng | Creator đang phát triển, cần giữ nhịp đăng tối thiểu. |",
        "| Mass | <10,000 followers | Không áp dụng điều kiện video để lên tier | Creator nhóm đại trà/cần nuôi tăng trưởng. |",
        "",
        "## Nguyên tắc hạ Tier",
        "- Nếu Creator đạt điều kiện followers của một tier nhưng **không đạt điều kiện số video/tháng**, Creator sẽ được **hạ 1 tier** so với tier theo followers.",
        "- Ví dụ: Creator có 120K followers nhưng chỉ đăng 30 video/tháng thì không giữ Tier 1, mà được xếp Tier 2.",
        "- Ví dụ: Creator có 60K followers nhưng chỉ đăng 20 video/tháng thì không giữ Tier 2, mà được xếp Tier 3.",
        "- Ví dụ: Creator có 20K followers nhưng chỉ đăng 8 video/tháng thì không giữ Tier 3, mà được xếp Mass.",
    ]

    if analysis is None:
        return "\n".join(lines)

    rows = analysis["scored"].copy()
    classifications = rows.apply(
        lambda row: classify_creator_tier(float(row["followers"]), float(row["videos_this_month"])),
        axis=1,
    )
    rows["tier_by_followers"] = [item["follower_tier"] for item in classifications]
    rows["recommended_tier"] = [item["final_tier"] for item in classifications]
    rows["required_videos_for_tier"] = [item["required_videos"] for item in classifications]
    rows["is_tier_downgraded"] = [item["is_downgraded"] for item in classifications]
    rows["tier_reason"] = [item["reason"] for item in classifications]
    rows["tier_sort"] = rows["recommended_tier"].map(TIER_RANK).fillna(99).astype(int)

    summary = (
        rows.groupby("recommended_tier")
        .agg(
            creators=("creator_id", "count"),
            median_followers=("followers", "median"),
            median_videos=("videos_this_month", "median"),
            downgraded=("is_tier_downgraded", "sum"),
        )
        .reset_index()
    )
    summary["tier_sort"] = summary["recommended_tier"].map(TIER_RANK).fillna(99).astype(int)
    summary = summary.sort_values("tier_sort")

    downgraded = rows[rows["is_tier_downgraded"]].copy()
    downgraded = downgraded.sort_values(["tier_sort", "followers"], ascending=[True, False])
    if limit and limit > 0:
        downgraded_display = downgraded.head(limit)
    else:
        downgraded_display = downgraded

    lines.extend(
        [
            "",
            "## Áp dụng vào dữ liệu Creator hiện tại",
            "| Tier đề xuất | Số Creator | Median followers | Median videos/month | Creator bị hạ tier |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in frame_to_records(summary):
        lines.append(
            "| {tier} | {creators} | {followers:,.0f} | {videos:.1f} | {downgraded} |".format(
                tier=row["recommended_tier"],
                creators=int(row["creators"]),
                followers=float(row["median_followers"]),
                videos=float(row["median_videos"]),
                downgraded=int(row["downgraded"]),
            )
        )

    lines.extend(
        [
            "",
            f"## Creator bị hạ tier do chưa đạt sản lượng video ({int(downgraded.shape[0])} Creator)",
            "| Creator ID | Creator name | Category | Followers | Videos/month | Tier theo followers | Tier đề xuất | Lý do |",
            "|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in frame_to_records(downgraded_display):
        lines.append(
            "| {creator_id} | {creator_name} | {category} | {followers:,.0f} | {videos:.0f} | {follower_tier} | {recommended_tier} | {reason} |".format(
                creator_id=row["creator_id"],
                creator_name=row["creator_name"],
                category=row["category"],
                followers=float(row["followers"]),
                videos=float(row["videos_this_month"]),
                follower_tier=row["tier_by_followers"],
                recommended_tier=row["recommended_tier"],
                reason=row["tier_reason"],
            )
        )
    if limit and int(downgraded.shape[0]) > limit:
        lines.append(f"\nĐang hiển thị {limit} Creator đầu tiên cần rà soát. Có thể hỏi `liệt kê toàn bộ Creator bị hạ tier` nếu cần xem đầy đủ.")
    return "\n".join(lines)


def render_declining_creator_list(analysis: dict[str, Any], category: str = "", limit: int = 5) -> str:
    rows = analysis["scored"].copy()
    if category:
        rows = rows[rows["category"].str.lower() == category.lower()]
    if rows.empty:
        return f"Không có dữ liệu Creator phù hợp cho category `{category}`."

    rows["risk_score"] = (
        0.34 * percentile(rows["reject_rate"])
        + 0.28 * percentile(rows["follower_growth_rate"], higher_is_better=False)
        + 0.22 * percentile(rows["engagement_rate"], higher_is_better=False)
        + 0.16 * percentile(rows["avg_views_per_video"], higher_is_better=False)
    ) * 100
    rows = rows.sort_values(["risk_score", "reject_rate"], ascending=[False, False]).head(limit)
    title = "5 Creator có nguy cơ suy giảm"
    if category:
        title = f"{title}: {category}"

    lines = [
        f"# {title}",
        "",
        "| Rank | Creator | Category | Risk | Growth | Engagement | Reject | Views/video | KAM nên làm gì |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(frame_to_records(rows), start=1):
        action = "Audit 10 video gần nhất, giảm volume nếu reject tăng."
        if float(row["engagement_rate"]) <= rows["engagement_rate"].median():
            action = "Coaching lại hook/CTA và chọn lại format có tín hiệu tương tác."
        if float(row["reject_rate"]) > 0:
            action = "Ưu tiên review guideline/reject trước khi scale nội dung."
        lines.append(
            "| {rank} | {name} | {category} | {risk:.1f} | {growth:.2%} | {engagement:.2%} | {reject:.2%} | {views:,.0f} | {action} |".format(
                rank=index,
                name=row["creator_name"],
                category=row["category"],
                risk=float(row["risk_score"]),
                growth=float(row["follower_growth_rate"]),
                engagement=float(row["engagement_rate"]),
                reject=float(row["reject_rate"]),
                views=float(row["avg_views_per_video"]),
                action=action,
            )
        )
    return "\n".join(lines)


def render_low_video_creator_outreach(analysis: dict[str, Any], category: str = "", limit: int = 0) -> str:
    rows = analysis["scored"].copy()
    if category:
        rows = rows[rows["category"].str.lower() == category.lower()]
    if rows.empty:
        return f"Không có dữ liệu Creator phù hợp cho category `{category}`."

    category_bench = (
        rows.groupby("category")
        .agg(
            category_total=("creator_id", "count"),
            category_median_videos=("videos_this_month", "median"),
            category_median_score=("performance_score", "median"),
            category_median_views=("avg_views_per_video", "median"),
            category_median_engagement=("engagement_rate", "median"),
            category_median_growth=("follower_growth_rate", "median"),
            category_median_reject=("reject_rate", "median"),
        )
        .reset_index()
    )

    low_rows = rows[rows["videos_this_month"] < 10].copy()
    if low_rows.empty:
        return "Không có Creator nào có `videos_this_month < 10` trong dữ liệu hiện tại."

    low_rows = low_rows.merge(category_bench, on="category", how="left")
    low_rows["low_video_priority_score"] = (
        0.30 * percentile(low_rows["videos_this_month"], higher_is_better=False)
        + 0.20 * percentile(low_rows["performance_score"], higher_is_better=False)
        + 0.18 * percentile(low_rows["follower_growth_rate"], higher_is_better=False)
        + 0.17 * percentile(low_rows["engagement_rate"], higher_is_better=False)
        + 0.15 * percentile(low_rows["reject_rate"])
    ) * 100
    low_rows = low_rows.sort_values(
        ["low_video_priority_score", "videos_this_month", "performance_score"],
        ascending=[False, True, True],
    )
    total_low = int(low_rows.shape[0])
    if limit and limit > 0:
        low_rows = low_rows.head(limit)

    low_summary = (
        rows.assign(is_low_video=rows["videos_this_month"] < 10)
        .groupby("category")
        .agg(
            low_video_creators=("is_low_video", "sum"),
            total_creators=("creator_id", "count"),
            median_category_videos=("videos_this_month", "median"),
            median_category_score=("performance_score", "median"),
        )
        .reset_index()
    )
    low_summary["low_video_rate"] = low_summary["low_video_creators"] / low_summary["total_creators"].clip(lower=1)
    low_summary = low_summary.sort_values(
        ["low_video_creators", "low_video_rate", "median_category_score"],
        ascending=[False, False, False],
    )

    def outreach_message(row: dict[str, Any]) -> str:
        return (
            f"Chào {row['creator_name']}, team KAM thấy tháng này bạn mới có {float(row['videos_this_month']):.0f} video "
            f"ở nhóm {row['category']}. Mình muốn hiểu khó khăn hiện tại và hỗ trợ brief, format, lịch đăng phù hợp hơn cho bạn."
        )

    def predicted_causes(row: dict[str, Any]) -> str:
        causes = []
        videos = float(row["videos_this_month"])
        score = float(row["performance_score"])
        views = float(row["avg_views_per_video"])
        engagement = float(row["engagement_rate"])
        growth = float(row["follower_growth_rate"])
        reject = float(row["reject_rate"])
        category_score = float(row["category_median_score"])
        category_views = float(row["category_median_views"])
        category_engagement = float(row["category_median_engagement"])
        category_growth = float(row["category_median_growth"])

        if videos <= 3:
            causes.append("gần như mất nhịp sản xuất/backlog")
        else:
            causes.append("đăng không đều, chưa đủ tần suất để giữ đà đề xuất")
        if score >= category_score and views >= category_views:
            causes.append("chất lượng/reach còn ổn nên khả năng chính là vướng lịch hoặc capacity")
        if views < category_views:
            causes.append("reach thấp hơn benchmark category, có thể Creator giảm động lực đăng")
        if engagement < category_engagement:
            causes.append("tương tác thấp, hook/CTA/chủ đề có thể chưa kích hoạt người xem")
        if growth <= category_growth:
            causes.append("growth chưa vượt benchmark, dễ làm Creator mất đà")
        if reject > 0:
            causes.append("có rủi ro guideline/reject khiến Creator ngại scale")
        return "; ".join(causes[:4])

    def creator_insight(row: dict[str, Any]) -> str:
        insight = []
        if float(row["performance_score"]) >= float(row["category_median_score"]):
            insight.append("score vẫn ngang/cao hơn category")
        else:
            insight.append("score dưới benchmark category")
        if float(row["avg_views_per_video"]) >= float(row["category_median_views"]):
            insight.append("views/video còn có tiềm năng")
        else:
            insight.append("views/video yếu hơn nhóm")
        if float(row["videos_this_month"]) <= 3:
            insight.append("rủi ro mất nhịp rất cao")
        else:
            insight.append("cần khôi phục lịch đăng đều")
        return "; ".join(insight)

    def category_insight(row: dict[str, Any]) -> str:
        low_count = int(low_summary.loc[low_summary["category"] == row["category"], "low_video_creators"].iloc[0])
        total_count = int(row["category_total"])
        rate = low_count / max(total_count, 1)
        return (
            f"{row['category']} có {low_count}/{total_count} Creator dưới 10 video "
            f"({rate:.1%}); median category {float(row['category_median_videos']):.1f} videos/tháng."
        )

    def next_action(row: dict[str, Any]) -> str:
        videos = float(row["videos_this_month"])
        score = float(row["performance_score"])
        views = float(row["avg_views_per_video"])
        reject = float(row["reject_rate"])
        category_score = float(row["category_median_score"])
        category_views = float(row["category_median_views"])

        if reject > 0:
            return "Review guideline trước, pre-check 3 ý tưởng tiếp theo rồi mới tăng sản lượng."
        if videos <= 3:
            return "Book check-in 15 phút, hỏi rào cản, chốt mini sprint 3 video trong 7 ngày."
        if score >= category_score and views >= category_views:
            return "Ưu tiên tháo rào cản lịch đăng; giữ format hiện tại và đặt nhịp 2-3 video/tuần."
        if views < category_views:
            return "Audit 5 video gần nhất, viết lại hook/thumbnail, test 2 format dễ sản xuất."
        return "Gửi brief mẫu theo category, hẹn review sau 48h và theo dõi video mới đầu tiên."

    title = "Creator có videos_this_month < 10"
    if category:
        title = f"{title}: {category}"
    display_count = int(low_rows.shape[0])
    lines = [
        f"# {title}",
        "",
        f"Mình tìm thấy **{total_low} Creator** có `videos_this_month < 10`."
        + (f" Bảng dưới đang hiển thị {display_count} Creator ưu tiên can thiệp trước." if display_count != total_low else ""),
        "",
        "## Tóm tắt theo category",
        "| Category | Creator <10 video | Tổng Creator | Tỷ lệ | Median videos/month | Median score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame_to_records(low_summary):
        lines.append(
            "| {category} | {low} | {total} | {rate:.1%} | {videos:.1f} | {score:.2f} |".format(
                category=row["category"],
                low=int(row["low_video_creators"]),
                total=int(row["total_creators"]),
                rate=float(row["low_video_rate"]),
                videos=float(row["median_category_videos"]),
                score=float(row["median_category_score"]),
            )
        )

    lines.extend(
        [
            "",
            "## Danh sách Creator cần tiếp cận",
            "| Priority | Creator ID | Creator name | Category | Videos/month | Score | Views/video | Growth | Engagement | Reject | Tin nhắn gợi mở (<50 chữ) |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    records = frame_to_records(low_rows)
    for index, row in enumerate(records, start=1):
        lines.append(
            "| {rank} | {creator_id} | {creator_name} | {category} | {videos:.0f} | {score:.2f} | {views:,.0f} | {growth:.2%} | {engagement:.2%} | {reject:.2%} | {message} |".format(
                rank=index,
                creator_id=row["creator_id"],
                creator_name=row["creator_name"],
                category=row["category"],
                videos=float(row["videos_this_month"]),
                score=float(row["performance_score"]),
                views=float(row["avg_views_per_video"]),
                growth=float(row["follower_growth_rate"]),
                engagement=float(row["engagement_rate"]),
                reject=float(row["reject_rate"]),
                message=outreach_message(row),
            )
        )

    lines.extend(
        [
            "",
            "## Insight và hành động tiếp theo cho từng Creator",
            "| Creator ID | Creator name | Nguyên nhân dự đoán | Creator Insight | Category Insight | Hành động KAM cần làm |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in records:
        lines.append(
            "| {creator_id} | {creator_name} | {causes} | {creator_insight} | {category_insight} | {action} |".format(
                creator_id=row["creator_id"],
                creator_name=row["creator_name"],
                causes=predicted_causes(row),
                creator_insight=creator_insight(row),
                category_insight=category_insight(row),
                action=next_action(row),
            )
        )
    return "\n".join(lines)


def render_effective_category_list(analysis: dict[str, Any], limit: int = 6) -> str:
    rows = analysis["scored"].copy()
    summary = (
        rows.groupby("category")
        .agg(
            creators=("creator_id", "count"),
            median_score=("performance_score", "median"),
            avg_views=("avg_views_per_video", "mean"),
            avg_engagement=("engagement_rate", "mean"),
            avg_growth=("follower_growth_rate", "mean"),
            avg_reject=("reject_rate", "mean"),
        )
        .reset_index()
        .sort_values(["median_score", "avg_engagement", "avg_views"], ascending=[False, False, False])
        .head(limit)
    )
    lines = [
        "# Category đang hiệu quả theo dữ liệu",
        "",
        "| Rank | Category | Creators | Median score | Avg views/video | Avg engagement | Avg growth | Avg reject |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(frame_to_records(summary), start=1):
        lines.append(
            "| {rank} | {category} | {creators} | {score:.2f} | {views:,.0f} | {engagement:.2%} | {growth:.2%} | {reject:.2%} |".format(
                rank=index,
                category=row["category"],
                creators=int(row["creators"]),
                score=float(row["median_score"]),
                views=float(row["avg_views"]),
                engagement=float(row["avg_engagement"]),
                growth=float(row["avg_growth"]),
                reject=float(row["avg_reject"]),
            )
        )
    return "\n".join(lines)


def render_active_category_ranking(analysis: dict[str, Any], per_category: int = 5) -> str:
    rows = analysis["scored"].copy()
    per_category = max(1, int(per_category or 5))
    if rows.empty:
        return "Không có dữ liệu Creator để xếp hạng category active."

    medians = rows[
        ["videos_this_month", "performance_score", "avg_views_per_video", "engagement_rate", "reject_rate"]
    ].median()
    video_threshold = max(1.0, float(medians["videos_this_month"]))
    score_threshold = float(medians["performance_score"])
    reject_threshold = float(medians["reject_rate"])

    rows["active_score"] = (
        0.34 * percentile(rows["videos_this_month"])
        + 0.28 * (rows["performance_score"] / 100)
        + 0.16 * percentile(rows["avg_views_per_video"])
        + 0.12 * percentile(rows["engagement_rate"])
        + 0.10 * percentile(rows["reject_rate"], higher_is_better=False)
    ) * 100
    rows["is_active_creator"] = (
        (rows["videos_this_month"] >= video_threshold)
        & (rows["performance_score"] >= score_threshold)
        & (rows["reject_rate"] <= reject_threshold)
    )

    active_rows = rows[rows["is_active_creator"]].copy()
    if active_rows.empty:
        rows["is_active_creator"] = (rows["videos_this_month"] > 0) & (rows["performance_score"] >= score_threshold)
        active_rows = rows[rows["is_active_creator"]].copy()
    if active_rows.empty:
        return "Mình chưa tìm thấy Creator nào đủ điều kiện active/performance trong file hiện tại."

    active_rows = active_rows.sort_values(
        ["category", "active_score", "performance_score", "videos_this_month"],
        ascending=[True, False, False, False],
    )
    active_rows["active_rank_in_category"] = (
        active_rows.groupby("category")["active_score"].rank(method="first", ascending=False).astype(int)
    )

    totals = (
        rows.groupby("category")
        .agg(
            total_creators=("creator_id", "count"),
            total_videos=("videos_this_month", "sum"),
        )
        .reset_index()
    )
    active_summary = (
        active_rows.groupby("category")
        .agg(
            active_creators=("creator_id", "count"),
            median_videos=("videos_this_month", "median"),
            median_score=("performance_score", "median"),
            avg_views=("avg_views_per_video", "mean"),
            avg_reject=("reject_rate", "mean"),
            top_active_score=("active_score", "max"),
        )
        .reset_index()
    )
    summary = totals.merge(active_summary, on="category", how="left")
    fill_values = {
        "active_creators": 0,
        "median_videos": 0,
        "median_score": 0,
        "avg_views": 0,
        "avg_reject": 0,
        "top_active_score": 0,
    }
    summary = summary.fillna(fill_values)
    summary["active_creators"] = summary["active_creators"].astype(int)
    summary["active_rate"] = summary["active_creators"] / summary["total_creators"].clip(lower=1)
    summary = summary.sort_values(
        ["active_creators", "active_rate", "median_score", "avg_views"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    top_category = summary.iloc[0]
    lowest_category = summary.iloc[-1]
    top_name = str(top_category["category"])
    low_name = str(lowest_category["category"])
    top_count = int(top_category["active_creators"])

    lines = [
        f"# {top_name} - {top_count} Creator đang hoạt động tích cực",
        "",
        f"Category active nhất hiện là **{top_name}** với **{top_count} Creator** đạt điều kiện active/performance. Mình xếp hạng các category từ cao xuống thấp theo số Creator active.",
        "",
        "## Điều kiện tính Creator active",
        f"- Videos/month >= {video_threshold:.0f}",
        f"- Performance score >= {score_threshold:.2f}",
        f"- Reject rate <= {reject_threshold:.2%}",
        "- Trong nhóm đạt điều kiện, Creator được ưu tiên theo active score: sản lượng đăng, performance, views/video, engagement và reject thấp.",
        "",
        "## Ranking category theo số Creator active",
        "| Rank | Category | Creator active | Tổng Creator | Active rate | Median videos/month | Median score | Avg views/video | Avg reject |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(frame_to_records(summary), start=1):
        lines.append(
            "| {rank} | {category} | {active} | {total} | {rate:.1%} | {videos:.1f} | {score:.2f} | {views:,.0f} | {reject:.2%} |".format(
                rank=index,
                category=row["category"],
                active=int(row["active_creators"]),
                total=int(row["total_creators"]),
                rate=float(row["active_rate"]),
                videos=float(row["median_videos"]),
                score=float(row["median_score"]),
                views=float(row["avg_views"]),
                reject=float(row["avg_reject"]),
            )
        )

    lines.extend(
        [
            "",
            f"## Creator active nổi bật theo từng category (Top {per_category}/category)",
            "| Category | Rank trong category | Creator ID | Creator name | Videos/month | Score | Views/video | Engagement | Reject | Lý do đạt active |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    display_rows = active_rows[active_rows["active_rank_in_category"] <= per_category].copy()
    category_order = {str(row["category"]): index for index, row in enumerate(frame_to_records(summary), start=1)}
    display_rows["category_order"] = display_rows["category"].map(category_order).fillna(999).astype(int)
    display_rows = display_rows.sort_values(["category_order", "active_rank_in_category"])
    for row in frame_to_records(display_rows):
        reason = (
            f"videos {float(row['videos_this_month']):.0f} >= {video_threshold:.0f}; "
            f"score {float(row['performance_score']):.2f} >= {score_threshold:.2f}; "
            f"reject {float(row['reject_rate']):.2%} <= {reject_threshold:.2%}"
        )
        lines.append(
            "| {category} | {rank} | {creator_id} | {creator_name} | {videos:.0f} | {score:.2f} | {views:,.0f} | {engagement:.2%} | {reject:.2%} | {reason} |".format(
                category=row["category"],
                rank=int(row["active_rank_in_category"]),
                creator_id=row["creator_id"],
                creator_name=row["creator_name"],
                videos=float(row["videos_this_month"]),
                score=float(row["performance_score"]),
                views=float(row["avg_views_per_video"]),
                engagement=float(row["engagement_rate"]),
                reject=float(row["reject_rate"]),
                reason=reason,
            )
        )

    lines.extend(
        [
            "",
            "## Gợi ý cho team KAM",
            f"### Tăng thêm active cho category cao nhất: {top_name}",
            "- Giữ nhịp với nhóm top active bằng lịch check-in hằng tuần, nhắc lịch đăng và review sớm các video có khả năng bị reject.",
            "- Nhân rộng format của nhóm active score cao: lấy 3-5 format có views/video tốt, chuyển thành brief mẫu cho Creator cùng category.",
            "- Mở rộng từ P1 sang nhóm kế cận: chọn Creator có videos/month cao nhưng score sát ngưỡng để coaching hook, thumbnail và CTA.",
            f"### Kéo category thấp nhất: {low_name}",
            "- Tách nguyên nhân thấp: thiếu sản lượng, score yếu hay reject cao; mỗi nhóm cần một playbook coaching khác nhau.",
            "- Chạy thử cohort phục hồi 2 tuần: chọn 10-15 Creator gần đạt ngưỡng, giao brief đơn giản, theo dõi videos/month và reject mỗi 48h.",
            "- Dùng benchmark từ category cao nhất để đưa format dễ làm, tránh ép Creator yếu scale quá nhanh trước khi ổn guideline.",
        ]
    )
    return "\n".join(lines)


def render_campaign_creator_fit(analysis: dict[str, Any], category: str = "", limit: int = 5) -> str:
    rows = analysis["scored"].copy()
    if category:
        rows = rows[rows["category"].str.lower() == category.lower()]
    if rows.empty:
        return f"Không có Creator phù hợp cho category `{category}`."

    rows["campaign_fit_score"] = (
        0.42 * percentile(rows["performance_score"])
        + 0.24 * percentile(rows["avg_views_per_video"])
        + 0.20 * percentile(rows["engagement_rate"])
        + 0.14 * percentile(rows["reject_rate"], higher_is_better=False)
    ) * 100
    rows = rows.sort_values(["campaign_fit_score", "performance_score"], ascending=[False, False]).head(limit)
    title = "Creator phù hợp cho chiến dịch"
    if category:
        title = f"{title}: {category}"
    lines = [
        f"# {title}",
        "",
        "| Rank | Creator | Category | Fit | Tier | Score | Views/video | Engagement | Reject | Lý do fit |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(frame_to_records(rows), start=1):
        reason = "Hiệu suất tổng hợp tốt, reach ổn và reject thấp."
        if float(row["avg_views_per_video"]) >= rows["avg_views_per_video"].median():
            reason = "Reach tốt, phù hợp campaign cần mở rộng tiếp cận."
        if float(row["reject_rate"]) <= rows["reject_rate"].median():
            reason += " Brand-safety tương đối ổn theo dữ liệu reject."
        lines.append(
            "| {rank} | {name} | {category} | {fit:.1f} | {tier} | {score:.2f} | {views:,.0f} | {engagement:.2%} | {reject:.2%} | {reason} |".format(
                rank=index,
                name=row["creator_name"],
                category=row["category"],
                fit=float(row["campaign_fit_score"]),
                tier=row["tier"],
                score=float(row["performance_score"]),
                views=float(row["avg_views_per_video"]),
                engagement=float(row["engagement_rate"]),
                reject=float(row["reject_rate"]),
                reason=reason,
            )
        )
    return "\n".join(lines)


def render_daily_promotion_plan(analysis: dict[str, Any], category: str = "", per_category: int = 5) -> str:
    rows = analysis["scored"].copy()
    per_category = max(1, int(per_category or 5))
    if category:
        rows = rows[rows["category"].str.lower() == category.lower()]
    if rows.empty:
        return f"Không có dữ liệu Creator phù hợp cho category `{category}`."

    rows["daily_promotion_score"] = (
        0.42 * (rows["performance_score"] / 100)
        + 0.24 * percentile(rows["avg_views_per_video"])
        + 0.16 * percentile(rows["engagement_rate"])
        + 0.10 * percentile(rows["follower_growth_rate"])
        + 0.08 * percentile(rows["reject_rate"], higher_is_better=False)
    ) * 100
    rows = rows.sort_values(
        ["category", "daily_promotion_score", "performance_score", "avg_views_per_video"],
        ascending=[True, False, False, False],
    )
    rows["promotion_rank"] = rows.groupby("category")["daily_promotion_score"].rank(method="first", ascending=False).astype(int)
    selected = rows[rows["promotion_rank"] <= per_category].copy()
    selected = selected.sort_values(["category", "promotion_rank"])

    rows["is_high_performance_candidate"] = rows["performance_score"] >= rows.groupby("category")[
        "performance_score"
    ].transform("median")
    category_totals = (
        rows.groupby("category")
        .agg(
            total_creators=("creator_id", "count"),
            high_performance_pool=("is_high_performance_candidate", "sum"),
        )
        .reset_index()
    )

    category_summary = (
        selected.groupby("category")
        .agg(
            recommended_creators=("creator_id", "count"),
            p1_creators=("promotion_rank", lambda value: int((value <= 2).sum())),
            median_performance=("performance_score", "median"),
            avg_views=("avg_views_per_video", "mean"),
            avg_reject=("reject_rate", "mean"),
        )
        .reset_index()
        .merge(category_totals, on="category", how="left")
        .sort_values(["recommended_creators", "median_performance"], ascending=[False, False])
    )

    title = "Daily Promotion Plan tháng này"
    if category:
        title = f"{title}: {category}"

    category_count = int(selected["category"].nunique())
    creator_count = int(selected.shape[0])
    intro = (
        f"Mình đề xuất {creator_count} Creator để daily promotion tháng này, chia theo {category_count} category. "
        "Trong mỗi category, Creator có performance cao, views/video tốt và reject thấp sẽ được ưu tiên đẩy traffic trước."
    )

    lines = [
        f"# {title}",
        "",
        intro,
        "",
        "## Nguyên tắc chọn",
        f"- Mỗi category lấy tối đa {per_category} Creator có `daily_promotion_score` cao nhất.",
        "- `P1` là 2 Creator đầu mỗi category: nên ưu tiên slot traffic daily trước.",
        "- `P2` là nhóm rotation/dự phòng: dùng khi cần mở rộng slot hoặc thay thế P1 có rủi ro vận hành.",
        "- Score kết hợp performance tổng hợp, views/video, engagement, growth và reject rate thấp.",
        "- Bảng bên dưới luôn hiển thị đầy đủ `creator_id` và `creator_name` để KAM thao tác ngay.",
        "",
        "## Quota đề xuất theo category",
        "| Category | Tổng Creator | Pool performance cao | Số Creator đề xuất | P1 nên đẩy trước | Median performance | Avg views/video | Avg reject |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame_to_records(category_summary):
        lines.append(
            "| {category} | {total} | {pool} | {count} | {p1} | {performance:.2f} | {views:,.0f} | {reject:.2%} |".format(
                category=row["category"],
                total=int(row["total_creators"]),
                pool=int(row["high_performance_pool"]),
                count=int(row["recommended_creators"]),
                p1=int(row["p1_creators"]),
                performance=float(row["median_performance"]),
                views=float(row["avg_views"]),
                reject=float(row["avg_reject"]),
            )
        )

    lines.extend(
        [
            "",
            "## Danh sách Creator nên daily promotion",
            "| Category | Priority | Creator ID | Creator name | Score | Promotion score | Views/video | Growth | Engagement | Reject | Lý do ưu tiên đẩy traffic |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in frame_to_records(selected):
        rank = int(row["promotion_rank"])
        priority = "P1" if rank <= 2 else "P2"
        reason_parts = [
            f"rank {rank} trong category",
            f"performance {float(row['performance_score']):.2f}",
            f"views/video {float(row['avg_views_per_video']):,.0f}",
        ]
        if float(row["reject_rate"]) <= selected["reject_rate"].median():
            reason_parts.append("reject thấp")
        if float(row["follower_growth_rate"]) > 0:
            reason_parts.append(f"growth {float(row['follower_growth_rate']):.2%}")
        reason = "; ".join(reason_parts)
        lines.append(
            "| {category} | {priority} | {creator_id} | {creator_name} | {score:.2f} | {promo:.1f} | {views:,.0f} | {growth:.2%} | {engagement:.2%} | {reject:.2%} | {reason} |".format(
                category=row["category"],
                priority=priority,
                creator_id=row["creator_id"],
                creator_name=row["creator_name"],
                score=float(row["performance_score"]),
                promo=float(row["daily_promotion_score"]),
                views=float(row["avg_views_per_video"]),
                growth=float(row["follower_growth_rate"]),
                engagement=float(row["engagement_rate"]),
                reject=float(row["reject_rate"]),
                reason=reason,
            )
        )

    lines.extend(
        [
            "",
            "## Gợi ý triển khai cho KAM",
            "- Tuần 1: chạy daily promotion cho toàn bộ P1, theo dõi views/video và reject sau mỗi 24-48h.",
            "- Tuần 2: nếu P1 giữ reject thấp và views ổn, tiếp tục scale; nếu giảm hiệu quả, xoay sang P2 cùng category.",
            "- Khi cần chia slot traffic, dùng bảng quota category ở trên để giữ đủ độ phủ nội dung thay vì dồn vào một category.",
        ]
    )
    return "\n".join(lines)


def build_creator_insights(scored: pd.DataFrame, top_creators: pd.DataFrame, category: str = "") -> dict[str, Any]:
    cohort = top_creators.copy()
    if category:
        cohort = cohort[cohort["category"].str.lower() == category.lower()]
    if cohort.empty:
        return {
            "common_strengths": [],
            "effective_content_trends": [],
            "growth_formulas": [],
            "viewer_behaviors": [],
            "replicable_success_factors": [],
        }

    top_median = cohort[
        ["videos_this_month", "follower_growth_rate", "avg_views_per_video", "engagement_rate", "reject_rate"]
    ].median()
    all_median = scored[
        ["videos_this_month", "follower_growth_rate", "avg_views_per_video", "engagement_rate", "reject_rate"]
    ].median()

    common_strengths = []
    if top_median["engagement_rate"] >= all_median["engagement_rate"]:
        common_strengths.append("Engagement cao hơn mặt bằng chung, cho thấy nội dung có khả năng kích hoạt người xem.")
    if top_median["follower_growth_rate"] >= all_median["follower_growth_rate"]:
        common_strengths.append("Tốc độ tăng follower tốt, phù hợp làm benchmark tăng trưởng.")
    if top_median["reject_rate"] <= all_median["reject_rate"]:
        common_strengths.append("Tỷ lệ reject thấp hơn baseline, có thể mở rộng sản lượng an toàn hơn.")
    if top_median["videos_this_month"] >= all_median["videos_this_month"]:
        common_strengths.append("Tần suất đăng tải ổn định, tạo đủ dữ liệu để tối ưu format.")

    category_counts = cohort["category"].value_counts().head(8)
    effective_content_trends = [
        f"{category_name}: {count} Creator trong nhóm top, nên ưu tiên phân tích format nổi bật."
        for category_name, count in category_counts.items()
    ]

    growth_formulas = [
        "Tăng trưởng tốt nhất khi Creator kết hợp tần suất đăng đều, engagement cao và reject rate thấp.",
        "Nên scale format đã có views/video trên median trước khi mở rộng sang chủ đề mới.",
        "Creator có growth cao nhưng reject cao cần được KAM coaching về guideline trước khi đẩy sản lượng.",
    ]
    viewer_behaviors = [
        "Avg views/video cao phản ánh reach; engagement rate cao phản ánh mức độ phù hợp với nhu cầu người xem.",
        "Nếu views cao nhưng engagement thấp, nên tối ưu hook, CTA và độ rõ của lời hứa nội dung.",
        "Nếu engagement cao nhưng views thấp, nên giữ format và thử nghiệm packaging/title/thumbnail/khung giờ.",
    ]
    replicable_success_factors = [
        "Dùng benchmark theo category thay vì so sánh toàn bộ Creator bằng một ngưỡng chung.",
        "Ưu tiên nhân rộng format của Creator có performance_score cao và reject_rate thấp.",
        "Theo dõi weekly growth, engagement, views/video và reject_rate để quyết định scale hay điều chỉnh.",
    ]
    return {
        "common_strengths": common_strengths,
        "effective_content_trends": effective_content_trends,
        "growth_formulas": growth_formulas,
        "viewer_behaviors": viewer_behaviors,
        "replicable_success_factors": replicable_success_factors,
    }


def render_creator_performance_analysis(analysis: dict[str, Any]) -> str:
    scored = analysis["scored"]
    top_30 = analysis["top_30"]
    insights = build_creator_insights(scored, top_30)
    metric_summary = scored[
        ["followers", "videos_this_month", "follower_growth_rate", "avg_views_per_video", "engagement_rate", "reject_rate", "performance_score"]
    ].median()

    lines = [
        "# Creator Performance Analysis",
        "",
        "## Benchmark tổng quan",
        f"- Median followers: {metric_summary['followers']:,.0f}",
        f"- Median videos/month: {metric_summary['videos_this_month']:.1f}",
        f"- Median follower growth: {metric_summary['follower_growth_rate']:.2%}",
        f"- Median views/video: {metric_summary['avg_views_per_video']:,.0f}",
        f"- Median engagement: {metric_summary['engagement_rate']:.2%}",
        f"- Median reject rate: {metric_summary['reject_rate']:.2%}",
        f"- Median performance score: {metric_summary['performance_score']:.2f}",
        "",
        "## Điểm mạnh chung của Top Creator",
        *[f"- {item}" for item in insights["common_strengths"]],
        "",
        "## Công thức tăng trưởng nổi bật",
        *[f"- {item}" for item in insights["growth_formulas"]],
    ]
    return "\n".join(lines)


def render_category_insight_report(analysis: dict[str, Any], category: str = "") -> str:
    scored = analysis["scored"]
    top_by_category = analysis["top_by_category"]
    insights = build_creator_insights(scored, top_by_category, category)
    title = category or "Tất cả category"
    lines = [
        f"# Category Insight Report: {title}",
        "",
        "## Điểm mạnh chung",
        *[f"- {item}" for item in insights["common_strengths"]],
        "",
        "## Xu hướng nội dung hiệu quả",
        *[f"- {item}" for item in insights["effective_content_trends"]],
        "",
        "## Hành vi người xem đặc trưng",
        *[f"- {item}" for item in insights["viewer_behaviors"]],
        "",
        "## Yếu tố thành công có thể nhân rộng",
        *[f"- {item}" for item in insights["replicable_success_factors"]],
    ]
    return "\n".join(lines)


def select_category_rows(analysis: dict[str, Any], category: str = "") -> pd.DataFrame:
    rows = analysis["top_by_category"]
    if not category:
        return rows
    return rows[rows["category"].str.lower() == category.lower()]


def generate_content_growth_recommendation(analysis: dict[str, Any], category: str = "") -> str:
    rows = select_category_rows(analysis, category)
    if rows.empty:
        return f"Không có Creator top cho category `{category}`."

    avg_videos = rows["videos_this_month"].mean()
    avg_growth = rows["follower_growth_rate"].mean()
    avg_views = rows["avg_views_per_video"].mean()
    avg_engagement = rows["engagement_rate"].mean()
    avg_reject = rows["reject_rate"].mean()
    title = category or "Tất cả category"
    return f"""# Content Growth Recommendation: {title}

## Định hướng tăng trưởng
- Duy trì baseline khoảng {max(4, round(avg_videos))} video/tháng, tăng sản lượng khi reject rate được kiểm soát.
- Ưu tiên format có views/video trên {avg_views:,.0f} và engagement trên {avg_engagement:.2%}.
- Nếu growth hiện tại thấp hơn {avg_growth:.2%}, KAM nên coaching lại hook, chủ đề lặp lại và series format.

## Công thức để nhân rộng
1. Chọn 2 format có engagement cao nhất trong category.
2. Tạo 3 biến thể cho mỗi format trong 14 ngày.
3. Đo weekly: growth, views/video, engagement, reject_rate.
4. Scale format có engagement cao và reject rate dưới {max(avg_reject, 0.02):.2%}.

## Cảnh báo
- Không scale volume khi reject_rate tăng.
- Không copy format giữa category nếu audience intent khác nhau.
- Không tối ưu views mà bỏ qua engagement và follower growth.
"""


def generate_handbook(analysis: dict[str, Any], category: str = "") -> str:
    rows = select_category_rows(analysis, category)
    title = category or "Tất cả category"
    if rows.empty:
        return f"# Creator Handbook: {title}\n\nKhông có Top Creator trong category này."

    avg_videos = rows["videos_this_month"].mean()
    avg_growth = rows["follower_growth_rate"].mean()
    avg_views = rows["avg_views_per_video"].mean()
    avg_engagement = rows["engagement_rate"].mean()
    avg_reject = rows["reject_rate"].mean()
    top_names = ", ".join(rows["creator_name"].head(5).tolist())

    return f"""# Creator Handbook: {title}

## Tổng quan nhóm Creator
- Số Creator benchmark: {len(rows)}
- Creator tiêu biểu: {top_names}
- Benchmark: {avg_videos:.1f} video/tháng, {avg_views:,.0f} views/video, {avg_engagement:.2%} engagement.

## Chân dung Creator thành công
- Đăng nội dung đều, có format lặp lại để audience nhận diện.
- Có engagement và follower growth tốt, không chỉ có views cao.
- Giữ reject rate thấp trước khi tăng sản lượng.

## Chủ đề nội dung hiệu quả
- Ưu tiên chủ đề đã tạo engagement cao trong category.
- Biến câu hỏi/lực cản của người xem thành series nội dung.
- Chọn chủ đề có thể lặp lại hàng tuần, không phụ thuộc vào một video viral.

## Format video khuyến nghị
- Hook rõ trong 3 giây đầu.
- Mỗi video chỉ nên có một lời hứa nội dung.
- Dùng series format: before/after, checklist, case study, myth/fact, reaction có giá trị.

## Tần suất đăng tải
- Mức nên bắt đầu: {max(4, round(avg_videos))} video/tháng.
- Chỉ tăng volume khi reject rate dưới {max(avg_reject, 0.02):.2%}.

## KPI tham khảo
- Follower growth: >= {avg_growth:.2%}
- Views/video: >= {avg_views:,.0f}
- Engagement rate: >= {avg_engagement:.2%}
- Reject rate: <= {max(avg_reject, 0.02):.2%}

## Action Plan 30 ngày
- Tuần 1: Audit 10 video gần nhất, chọn 2 format tốt nhất.
- Tuần 2: Sản xuất 3 biến thể cho mỗi format.
- Tuần 3: Scale format thắng, dừng chủ đề kém tín hiệu.
- Tuần 4: Review KPI và chốt playbook tháng tiếp theo cho Creator.
"""


def generate_kam_action_plan(analysis: dict[str, Any], category: str = "") -> str:
    rows = select_category_rows(analysis, category)
    title = category or "Tất cả category"
    if rows.empty:
        return f"Không có dữ liệu top creator cho `{title}`."

    high_risk = rows.sort_values(["reject_rate", "follower_growth_rate"], ascending=[False, True]).head(5)
    high_growth = rows.sort_values("follower_growth_rate", ascending=False).head(5)
    return "\n".join(
        [
            f"# Action Plan cho KAM: {title}",
            "",
            "## Creator nên ưu tiên scale",
            *[
                f"- {row.creator_name} ({row.category}): growth {row.follower_growth_rate:.2%}, score {row.performance_score:.2f}"
                for row in high_growth.itertuples()
            ],
            "",
            "## Creator cần coaching/risk control",
            *[
                f"- {row.creator_name} ({row.category}): reject {row.reject_rate:.2%}, engagement {row.engagement_rate:.2%}"
                for row in high_risk.itertuples()
            ],
            "",
            "## Việc KAM nên làm trong 30 ngày",
            "- Chọn top Creator có growth cao và reject thấp để làm benchmark.",
            "- Tạo playbook format theo category từ các Creator top.",
            "- Coaching riêng nhóm reject cao trước khi đẩy sản lượng.",
            "- Review KPI hằng tuần: growth, views/video, engagement, reject_rate.",
        ]
    )


def write_pdf(markdown_text: str, output_name: str) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_REPORT_DIR / output_name
    styles = getSampleStyleSheet()
    story = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 8))
        elif stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], styles["Title"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], styles["Heading2"]))
        elif stripped.startswith("- "):
            story.append(Paragraph(f"- {stripped[2:]}", styles["BodyText"]))
        else:
            story.append(Paragraph(stripped, styles["BodyText"]))
    SimpleDocTemplate(str(output_path), pagesize=A4).build(story)
    return str(output_path)


def tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"\w+", fold_text(value), flags=re.UNICODE) if len(token) > 2]


def chunk_text(text: str, source: str, size: int = 1400, overlap: int = 180) -> list[dict[str, str]]:
    normalized = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    index = 1
    while start < len(normalized):
        end = min(start + size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append({"source": source, "chunk_id": f"{source}#{index}", "text": chunk})
        index += 1
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def extract_documents(paths: list[str]) -> list[dict[str, str]]:
    all_paths: list[Path] = []
    for value in paths:
        path = as_project_path(value)
        if path.is_dir():
            all_paths.extend(
                sorted(
                    item
                    for item in path.rglob("*")
                    if item.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
                )
            )
        elif path.exists():
            all_paths.append(path)

    documents = []
    for path in all_paths:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                documents.extend(chunk_text(text, f"{path.name} p.{page_number}"))
        elif suffix == ".docx":
            from docx import Document

            document = Document(str(path))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            documents.extend(chunk_text(text, path.name))
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            documents.extend(chunk_text(text, path.name))
    return documents


def retrieve_relevant_chunks(question: str, documents: list[dict[str, str]], limit: int = 5) -> list[dict[str, Any]]:
    query_tokens = Counter(tokenize(question))
    if not query_tokens:
        return []

    ranked = []
    for document in documents:
        doc_tokens = Counter(tokenize(document["text"]))
        lexical_score = sum(min(count, doc_tokens[token]) for token, count in query_tokens.items())
        if lexical_score > 0:
            ranked.append({**document, "score": lexical_score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def compact_excerpt(text: str, source: str = "", limit: int = 620) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    source_title = re.sub(r"\.pdf.*$", "", source, flags=re.IGNORECASE).strip()
    if source_title and fold_text(cleaned).startswith(fold_text(source_title)):
        cleaned = cleaned[len(source_title) :].strip(" :-–—")
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rsplit(" ", 1)[0].strip() + "..."
    cleaned = re.sub(r"\s*Mục tiêu:\s*", "\n**Mục tiêu**\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([1-9])\.\s+", r"\n\1. ", cleaned)
    return cleaned.strip()


def format_policy_sources(chunks: list[dict[str, Any]], llm_unavailable: bool = False) -> str:
    opening = (
        "LLM hiện chưa khả dụng hoặc API key chưa được authorize. "
        "Mình chỉ hiển thị các đoạn khớp nhất từ knowledge base và không suy diễn ngoài tài liệu."
        if llm_unavailable
        else "Mình tìm thấy các đoạn tài liệu liên quan dưới đây."
    )
    lines = [
        "# Policy Reference",
        "",
        f"> {opening}",
        "",
        "## Nguồn trích dẫn",
    ]
    for index, chunk in enumerate(chunks, start=1):
        lines.extend(
            [
                "",
                f"### [{index}] {chunk['source']}",
                "**Trích đoạn liên quan**",
                *compact_excerpt(chunk["text"], chunk["source"]).splitlines(),
            ]
        )
    return "\n".join(lines)


def answer_from_knowledge(question: str, paths: list[str]) -> str:
    documents = extract_documents(paths)
    if not documents:
        return "Chưa có tài liệu knowledge. Hãy đặt PDF/DOCX/TXT/MD vào thư mục `knowledge/`."

    chunks = retrieve_relevant_chunks(question, documents)
    if not chunks:
        return "Mình chưa tìm thấy đoạn tài liệu liên quan. Mình sẽ không suy diễn ngoài tài liệu."

    context = "\n\n".join(
        f"[{index}] Source: {chunk['source']}\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )
    prompt = (
        "Trả lời rõ ràng bằng cùng ngôn ngữ với câu hỏi; nếu câu hỏi trộn ngôn ngữ thì ưu tiên tiếng Việt. "
        "Chỉ sử dụng các excerpt bên dưới. Luôn cite nguồn bằng [1], [2]. "
        "Nếu excerpt không đủ thông tin, hãy nói là tài liệu không đủ thông tin.\n\n"
        f"Question: {question}\n\nExcerpts:\n{context}"
    )

    try:
        return str(get_llm().invoke(prompt).content)
    except Exception:
        return format_policy_sources(chunks, llm_unavailable=True)


@tool
def analyze_creator_performance_tool(creator_data_path: str) -> str:
    """Create a Creator Performance Analysis from a Creator CSV/Excel file."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_creator_performance_analysis(analysis)


@tool
def top_5_creator_list_tool(creator_data_path: str, sort_by: str = "performance_score", category: str = "") -> str:
    """Return a concise Top 5 Creator list. Use sort_by=performance_score, follower_growth_rate, avg_views_per_video, or engagement_rate."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_top_5_creator_list(analysis, sort_by=sort_by, category=category)


@tool
def top_creator_report_tool(creator_data_path: str) -> str:
    """Create Top Creator Report: Top 5 Creator per category and Top 30 representative creators."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_top_creator_report(analysis)


@tool
def declining_creator_list_tool(creator_data_path: str, category: str = "") -> str:
    """Return creators with decline/risk signals based on growth, engagement, reject rate, and views."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_declining_creator_list(analysis, category=category)


@tool
def low_video_creator_outreach_tool(creator_data_path: str, category: str = "", limit: int = 0) -> str:
    """Return creators with videos_this_month < 10, outreach messages, predicted causes, insights, and next KAM actions."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_low_video_creator_outreach(analysis, category=category, limit=limit)


@tool
def tier_classification_tool(creator_data_path: str = "", limit: int = 30) -> str:
    """Return the professional Creator tier classification rule and optionally apply it to Creator data."""
    if not creator_data_path:
        return render_tier_classification()
    analysis = analyze_creator_performance(creator_data_path)
    return render_tier_classification(analysis, limit=limit)


@tool
def effective_category_list_tool(creator_data_path: str) -> str:
    """Return effective categories ranked by creator performance metrics."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_effective_category_list(analysis)


@tool
def active_category_ranking_tool(creator_data_path: str, per_category: int = 5) -> str:
    """Rank categories by number of active performance-qualified creators and list creator_id plus creator_name."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_active_category_ranking(analysis, per_category=per_category)


@tool
def campaign_creator_fit_tool(creator_data_path: str, category: str = "") -> str:
    """Return creators suitable for a campaign based on performance, reach, engagement, and reject rate."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_campaign_creator_fit(analysis, category=category)


@tool
def daily_promotion_plan_tool(creator_data_path: str, category: str = "", per_category: int = 5) -> str:
    """Return a monthly daily-promotion recommendation plan by category with creator_id, creator_name, counts, and traffic priority."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_daily_promotion_plan(analysis, category=category, per_category=per_category)


@tool
def category_insight_report_tool(creator_data_path: str, category: str = "") -> str:
    """Create Category Insight Report with strengths, trends, growth formulas, viewer behavior, and repeatable success factors."""
    analysis = analyze_creator_performance(creator_data_path)
    return render_category_insight_report(analysis, category)


@tool
def content_growth_recommendation_tool(creator_data_path: str, category: str = "") -> str:
    """Create data-backed content growth recommendations for all categories or a specific category."""
    analysis = analyze_creator_performance(creator_data_path)
    return generate_content_growth_recommendation(analysis, category)


@tool
def creator_handbook_tool(creator_data_path: str, category: str = "", generate_pdf: bool = False) -> str:
    """Generate a Creator Handbook for all categories or a specific content category. Can optionally export PDF."""
    analysis = analyze_creator_performance(creator_data_path)
    handbook = generate_handbook(analysis, category)
    if generate_pdf:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", (category or "all-categories").lower()).strip("-")
        pdf_path = write_pdf(handbook, f"zvideo-handbook-{slug}.pdf")
        return f"{handbook}\n\nPDF exported: `{pdf_path}`"
    return handbook


@tool
def kam_action_plan_tool(creator_data_path: str, category: str = "") -> str:
    """Create an action plan for KAM based on Creator data and optional category."""
    analysis = analyze_creator_performance(creator_data_path)
    return generate_kam_action_plan(analysis, category)


@tool
def answer_policy_question_tool(question: str, knowledge_paths: str = "knowledge") -> str:
    """Answer policy/guideline/reject/restrict questions using uploaded knowledge documents with citations."""
    paths = [item.strip() for item in knowledge_paths.split(",") if item.strip()]
    return answer_from_knowledge(question, paths or ["knowledge"])


def fallback_without_llm(message: str, payload: dict[str, Any]) -> str:
    data_path = payload.get("creator_data_path") or payload.get("data_path") or payload.get("file_path")
    knowledge_paths = payload.get("knowledge_paths") or ["knowledge"]
    if isinstance(knowledge_paths, list):
        knowledge_paths_value = ",".join(str(path) for path in knowledge_paths)
    else:
        knowledge_paths_value = str(knowledge_paths)

    lowered = fold_text(message)
    casual_reply = casual_chat_reply(message)
    if casual_reply:
        return casual_reply

    greeting_words = ["chao", "hello", "hi", "hey", "xin chao"]
    if any(word in lowered for word in greeting_words) and not data_path:
        return (
            "Chào bạn, mình đây. Mình có thể giúp team KAM tra cứu guideline/reject/restrict từ knowledge base đã nạp sẵn, "
            "hoặc phân tích Creator khi bạn attach file CSV/Excel. Bạn muốn bắt đầu bằng câu hỏi policy hay phân tích dữ liệu Creator?"
        )

    if any(keyword in lowered for keyword in ["reject", "restrict", "chinh sach", "huong dan"]):
        return answer_policy_question_tool.invoke({"question": message, "knowledge_paths": knowledge_paths_value})

    tier_question = is_tier_classification_question(lowered)
    if tier_question and not data_path:
        return render_tier_classification()

    if not data_path:
        return (
            "Mình chưa thấy file Creator CSV/Excel đi kèm, nên chưa thể phân tích số liệu Creator cho bạn. "
            "Bạn có thể attach file ở ô chat bên dưới; còn nếu muốn hỏi về guideline, reject hoặc restrict thì cứ hỏi luôn, mình sẽ tra trong knowledge base."
        )

    analysis = analyze_creator_performance(str(data_path))
    category = resolve_category_from_message(message, analysis)

    if tier_question:
        show_all = any(keyword in lowered for keyword in ["toan bo", "tat ca", "full", "all"])
        return render_tier_classification(analysis, limit=0 if show_all else 30)

    low_video_patterns = [
        r"video[s_ ]*(this_month|thang nay)?\s*<\s*10",
        r"video[s_ ]*(this_month|thang nay)?\s*(duoi|nho hon|it hon)\s*10",
        r"(duoi|nho hon|it hon)\s*10\s*video",
    ]
    asks_low_video_outreach = (
        any(re.search(pattern, lowered) for pattern in low_video_patterns)
        or any(
            keyword in lowered
            for keyword in [
                "it video",
                "low video",
                "low posting",
                "dang it",
                "san luong thap",
                "it dang",
                "duoi 10 video",
            ]
        )
    ) and any(keyword in lowered for keyword in ["creator", "creato", "video", "posting", "dang"])
    if asks_low_video_outreach:
        return render_low_video_creator_outreach(analysis, category=category)

    asks_active_category = (
        any(keyword in lowered for keyword in ["active", "hoat dong", "tich cuc", "dang hoat dong"])
        and any(keyword in lowered for keyword in ["category", "nhom noi dung", "creator"])
    )
    asks_active_volume = any(
        keyword in lowered
        for keyword in [
            "so luong creator active",
            "nhieu creator active",
            "creator active nhat",
            "active nhat",
            "hoat dong tich cuc nhat",
            "hoat dong nhieu nhat",
        ]
    )
    if asks_active_category or asks_active_volume:
        return render_active_category_ranking(analysis)

    daily_promotion_keywords = [
        "daily",
        "promotion",
        "promo",
        "promote",
        "traffic",
        "push",
        "quang ba",
        "day traffic",
        "day view",
        "day slot",
        "slot promotion",
    ]
    if any(keyword in lowered for keyword in daily_promotion_keywords):
        return render_daily_promotion_plan(analysis, category=category)

    if any(keyword in lowered for keyword in ["suy giam", "nguy co", "risk", "decline", "tut", "yeu di", "can coaching"]):
        return render_declining_creator_list(analysis, category=category)

    if any(keyword in lowered for keyword in ["chien dich", "campaign", "phu hop", "booking", "brand", "nhan hang"]):
        return render_campaign_creator_fit(analysis, category=category)

    if "handbook" in lowered or "playbook" in lowered:
        return generate_handbook(analysis, category)
    if "action plan" in lowered or "ke hoach" in lowered:
        return generate_kam_action_plan(analysis, category)

    if any(keyword in lowered for keyword in ["category", "nhom noi dung", "nhóm nội dung"]):
        if any(keyword in lowered for keyword in ["hieu qua", "hiệu quả", "tot", "tốt", "best"]):
            if category:
                return render_category_insight_report(analysis, category)
            return render_effective_category_list(analysis)
        return render_category_insight_report(analysis, category)

    if "insight" in lowered or "xu huong" in lowered or "trend" in lowered:
        return render_category_insight_report(analysis, category)

    asks_for_creator_list = any(keyword in lowered for keyword in ["creator", "creato"]) and (
        re.search(r"\b(5|five)\b", lowered)
        or any(keyword in lowered for keyword in ["list", "danh sach", "top", "ranking", "xep hang"])
        or (any(keyword in lowered for keyword in ["growth", "tang truong"]) and any(keyword in lowered for keyword in ["nao", "tot nhat"]))
    )
    if asks_for_creator_list:
        sort_by = "performance_score"
        if "growth" in lowered or "tang truong" in lowered:
            sort_by = "follower_growth_rate"
        elif "view" in lowered:
            sort_by = "avg_views_per_video"
        elif "engagement" in lowered or "tuong tac" in lowered:
            sort_by = "engagement_rate"
        return render_top_5_creator_list(analysis, sort_by=sort_by, category=category)
    if "growth" in lowered or "tang truong" in lowered or "khuyen nghi" in lowered:
        return generate_content_growth_recommendation(analysis, category)
    if "top" in lowered or "ranking" in lowered or "xep hang" in lowered:
        return render_top_5_creator_list(analysis, category=category)
    return render_creator_performance_analysis(analysis)


def extract_agent_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        return str(last_message.content)
    content = getattr(last_message, "content", "")
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def build_user_message(message: str, payload: dict[str, Any]) -> str:
    data_path = payload.get("creator_data_path") or payload.get("data_path") or payload.get("file_path")
    knowledge_paths = payload.get("knowledge_paths") or ["knowledge"]
    context_lines = []
    if data_path:
        context_lines.append(f"creator_data_path: {data_path}")
    if knowledge_paths:
        context_lines.append(f"knowledge_paths: {knowledge_paths}")
    if payload.get("category"):
        context_lines.append(f"category: {payload.get('category')}")
    context = "\n".join(context_lines)
    if context:
        return f"{message}\n\nContext:\n{context}"
    return message


@app.entrypoint
def handler(payload: dict, context: RequestContext) -> dict:
    message = payload.get("message", "")
    try:
        try:
            agent = get_agent()
            result = agent.invoke({"messages": [{"role": "user", "content": build_user_message(message, payload)}]})
            response = extract_agent_text(result)
            mode = "langchain_agent"
        except Exception as exc:
            response = fallback_without_llm(message, payload)
            mode = f"deterministic_fallback: {type(exc).__name__}"

        return {
            "status": "success",
            "mode": mode,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "session_id": context.session_id,
        }
    except Exception as exc:
        return {
            "status": "error",
            "response": str(exc),
            "timestamp": datetime.now().isoformat(),
            "session_id": context.session_id,
        }


@app.ping
def health_check() -> PingStatus:
    return PingStatus.HEALTHY


if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")
