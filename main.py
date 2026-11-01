"""
CrewAI Multi-Agent Backend — FastAPI server
============================================
All 10 agent endpoints are faithful ports of the standalone scripts in d:\\standalone.

  /agent1  - Blog Post Crew              (Crew, Sequential, 3 agents)
  /agent2  - Competitor Analysis Crew    (Crew, Hierarchical, 4 agents + scraper)
  /agent3  - Meeting Notes Crew          (Crew, Parallel async tasks, 4 agents)
  /agent4  - Data Report Crew            (Crew, Structured Pydantic output, 3 agents)
  /agent5  - Customer Support Flow       (Flow, Router + or_)
  /agent6  - Resume Screening Flow       (Flow, Router + or_ + run_crew helper)
  /agent7  - Bug Triage Flow             (Flow, Router + or_ + run_crew helper)
  /agent8  - Content Repurposing Flow    (Flow, and_ parallel + run_crew helper)
  /agent9  - Lead Qualification Flow     (Flow, Router + or_ + run_crew helper)
  /agent10 - Deep Research Report Flow   (Flow with research/writing/revision crews)
"""

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP — must run BEFORE any crewai import on Windows
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import tempfile

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _tmp = os.path.join(tempfile.gettempdir(), "crewai_localappdata")
    os.makedirs(_tmp, exist_ok=True)
    os.environ["LOCALAPPDATA"] = _tmp

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import csv
import datetime
import io
import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# FLOW STATE MODELS  (module-level — required by CrewAI Flow generic typing)
# ─────────────────────────────────────────────────────────────────────────────

class SupportTicketState(BaseModel):
    """Agent 5 — Customer Support Flow state."""
    ticket_id: str = ""
    customer_name: str = ""
    issue_description: str = ""
    category: str = ""
    priority: str = "medium"
    resolution: str = ""
    satisfaction_score: int = 0


class ResumeState(BaseModel):
    """Agent 6 — Resume Screening Flow state."""
    resume_path: str = ""
    resume_text: str = ""
    candidate: str = ""
    score: int = 0
    decision: str = ""
    output_text: str = ""


class IssueState(BaseModel):
    """Agent 7 — Bug Triage Flow state."""
    title: str = ""
    body: str = ""
    issue_type: str = ""
    reply: str = ""
    label: str = ""


class RepurposeState(BaseModel):
    """Agent 8 — Content Repurposing Flow state."""
    source: str = ""
    article: str = ""
    key_points: str = ""
    linkedin: str = ""
    twitter: str = ""
    newsletter: str = ""


class LeadState(BaseModel):
    """Agent 9 — Lead Qualification Flow state."""
    name: str = ""
    website: str = ""
    profile: str = ""
    score: str = ""
    action: str = ""
    email: str = ""


class ReportState(BaseModel):
    """Agent 10 — Deep Research Report Flow state."""
    topic: str = ""
    audience: str = ""
    research_notes: str = ""
    draft: str = ""
    feedback: str = ""
    final_report: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP & REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CrewAI Multi-Agent Backend",
    description=(
        "10 production-ready AI agent endpoints powered by CrewAI + Gemini.\n\n"
        "Agents 1-4 are sequential/hierarchical/parallel Crews. "
        "Agents 5-10 are Flows with routers, or_, and_ and nested crews."
    ),
    version="3.0.0",
)


class AgentRequest(BaseModel):
    prompt: str = Field(..., description="Main input for the agent")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Extra data: transcript, csv_data, resume_text, website_url, audience, etc."
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Execution parameters: model, temperature, max_tokens"
    )


class AgentResponse(BaseModel):
    agent_id: str
    status: str
    input_received: AgentRequest
    result: str
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("CREWAI_MODEL", "gemini/gemini-2.5-flash")


def get_llm(temperature: float = 0.3, parameters: Optional[Dict[str, Any]] = None):
    """Build a CrewAI LLM object from request parameters (matching standalone get_llm pattern)."""
    from crewai import LLM
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable first.")
    p = parameters or {}
    return LLM(
        model=p.get("model", MODEL),
        api_key=key,
        temperature=float(p.get("temperature", temperature)),
        max_tokens=int(p.get("max_tokens", 2000)),
    )


def duckduckgo_search_fn(query: str) -> str:
    """DuckDuckGo search with graceful failure — mirrors standalone helper."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as exc:
        return f"Search failed: {exc}"
    if not results:
        return "No results found."
    return "\n\n".join(f"{r.get('title')}\n{r.get('href')}\n{r.get('body')}" for r in results)


def run_crew_helper(role, goal, backstory, description, expected_output,
                    tools=None, inputs=None, temperature=0.3) -> str:
    """
    Build a tiny Crew (1 Agent + 1 Task) and run it.
    Mirrors the standalone run_crew() helper used in projects 6-9.
    """
    from crewai import Agent, Task, Crew
    llm = get_llm(temperature=temperature)
    agent = Agent(role=role, goal=goal, backstory=backstory, llm=llm,
                  tools=tools or [], verbose=False, allow_delegation=False, max_iter=6)
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return crew.kickoff(inputs=inputs or {}).raw


def write_if_missing(path: str, text: str) -> None:
    """Create a sample file the first time only — mirrors standalone helper."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


async def _run_in_thread(fn: Callable, *args, **kwargs) -> str:
    """Run a blocking CrewAI function in a thread pool so FastAPI stays async."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def _execute(agent_num: int, runner: Callable, request: AgentRequest) -> AgentResponse:
    """Generic async wrapper: calls runner in thread, returns AgentResponse."""
    try:
        result_text = await _run_in_thread(
            runner,
            request.prompt,
            request.context,
            request.parameters,
        )
        return AgentResponse(
            agent_id=f"agent{agent_num}",
            status="success",
            input_received=request,
            result=str(result_text),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent {agent_num} execution failed: {exc}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "CrewAI Multi-Agent API is running.",
        "agents": {
            "/agent1":  "Blog Post Crew (Sequential: Researcher → Writer → Editor)",
            "/agent2":  "Competitor Analysis Crew (Hierarchical: Scraper/Pricing/News → Strategist)",
            "/agent3":  "Meeting Notes Crew (Parallel async: Summarizer | Action | Decision → Formatter)",
            "/agent4":  "Data Report Crew (Structured Pydantic output: Analyst → Insight → Reviewer)",
            "/agent5":  "Customer Support Flow (Router: billing/account_access/technical_issue/feature_request/other)",
            "/agent6":  "Resume Screening Flow (Router: shortlist → summarize | reject → email → save_results)",
            "/agent7":  "Bug Triage Flow (Router: bug → triage | feature → story | question → answer)",
            "/agent8":  "Content Repurposing Flow (and_: extract_points → LinkedIn + Twitter + Newsletter)",
            "/agent9":  "Lead Qualification Flow (Router: hot → outreach | warm → nurture | cold → archive)",
            "/agent10": "Deep Research Report Flow (research crew → write crew → revision crew)",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — Blog Post Crew  (Sequential: Researcher → Writer → Editor)
# Standalone: project1_blog_post_crew.py
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent1(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import FileWriterTool
    from crewai.tools import tool

    @tool("DuckDuckGoSearch")
    def duckduckgo_search(query: str) -> str:
        """Search the web with DuckDuckGo and return the top results (title, link, snippet)."""
        return duckduckgo_search_fn(query)

    topic = prompt.strip() or "AI in farming"
    llm = get_llm(temperature=0.3, parameters=parameters)
    file_writer = FileWriterTool()

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Find accurate, recent facts about {topic}",
        backstory="You are a careful researcher who always checks facts and keeps the source links.",
        llm=llm,
        tools=[duckduckgo_search],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    writer = Agent(
        role="Blog Writer",
        goal="Write an engaging 600-word blog post about {topic}",
        backstory="You turn research notes into clear, friendly articles that a beginner can follow.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    editor = Agent(
        role="Blog Editor",
        goal="Fix tone, grammar and factual mistakes, then save the final post",
        backstory="You are a strict editor. You never add facts that are not in the research.",
        llm=llm,
        tools=[file_writer],
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Search the web for '{topic}'. Collect 6-8 key facts, real examples and "
            "statistics. Write each fact as a bullet and keep its source link."
        ),
        expected_output="A bullet list of 6-8 facts with source links.",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Using the research notes, write a blog post of about 600 words on '{topic}'. "
            "Use a catchy title, an introduction, 3 sections with sub-headings and a conclusion."
        ),
        expected_output="A 600-word blog post in Markdown.",
        agent=writer,
    )

    edit_task = Task(
        description=(
            "Edit the blog post: fix grammar, make the tone friendly and professional, and "
            "remove any claim that is not supported by the research. Then save it with the "
            "File Writer Tool using filename 'blog_post.md', directory 'output' and overwrite=True. "
            "After saving, reply with the full final post."
        ),
        expected_output="The final edited blog post (and the file output/blog_post.md is saved).",
        agent=editor,
    )

    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, write_task, edit_task],
        process=Process.sequential,
        verbose=True,
    )

    return str(crew.kickoff(inputs={"topic": topic}).raw).strip()


@app.post("/agent1", response_model=AgentResponse, tags=["Agents"])
async def agent1_endpoint(request: AgentRequest):
    """Blog Post Crew — research + write + edit a 600-word blog post.

    **prompt**: the blog topic (e.g. "AI in farming").
    """
    return await _execute(1, _run_agent1, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — Competitor Analysis Crew  (Hierarchical process)
# Standalone: project2_competitor_analysis_crew.py
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent2(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import ScrapeWebsiteTool, FileWriterTool
    from crewai.tools import tool

    @tool("DuckDuckGoSearch")
    def duckduckgo_search(query: str) -> str:
        """Search the web with DuckDuckGo and return the top results (title, link, snippet)."""
        return duckduckgo_search_fn(query)

    competitors = prompt.strip() or "https://www.notion.so, https://www.coda.io, https://clickup.com"
    llm = get_llm(temperature=0.3, parameters=parameters)

    site_scraper = Agent(
        role="Website Scraper",
        goal="Read each competitor website and summarise what the company offers",
        backstory="You visit websites and extract the product list and key messages.",
        llm=llm,
        tools=[ScrapeWebsiteTool()],
        verbose=True,
        allow_delegation=False,
    )

    pricing_analyst = Agent(
        role="Pricing Analyst",
        goal="Find the plans and prices of each competitor",
        backstory="You read pricing pages and compare plans, limits and free tiers.",
        llm=llm,
        tools=[ScrapeWebsiteTool()],
        verbose=True,
        allow_delegation=False,
    )

    news_analyst = Agent(
        role="News Analyst",
        goal="Find the latest news, launches and announcements of each competitor",
        backstory="You track company news and notice what changed recently.",
        llm=llm,
        tools=[duckduckgo_search],
        verbose=True,
        allow_delegation=False,
    )

    strategist = Agent(
        role="Strategy Consultant",
        goal="Turn the research into a clear SWOT table and recommendations",
        backstory="You compare companies and give practical advice in simple language.",
        llm=llm,
        tools=[FileWriterTool()],
        verbose=True,
        allow_delegation=False,
    )

    scrape_task = Task(
        description="Visit these competitor websites: {competitors}. For each, summarise what the company does and its main products.",
        expected_output="One short paragraph per competitor.",
    )

    pricing_task = Task(
        description="For these competitors: {competitors}, find the pricing plans (names, prices, limits). If no price is public, say so.",
        expected_output="A table of plans and prices per competitor.",
    )

    news_task = Task(
        description="Search the latest news (last 3 months) for the companies behind these sites: {competitors}.",
        expected_output="3 bullet points of recent news per competitor, with links.",
    )

    strategy_task = Task(
        description=(
            "Using all the research, build a SWOT table (Strengths, Weaknesses, Opportunities, Threats) "
            "for each competitor and finish with 3 recommendations for us. "
            "Save it using the File Writer Tool: filename 'report.md', directory 'output', overwrite=True. "
            "Then reply with the full report."
        ),
        expected_output="A Markdown report with SWOT tables and 3 recommendations.",
        context=[scrape_task, pricing_task, news_task],
    )

    crew = Crew(
        agents=[site_scraper, pricing_analyst, news_analyst, strategist],
        tasks=[scrape_task, pricing_task, news_task, strategy_task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
    )

    return str(crew.kickoff(inputs={"competitors": competitors}).raw).strip()


@app.post("/agent2", response_model=AgentResponse, tags=["Agents"])
async def agent2_endpoint(request: AgentRequest):
    """Competitor Analysis Crew — hierarchical SWOT analysis with web scraping.

    **prompt**: competitor URLs or names, comma-separated (e.g. "https://notion.so, https://clickup.com").
    """
    return await _execute(2, _run_agent2, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — Meeting Notes Crew  (Parallel async tasks)
# Standalone: project3_meeting_notes_crew.py
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent3(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import FileReadTool, FileWriterTool

    llm = get_llm(temperature=0.3, parameters=parameters)

    # Write the transcript to a temp file so FileReadTool can read it
    default_transcript = (
        "Priya: Thanks everyone. Today we decide the launch plan for the mobile app.\n"
        "Arjun: Testing is 80% done. I can finish the remaining Android tests by Friday.\n"
        "Priya: Good. Meera, can you prepare the launch email by next Tuesday?\n"
        "Meera: Yes, I will draft it. Also, we should offer a 20% discount for the first week.\n"
        "Priya: Agreed, we will go with the 20% discount. And we will launch on the 15th, not the 10th.\n"
        "Arjun: One risk - the payment page is slow. I will ask Ravi to fix it before launch.\n"
        "Priya: Okay. Ravi owns the payment fix, due next Wednesday. Meeting closed."
    )
    # Use prompt as transcript if long enough, else check context, else use default
    transcript_text = (
        prompt if len(prompt.strip()) > 30
        else (context or {}).get("transcript", default_transcript)
    )

    # Write transcript to a temp file for FileReadTool
    transcript_path = os.path.join(tempfile.gettempdir(), "agent3_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    read_tool = FileReadTool(file_path=transcript_path)

    summarizer = Agent(
        role="Meeting Summarizer", goal="Summarise the meeting in 5 lines",
        backstory="You capture the main points without losing meaning.",
        llm=llm, tools=[read_tool], allow_delegation=False, verbose=True,
    )

    action_finder = Agent(
        role="Action Item Finder", goal="Find every action item with its owner and due date",
        backstory="You listen for who promised to do what, and by when.",
        llm=llm, tools=[read_tool], allow_delegation=False, verbose=True,
    )

    decision_logger = Agent(
        role="Decision Logger", goal="List every decision that was made",
        backstory="You record decisions clearly so nobody argues about them later.",
        llm=llm, tools=[read_tool], allow_delegation=False, verbose=True,
    )

    formatter = Agent(
        role="Notes Formatter", goal="Combine everything into one tidy notes document",
        backstory="You format meeting notes so they are easy to scan.",
        llm=llm, tools=[FileWriterTool()], allow_delegation=False, verbose=True,
    )

    summary_task = Task(
        description=f"Read the transcript file and write a 5-line summary of the meeting.",
        expected_output="Exactly 5 lines.",
        agent=summarizer,
        async_execution=True,
    )

    action_task = Task(
        description="Read the transcript file and list every action item as: Task - Owner - Due date.",
        expected_output="A bullet list of 'Task - Owner - Due date'.",
        agent=action_finder,
        async_execution=True,
    )

    decision_task = Task(
        description="Read the transcript file and list every decision that was made.",
        expected_output="A bullet list of decisions.",
        agent=decision_logger,
        async_execution=True,
    )

    format_task = Task(
        description=(
            "Combine the summary, action items and decisions into one Markdown document with the headings "
            "'Summary', 'Action Items' and 'Decisions'. Save it with the File Writer Tool: "
            "filename 'notes.md', directory 'output', overwrite=True. Then reply with the document."
        ),
        expected_output="The complete meeting notes in Markdown.",
        agent=formatter,
        context=[summary_task, action_task, decision_task],
    )

    crew = Crew(
        agents=[summarizer, action_finder, decision_logger, formatter],
        tasks=[summary_task, action_task, decision_task, format_task],
        process=Process.sequential,
        verbose=True,
    )

    return str(crew.kickoff().raw).strip()


@app.post("/agent3", response_model=AgentResponse, tags=["Agents"])
async def agent3_endpoint(request: AgentRequest):
    """Meeting Notes Crew — parallel async agents extract summary, actions & decisions.

    **prompt**: meeting transcript text. Or pass via `context.transcript`.
    """
    return await _execute(3, _run_agent3, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — Data Report Crew  (Structured Pydantic output)
# Standalone: project4_data_report_crew.py
# ─────────────────────────────────────────────────────────────────────────────

# Module-level SalesStats so it can be referenced in _run_agent4
class SalesStats(BaseModel):
    """The exact shape the Data Analyst must return."""
    total_revenue: float = Field(description="Sum of units * price")
    total_units: int = Field(description="Sum of units sold")
    top_product: str = Field(description="Product with the highest revenue")
    top_region: str = Field(description="Region with the highest revenue")
    revenue_by_product: List[str] = Field(description="Lines like 'Laptop: 12800'")


def _run_agent4(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
    from crewai_tools import FileReadTool, FileWriterTool

    llm = get_llm(temperature=0.1, parameters=parameters)

    default_csv = """\
date,product,region,units,price
2025-01-06,Laptop,North,5,800
2025-01-07,Phone,South,12,400
2025-01-08,Tablet,East,7,300
2025-01-09,Laptop,South,3,800
2025-01-10,Phone,North,9,400
2025-01-11,Headphones,East,20,50
2025-01-12,Phone,East,6,400
2025-01-13,Laptop,East,4,800
2025-01-14,Tablet,North,10,300
2025-01-15,Headphones,South,15,50
"""
    csv_content = (context or {}).get("csv_data", default_csv)

    # Write to a temp file so the tool (and FileReadTool) can open it
    sales_path = os.path.join(tempfile.gettempdir(), "agent4_sales.csv")
    with open(sales_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    @tool("SalesCSVAnalyzer")
    def analyze_sales_csv(file_path: str) -> str:
        """Read a sales CSV (date, product, region, units, price) and return exact totals computed with pandas."""
        df = pd.read_csv(file_path)
        df["revenue"] = df["units"] * df["price"]
        by_product = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
        by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
        return (
            f"Total revenue: {df['revenue'].sum()}\n"
            f"Total units: {int(df['units'].sum())}\n"
            f"Revenue by product:\n{by_product.to_string()}\n"
            f"Revenue by region:\n{by_region.to_string()}"
        )

    analyst = Agent(
        role="Data Analyst", goal="Compute exact statistics from the sales file",
        backstory="You never guess numbers - you always use the analysis tool.",
        llm=llm, tools=[FileReadTool(file_path=sales_path), analyze_sales_csv],
        allow_delegation=False, verbose=True,
    )

    insight_writer = Agent(
        role="Insight Writer", goal="Explain the top 3 findings in simple words",
        backstory="You turn numbers into insights a manager can act on.",
        llm=llm, allow_delegation=False, verbose=True,
    )

    reviewer = Agent(
        role="Report Reviewer", goal="Check every number and publish the final report",
        backstory="You compare each number in the report with the statistics and fix errors.",
        llm=llm, tools=[FileWriterTool()], allow_delegation=False, verbose=True,
    )

    stats_task = Task(
        description=f"Use the SalesCSVAnalyzer tool on the file '{sales_path}' and return the statistics.",
        expected_output="Sales statistics matching the SalesStats model.",
        agent=analyst,
        output_pydantic=SalesStats,
    )

    insight_task = Task(
        description="Using the statistics, write the 3 most important findings. Each finding: one headline and one sentence with a number.",
        expected_output="3 numbered findings.",
        agent=insight_writer,
        context=[stats_task],
    )

    review_task = Task(
        description=(
            "Build a short weekly sales report (title, key numbers table, the 3 findings). "
            "Check EVERY number against the statistics and correct any mistake. Save it with the "
            "File Writer Tool: filename 'report.md', directory 'output', overwrite=True. Reply with the report."
        ),
        expected_output="A verified Markdown sales report.",
        agent=reviewer,
        context=[stats_task, insight_task],
    )

    crew = Crew(
        agents=[analyst, insight_writer, reviewer],
        tasks=[stats_task, insight_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    # Try to extract structured stats from task 1
    try:
        stats: SalesStats = result.tasks_output[0].pydantic
        header = f"**Top product:** {stats.top_product} | **Total revenue:** {stats.total_revenue}\n\n"
    except Exception:
        header = ""
    return header + str(result.raw).strip()


@app.post("/agent4", response_model=AgentResponse, tags=["Agents"])
async def agent4_endpoint(request: AgentRequest):
    """Data Report Crew — pandas-powered analysis with Pydantic structured output.

    Pass CSV content as a string via `context.csv_data`. Default sample data used otherwise.
    **prompt**: optional description of what to analyze.
    """
    return await _execute(4, _run_agent4, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — Customer Support Flow  (Flow, Router + or_)
# Standalone: project5_customer_support_flow.py
# ─────────────────────────────────────────────────────────────────────────────
_SUPPORT_CATEGORIES = ["billing", "account_access", "technical_issue", "feature_request", "other"]


def _run_agent5(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai.flow.flow import Flow, listen, start, router, or_

    llm = get_llm(temperature=0, parameters=parameters)

    ticket_id = (context or {}).get("ticket_id", "TCK-1001")
    customer_name = (context or {}).get("customer_name", "Valued Customer")
    issue_description = prompt.strip() or "I was charged twice for my subscription this month."

    class CustomerSupportFlow(Flow[SupportTicketState]):
        @start()
        def receive_ticket(self):
            self.state.ticket_id = ticket_id
            self.state.customer_name = customer_name
            self.state.issue_description = issue_description

        @listen(receive_ticket)
        def categorize_ticket(self):
            _prompt = (
                f"Classify this support ticket into exactly one of: {', '.join(_SUPPORT_CATEGORIES)}.\n"
                f"Ticket: {self.state.issue_description}\n"
                "Reply with only the category word."
            )
            answer = llm.call(_prompt).strip().lower()
            self.state.category = answer if answer in _SUPPORT_CATEGORIES else "other"

        @router(categorize_ticket)
        def route_by_category(self):
            return self.state.category

        @listen("billing")
        def handle_billing_issue(self):
            self.state.priority = "high"

        @listen("account_access")
        def handle_access_issue(self):
            self.state.priority = "high"

        @listen("technical_issue")
        def handle_technical_issue(self):
            pass  # priority stays medium

        @listen("feature_request")
        def handle_feature_request(self):
            self.state.priority = "low"

        @listen("other")
        def handle_other_issue(self):
            pass  # priority stays medium

        @listen(or_(handle_billing_issue, handle_access_issue, handle_technical_issue,
                    handle_feature_request, handle_other_issue))
        def resolve_ticket(self):
            _prompt = (
                f"Write a short, polite reply (3 sentences) to {self.state.customer_name}.\n"
                f"Issue: {self.state.issue_description}\n"
                f"Category: {self.state.category}. Priority: {self.state.priority}."
            )
            self.state.resolution = llm.call(_prompt)
            return self.state.resolution

    flow = CustomerSupportFlow()
    flow.kickoff()
    return (
        f"**Ticket ID:** {flow.state.ticket_id}\n"
        f"**Category:** {flow.state.category}\n"
        f"**Priority:** {flow.state.priority.upper()}\n\n"
        f"**Resolution:**\n{flow.state.resolution}"
    )


@app.post("/agent5", response_model=AgentResponse, tags=["Agents"])
async def agent5_endpoint(request: AgentRequest):
    """Customer Support Flow — classifies ticket (5 categories), sets priority, drafts reply.

    **prompt**: the support ticket text.
    Optional context: `ticket_id`, `customer_name`.
    """
    return await _execute(5, _run_agent5, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 6 — Resume Screening Flow  (Flow, Router + or_ + run_crew helper)
# Standalone: project6_resume_screening_flow.py
# ─────────────────────────────────────────────────────────────────────────────
_JOB_DESCRIPTION = (
    "Junior Python Developer.\n"
    "Required: Python, SQL, Git, basic REST APIs. Nice to have: Docker, cloud (AWS/GCP)."
)


def _run_agent6(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai.flow.flow import Flow, listen, start, router, or_

    llm = get_llm(temperature=0, parameters=parameters)

    # Resume text comes from prompt (if long enough) or context
    default_resume = (
        "Ravi Kumar. 2 years Python developer. "
        "Built REST APIs with Flask, SQL databases, Git daily, Docker basics, deployed on AWS."
    )
    resume_text = (
        prompt if len(prompt.strip()) > 30
        else (context or {}).get("resume_text", default_resume)
    )
    candidate_name = (context or {}).get("candidate_name", "Candidate")
    job_desc = (context or {}).get("job_description", _JOB_DESCRIPTION)

    class ResumeScreeningFlow(Flow[ResumeState]):
        @start()
        def load_resumes(self):
            self.state.resume_text = resume_text
            self.state.candidate = candidate_name

        @listen(load_resumes)
        def score_resume(self):
            _prompt = (
                f"Job description:\n{job_desc}\n\nResume:\n{self.state.resume_text}\n\n"
                "Score how well this resume fits the job from 0 to 10. Reply with ONLY the number."
            )
            reply = llm.call(_prompt)
            match = re.search(r"\d+(\.\d+)?", reply)
            self.state.score = min(10, int(float(match.group()))) if match else 0

        @router(score_resume)
        def check_score(self):
            self.state.decision = "shortlist" if self.state.score >= 7 else "reject"
            return self.state.decision

        @listen("shortlist")
        def summarize_candidate(self):
            self.state.output_text = run_crew_helper(
                role="Recruiter",
                goal="Summarise a candidate's strengths and gaps for a hiring manager",
                backstory="You are an experienced technical recruiter.",
                description="Job: {job}\n\nResume: {resume}\n\nWrite 3 strengths and 2 gaps as bullet points.",
                expected_output="3 strengths and 2 gaps.",
                inputs={"job": job_desc, "resume": self.state.resume_text},
            )

        @listen("reject")
        def draft_rejection(self):
            self.state.output_text = llm.call(
                f"Write a short, kind rejection email (4 lines) to {self.state.candidate} for the "
                "Junior Python Developer role. Do not give false hope."
            )

        @listen(or_(summarize_candidate, draft_rejection))
        def save_results(self):
            # Append to results.csv in working directory
            results_path = os.path.join(os.getcwd(), "results.csv")
            new_file = not os.path.exists(results_path)
            with open(results_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(["candidate", "score", "decision", "summary_or_email"])
                writer.writerow([
                    self.state.candidate, self.state.score,
                    self.state.decision,
                    self.state.output_text.replace("\n", " "),
                ])
            return self.state.decision

    flow = ResumeScreeningFlow()
    flow.kickoff()
    return (
        f"**Candidate:** {flow.state.candidate}\n"
        f"**Score:** {flow.state.score}/10\n"
        f"**Decision:** {flow.state.decision.upper()}\n\n"
        f"**Evaluation / Message:**\n{flow.state.output_text}"
    )


@app.post("/agent6", response_model=AgentResponse, tags=["Agents"])
async def agent6_endpoint(request: AgentRequest):
    """Resume Screening Flow — scores resume 0-10, shortlists or rejects, saves to results.csv.

    **prompt**: the resume text (or a short description).
    Optional context: `candidate_name`, `job_description`, `resume_text`.
    """
    return await _execute(6, _run_agent6, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 7 — Bug Triage Flow  (Flow, Router + or_ + run_crew helper)
# Standalone: project7_bug_triage_flow.py
# ─────────────────────────────────────────────────────────────────────────────
_DOCS_URL = "https://docs.crewai.com/"


def _run_agent7(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai.flow.flow import Flow, listen, start, router, or_
    from crewai_tools import ScrapeWebsiteTool

    llm = get_llm(temperature=0, parameters=parameters)

    title = prompt.strip() or "App crashes when I upload a PNG"
    body = (context or {}).get("body",
        "Clicking Upload with a PNG larger than 5MB closes the app. Happens every time on Windows 11.")

    class BugTriageFlow(Flow[IssueState]):
        @start()
        def read_issue(self):
            self.state.title = title
            self.state.body = body

        @listen(read_issue)
        def classify_issue(self):
            _prompt = (
                "Classify this issue as exactly one word: bug, feature or question.\n"
                f"Title: {self.state.title}\nBody: {self.state.body}"
            )
            answer = llm.call(_prompt).strip().lower()
            self.state.issue_type = (
                answer if answer in ("bug", "feature", "question") else "question"
            )

        @router(classify_issue)
        def route_type(self):
            return self.state.issue_type

        @listen("bug")
        def bug_crew(self):
            self.state.reply = run_crew_helper(
                role="QA Engineer",
                goal="Judge bug severity and write clear reproduction steps",
                backstory="You triage bugs for a software team.",
                description=(
                    "Bug title: {title}\nDetails: {body}\n\n"
                    "Give: Severity (low/medium/high/critical) with a one-line reason, "
                    "then numbered steps to reproduce."
                ),
                expected_output="Severity line and numbered steps.",
                inputs={"title": self.state.title, "body": self.state.body},
            )

        @listen("feature")
        def feature_crew(self):
            self.state.reply = run_crew_helper(
                role="Product Manager",
                goal="Turn feature ideas into user stories",
                backstory="You write crisp user stories.",
                description=(
                    "Feature request: {title}\nDetails: {body}\n\n"
                    "Write: 'As a <user>, I want <goal>, so that <benefit>' plus 3 acceptance criteria."
                ),
                expected_output="One user story and 3 acceptance criteria.",
                inputs={"title": self.state.title, "body": self.state.body},
            )

        @listen("question")
        def answer_crew(self):
            self.state.reply = run_crew_helper(
                role="Support Engineer",
                goal="Answer questions using the official documentation",
                backstory="You read the docs and answer politely with a link.",
                description=(
                    "Question: {title} - {body}\n"
                    "Open the docs at {docs} with the scrape tool and answer in under 120 words with the most relevant link."
                ),
                expected_output="A short helpful answer with a link.",
                tools=[ScrapeWebsiteTool()],
                inputs={"title": self.state.title, "body": self.state.body, "docs": _DOCS_URL},
            )

        @listen(or_(bug_crew, feature_crew, answer_crew))
        def post_reply(self):
            self.state.label = {
                "bug": "type:bug", "feature": "type:feature", "question": "type:question"
            }.get(self.state.issue_type, "type:other")
            return self.state.reply

    flow = BugTriageFlow()
    flow.kickoff()
    return (
        f"**Issue Type:** {flow.state.issue_type.upper()}\n"
        f"**Label:** {flow.state.label}\n\n"
        f"**Triage Result:**\n{flow.state.reply}"
    )


@app.post("/agent7", response_model=AgentResponse, tags=["Agents"])
async def agent7_endpoint(request: AgentRequest):
    """Bug Triage Flow — classifies issue (bug/feature/question), runs specialist crew, posts reply.

    **prompt**: issue title.
    Optional context: `body` (issue description).
    """
    return await _execute(7, _run_agent7, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 8 — Content Repurposing Flow  (Flow, and_ parallel + run_crew helper)
# Standalone: project8_content_repurposing_flow.py
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_ARTICLE = (
    "Precision agriculture uses sensors, drones and AI to help farmers use water, "
    "seeds and fertiliser exactly where needed. It cuts costs and raises yields."
)


def _run_agent8(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai.flow.flow import Flow, listen, start, and_
    from crewai_tools import ScrapeWebsiteTool

    llm = get_llm(temperature=0.3, parameters=parameters)

    # Determine source: URL, prompt text, or context article
    source = (context or {}).get("source_url", "")
    article_text = prompt if len(prompt.strip()) > 30 else (context or {}).get("article", _DEFAULT_ARTICLE)

    class ContentRepurposingFlow(Flow[RepurposeState]):
        @start()
        def load_article(self):
            if source.startswith("http"):
                try:
                    text = ScrapeWebsiteTool(website_url=source).run()
                    self.state.article = text[:6000]
                except Exception:
                    self.state.article = article_text
            else:
                self.state.article = article_text

        @listen(load_article)
        def extract_points(self):
            self.state.key_points = llm.call(
                "Extract the 5 most important points of this article as short bullets:\n\n"
                + self.state.article
            )

        @listen(extract_points)
        def linkedin_crew(self):
            self.state.linkedin = run_crew_helper(
                role="LinkedIn Copywriter",
                goal="Write a professional LinkedIn post",
                backstory="You write posts that get read.",
                description="Key points:\n{points}\n\nWrite a LinkedIn post: a strong hook line, about 150 words, 3 hashtags.",
                expected_output="One LinkedIn post, ~150 words.",
                inputs={"points": self.state.key_points},
            )

        @listen(extract_points)
        def twitter_crew(self):
            self.state.twitter = run_crew_helper(
                role="Twitter Thread Writer",
                goal="Write a 6-tweet thread",
                backstory="You make complex ideas short and punchy.",
                description="Key points:\n{points}\n\nWrite a thread of exactly 6 tweets, numbered 1/6 to 6/6, each under 270 characters.",
                expected_output="6 numbered tweets.",
                inputs={"points": self.state.key_points},
            )

        @listen(extract_points)
        def newsletter_crew(self):
            self.state.newsletter = run_crew_helper(
                role="Newsletter Editor",
                goal="Write a friendly newsletter section",
                backstory="You write newsletters people look forward to.",
                description="Key points:\n{points}\n\nWrite a short intro (2 sentences) and exactly 3 bullet takeaways.",
                expected_output="Intro plus 3 bullets.",
                inputs={"points": self.state.key_points},
            )

        @listen(and_(linkedin_crew, twitter_crew, newsletter_crew))
        def publish_pack(self):
            return "3 content pieces generated successfully"

    flow = ContentRepurposingFlow()
    flow.kickoff()
    return (
        f"### LinkedIn Post\n\n{flow.state.linkedin}\n\n"
        f"---\n### Twitter/X Thread\n\n{flow.state.twitter}\n\n"
        f"---\n### Newsletter Section\n\n{flow.state.newsletter}"
    )


@app.post("/agent8", response_model=AgentResponse, tags=["Agents"])
async def agent8_endpoint(request: AgentRequest):
    """Content Repurposing Flow — extract key points → LinkedIn + Twitter thread + Newsletter (parallel).

    **prompt**: article text. Or pass `context.source_url` for a web URL, or `context.article`.
    """
    return await _execute(8, _run_agent8, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 9 — Lead Qualification Flow  (Flow, Router + or_ + run_crew helper)
# Standalone: project9_lead_qualification_flow.py
# ─────────────────────────────────────────────────────────────────────────────
_OUR_OFFER = "An AI tool that automates customer-support replies for online businesses."


def _run_agent9(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai.flow.flow import Flow, listen, start, router, or_
    from crewai_tools import ScrapeWebsiteTool

    llm = get_llm(temperature=0, parameters=parameters)

    company_name = prompt.strip() or "Example Shop"
    website_url = (context or {}).get("website_url", "https://www.shopify.com")

    class LeadQualificationFlow(Flow[LeadState]):
        @start()
        def capture_lead(self):
            self.state.name = company_name
            self.state.website = website_url

        @listen(capture_lead)
        def enrich_lead(self):
            self.state.profile = run_crew_helper(
                role="Sales Researcher",
                goal="Build a short company profile from its website",
                backstory="You research prospects before sales calls.",
                description=(
                    "Open {site} with the scrape tool. Write: what the company does, "
                    "who its customers are, and any sign it has many customers or support needs (max 100 words)."
                ),
                expected_output="A profile of max 100 words.",
                tools=[ScrapeWebsiteTool()],
                inputs={"site": self.state.website},
            )

        @router(enrich_lead)
        def score_lead(self):
            _prompt = (
                f"We sell: {_OUR_OFFER}\nLead profile: {self.state.profile}\n\n"
                "hot = clear need and a good fit; warm = possible fit; cold = poor fit.\n"
                "Reply with exactly one word: hot, warm or cold."
            )
            answer = llm.call(_prompt).strip().lower()
            self.state.score = answer if answer in ("hot", "warm", "cold") else "cold"
            return self.state.score

        @listen("hot")
        def outreach_crew(self):
            self.state.email = run_crew_helper(
                role="Sales Copywriter",
                goal="Write a personal outreach email",
                backstory="You write short emails that get replies.",
                description=(
                    "Lead: {name}\nProfile: {profile}\nWe sell: {offer}\n"
                    "Write a personal email under 120 words with one clear call to action."
                ),
                expected_output="Subject line and email body.",
                inputs={"name": self.state.name, "profile": self.state.profile, "offer": _OUR_OFFER},
            )
            self.state.action = "outreach email drafted"

        @listen("warm")
        def add_to_nurture(self):
            nurture_path = os.path.join(os.getcwd(), "nurture_list.txt")
            with open(nurture_path, "a", encoding="utf-8") as f:
                f.write(f"{self.state.name},{self.state.website}\n")
            self.state.action = "added to weekly tips list"

        @listen("cold")
        def archive_lead(self):
            archive_dir = os.path.join(os.getcwd(), "archive")
            os.makedirs(archive_dir, exist_ok=True)
            with open(os.path.join(archive_dir, f"{self.state.name}.txt"), "w", encoding="utf-8") as f:
                f.write(self.state.profile)
            self.state.action = "archived, no action"

        @listen(or_(outreach_crew, add_to_nurture, archive_lead))
        def log_lead(self):
            leads_path = os.path.join(os.getcwd(), "leads.csv")
            new_file = not os.path.exists(leads_path)
            with open(leads_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(["name", "website", "score", "action"])
                writer.writerow([self.state.name, self.state.website,
                                  self.state.score, self.state.action])
            return self.state.action

    flow = LeadQualificationFlow()
    flow.kickoff()
    return (
        f"**Company:** {flow.state.name}\n"
        f"**Website:** {flow.state.website}\n"
        f"**Qualification:** {flow.state.score.upper()}\n"
        f"**Action:** {flow.state.action}\n\n"
        f"**Profile:**\n{flow.state.profile}\n\n"
        + (f"**Draft Email:**\n{flow.state.email}" if flow.state.email else "")
    )


@app.post("/agent9", response_model=AgentResponse, tags=["Agents"])
async def agent9_endpoint(request: AgentRequest):
    """Lead Qualification Flow — scrape website, score (hot/warm/cold), act & log to leads.csv.

    **prompt**: company name.
    Optional context: `website_url`.
    """
    return await _execute(9, _run_agent9, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 10 — Deep Research Report Flow  (research → write → revision crews)
# Standalone: Projects/agent10/agent10.ipynb  (ApprovalReportFlow)
# Note: human_feedback loop is replaced with a single-pass auto-approval for API use.
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent10(prompt: str,
                 context: Optional[Dict[str, Any]] = None,
                 parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew
    from crewai.flow.flow import Flow, listen, start
    from crewai.tools import tool

    llm = get_llm(temperature=0.7, parameters=parameters)

    @tool("DuckDuckGo Search")
    def web_search(query: str) -> str:
        """Search the web and return titles, snippets and links."""
        return duckduckgo_search_fn(query)

    topic = prompt.strip() or "Multi-Agent AI Systems in Production 2026"
    audience = (context or {}).get("audience", "Software Engineers & Tech Leaders")

    def build_research_crew() -> Crew:
        researcher_agent = Agent(
            role="Reports Researcher",
            goal="Find reliable, recent sources about {topic}",
            backstory="You are skilled at finding credible sources quickly.",
            llm=llm, verbose=True, tools=[web_search],
            allow_delegation=False, max_iter=5,
        )
        analyst_agent = Agent(
            role="Research Analyst",
            goal="Extract the key facts and figures about {topic} for {audience}",
            backstory="You turn raw search results into clear findings.",
            llm=llm, verbose=True, tools=[],
            allow_delegation=False, max_iter=5,
        )
        checker_agent = Agent(
            role="Fact Checker",
            goal="Remove weak, duplicate or unsupported claims",
            backstory="You are strict about accuracy and always keep source links.",
            llm=llm, verbose=True, tools=[web_search],
            allow_delegation=False, max_iter=5,
        )
        analyst_task = Task(
            description="Search for information about {topic}. Collect sources.",
            expected_output="A list of 5 sources with short summaries and links.",
            agent=analyst_agent,
        )
        researcher_task = Task(
            description="From the sources, extract key facts for {audience}.",
            expected_output="A bullet list of key facts, each tied to a source.",
            agent=researcher_agent,
        )
        checker_task = Task(
            description="Check the facts. Drop anything weak or duplicated.",
            expected_output="A clean research notes document with sources.",
            agent=checker_agent,
        )
        return Crew(
            agents=[researcher_agent, analyst_agent, checker_agent],
            tasks=[researcher_task, analyst_task, checker_task],
            verbose=True,
        )

    def build_writing_crew() -> Crew:
        writer_agent = Agent(
            role="Report Writer",
            goal="Write a clear report for {audience}",
            backstory="You write structured, readable reports from research notes.",
            llm=llm, verbose=True, tools=[],
            allow_delegation=False, max_iter=5,
        )
        task = Task(
            description=(
                "Write a report on {topic} for {audience} using these notes:\n\n"
                "{research_notes}"
            ),
            expected_output="A markdown report with a title, sections and a sources list.",
            agent=writer_agent,
        )
        return Crew(agents=[writer_agent], tasks=[task])

    class ResearchReportFlow(Flow[ReportState]):
        @start()
        def collect_topic(self):
            self.state.topic = topic
            self.state.audience = audience

        @listen(collect_topic)
        def run_research(self):
            result = build_research_crew().kickoff(
                inputs={"topic": self.state.topic, "audience": self.state.audience}
            )
            self.state.research_notes = result.raw

        @listen(run_research)
        def write_report(self):
            result = build_writing_crew().kickoff(
                inputs={
                    "topic": self.state.topic,
                    "audience": self.state.audience,
                    "research_notes": self.state.research_notes,
                }
            )
            self.state.draft = result.raw
            self.state.final_report = self.state.draft
            # Save final report to disk
            os.makedirs("output", exist_ok=True)
            with open(os.path.join("output", "final_report.md"), "w", encoding="utf-8") as f:
                f.write(self.state.final_report)

    flow = ResearchReportFlow()
    flow.kickoff()
    return flow.state.final_report.strip()


@app.post("/agent10", response_model=AgentResponse, tags=["Agents"])
async def agent10_endpoint(request: AgentRequest):
    """Deep Research Report Flow — multi-crew pipeline: research → fact-check → write report.

    **prompt**: research topic.
    Optional context: `audience` (default: "Software Engineers & Tech Leaders").
    """
    return await _execute(10, _run_agent10, request)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
