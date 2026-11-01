"""
CrewAI Multi-Agent Backend
FastAPI server with 10 real CrewAI agents and flows wired as async POST endpoints.

Agents:
  /agent1  - Science Explainer
  /agent2  - Competitor Intelligence & Market Analysis
  /agent3  - Meeting Notes & Action Items Specialist
  /agent4  - Financial & Sales Analytics
  /agent5  - Customer Support Ticket Triage Flow
  /agent6  - Resume Screening & Technical Evaluation Flow
  /agent7  - Bug & Issue Triage Flow
  /agent8  - Multi-Channel Content Repurposing Flow
  /agent9  - B2B Lead Qualification Flow
  /agent10 - Autonomous Deep Research & Report Flow
"""

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP  (must run before any crewai import so LOCALAPPDATA is writable)
# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import io
import datetime
import os
import sys
import tempfile
from typing import Any, Callable, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _tmp = os.path.join(tempfile.gettempdir(), "crewai_localappdata")
    os.makedirs(_tmp, exist_ok=True)
    os.environ["LOCALAPPDATA"] = _tmp

# ─────────────────────────────────────────────────────────────────────────────
# FLOW STATE MODELS  (module-level — required by CrewAI Flow generic typing)
# ─────────────────────────────────────────────────────────────────────────────
class SupportState(BaseModel):
    ticket_text: str = ""
    customer_name: str = "Valued Customer"
    category: str = ""
    priority: str = "medium"
    resolution: str = ""


class ScreeningState(BaseModel):
    candidate: str = "Jordan Lee"
    resume_text: str = ""
    score: int = 0
    decision: str = ""
    evaluation: str = ""


class BugTriageState(BaseModel):
    issue_title: str = ""
    issue_body: str = ""
    issue_type: str = ""
    resolution: str = ""


class RepurposingState(BaseModel):
    article: str = ""
    linkedin_post: str = ""
    twitter_thread: str = ""
    newsletter_snippet: str = ""


class LeadState(BaseModel):
    company_name: str = ""
    website_url: str = ""
    scraped_info: str = ""
    qualification_tier: str = ""
    action_plan: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP & REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CrewAI Multi-Agent Backend",
    description=(
        "10 production-ready AI agent endpoints powered by CrewAI + Gemini.\n\n"
        "Each endpoint accepts a `prompt`, optional `context` dict, and optional "
        "`parameters` dict (supports `model`, `temperature`, `max_tokens`)."
    ),
    version="2.0.0",
)


class AgentRequest(BaseModel):
    prompt: str = Field(..., description="Task prompt or instruction for the agent")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional contextual data (e.g. transcript, csv_data, website_url)"
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
def _get_llm(parameters: Optional[Dict[str, Any]] = None,
             default_temp: float = 0.5,
             default_tokens: int = 1500):
    """Build a CrewAI LLM object from request parameters."""
    from crewai import LLM
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in .env")
    p = parameters or {}
    return LLM(
        model=p.get("model", "gemini/gemini-2.5-flash"),
        api_key=api_key,
        temperature=float(p.get("temperature", default_temp)),
        max_tokens=int(p.get("max_tokens", default_tokens)),
    )


def _safe_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo search with graceful failure."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        return "\n\n".join(
            f"{r.get('title')}\n{r.get('href')}\n{r.get('body')}" for r in results
        )
    except Exception as exc:
        return f"[Search unavailable: {exc}]"


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
            result=result_text,
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
            "/agent1":  "Science Explainer",
            "/agent2":  "Competitor Intelligence & Market Analysis",
            "/agent3":  "Meeting Notes & Action Items Specialist",
            "/agent4":  "Financial & Sales Analytics",
            "/agent5":  "Customer Support Ticket Triage Flow",
            "/agent6":  "Resume Screening & Technical Evaluation Flow",
            "/agent7":  "Bug & Issue Triage Flow",
            "/agent8":  "Multi-Channel Content Repurposing Flow",
            "/agent9":  "B2B Lead Qualification Flow",
            "/agent10": "Autonomous Deep Research & Report Flow",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — Science Explainer
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent1(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew
    llm = _get_llm(parameters, default_temp=0.7, default_tokens=1000)

    agent = Agent(
        role="Science Explainer",
        goal="Explain scientific concepts in an easy-to-understand and practical manner.",
        backstory=(
            "You are an experienced science teacher with 15+ years of explaining "
            "scientific topics with practical intuition and clear real-world examples."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    desc = f"Explain this topic or answer this question: {prompt}"
    if context:
        desc += f"\nAdditional context: {context}"

    task = Task(
        description=desc,
        expected_output="A clear, simple explanation in plain English with intuitive examples.",
        agent=agent,
    )
    return str(Crew(agents=[agent], tasks=[task], verbose=True).kickoff().raw).strip()


@app.post("/agent1", response_model=AgentResponse, tags=["Agents"])
async def agent1_endpoint(request: AgentRequest):
    """Science Explainer — explains any scientific topic in simple language."""
    return await _execute(1, _run_agent1, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — Competitor Intelligence & Market Analysis
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent2(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool

    llm = _get_llm(parameters, default_temp=0.3, default_tokens=1500)
    competitors = prompt.strip() or "Notion vs ClickUp"

    @tool("DuckDuckGoSearch")
    def ddg_search(query: str) -> str:
        """Search DuckDuckGo for latest info."""
        return _safe_search(query)

    scraper = Agent(
        role="Product Offering Analyst",
        goal="Analyze product offerings and core capabilities for: {competitors}",
        backstory="Expert at identifying core value propositions of SaaS products.",
        llm=llm, verbose=True, allow_delegation=False,
    )
    pricing = Agent(
        role="Pricing Analyst",
        goal="Research tiers, pricing structures, and plan limits for {competitors}",
        backstory="Specialist in SaaS pricing models and enterprise plan comparisons.",
        llm=llm, verbose=True, allow_delegation=False,
    )
    news = Agent(
        role="Market Intelligence Analyst",
        goal="Find recent product updates and announcements for {competitors}",
        backstory="Market intelligence researcher tracking launches and updates.",
        llm=llm, tools=[ddg_search], verbose=True, allow_delegation=False,
    )
    strategist = Agent(
        role="Strategy Consultant",
        goal="Synthesize findings into a SWOT matrix and strategic recommendations",
        backstory="Executive advisor providing actionable market insights.",
        llm=llm, verbose=True, allow_delegation=False,
    )

    t1 = Task(description="Analyze product offerings for: {competitors}.",
              expected_output="Structured summary per competitor.", agent=scraper)
    t2 = Task(description="Examine pricing tiers for: {competitors}.",
              expected_output="Comparative pricing overview.", agent=pricing)
    t3 = Task(description="Search latest news for: {competitors}.",
              expected_output="Recent updates per competitor.", agent=news)
    t4 = Task(
        description="Create SWOT matrix and 3 strategic recommendations from the gathered data.",
        expected_output="Comprehensive Markdown competitor analysis report.",
        agent=strategist, context=[t1, t2, t3],
    )

    crew = Crew(
        agents=[scraper, pricing, news, strategist],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff(inputs={"competitors": competitors}).raw).strip()


@app.post("/agent2", response_model=AgentResponse, tags=["Agents"])
async def agent2_endpoint(request: AgentRequest):
    """Competitor Intelligence — SWOT analysis and market positioning report.

    **prompt**: comma-separated competitor names or URLs (e.g. "Notion, ClickUp")
    """
    return await _execute(2, _run_agent2, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — Meeting Notes & Action Items Specialist
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent3(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew, Process

    llm = _get_llm(parameters, default_temp=0.3, default_tokens=1200)

    default_transcript = (
        "Alex: Let's finalize the Q3 launch date. I suggest August 15th.\n"
        "Sarah: August 15th works if the backend migration finishes by July 28th.\n"
        "David: I can finish the migration by July 28th, but I need two contractors approved by Friday.\n"
        "Alex: Approved. Sarah, please coordinate with HR to get contractors onboarded by Monday.\n"
        "Sarah: Will do. What about marketing collateral?\n"
        "Elena: Initial landing page review is set for August 5th."
    )
    transcript = prompt if len(prompt.strip()) > 30 else (context or {}).get("transcript", default_transcript)

    summarizer = Agent(
        role="Meeting Summarizer",
        goal="Extract high-level discussion points and decisions from meeting transcripts",
        backstory="Executive assistant expert in synthesizing discussions into crisp briefs.",
        llm=llm, verbose=True,
    )
    tracker = Agent(
        role="Action Items Specialist",
        goal="Identify action items, owners, deadlines from the transcript",
        backstory="Project manager ensuring every task has clear accountability.",
        llm=llm, verbose=True,
    )
    editor = Agent(
        role="Documentation Editor",
        goal="Format the summary and action items into a polished meeting brief",
        backstory="Technical communicator producing publication-ready team memos.",
        llm=llm, verbose=True,
    )

    t1 = Task(description="Analyze transcript: {transcript}. Summarize objectives and decisions.",
              expected_output="Executive summary of main topics.", agent=summarizer)
    t2 = Task(description="Extract action items with owners and deadlines from: {transcript}.",
              expected_output="Markdown table: Owner | Task | Deadline.", agent=tracker)
    t3 = Task(description="Combine summary and action table into one Markdown memo.",
              expected_output="Complete meeting memo in Markdown.",
              agent=editor, context=[t1, t2])

    crew = Crew(agents=[summarizer, tracker, editor], tasks=[t1, t2, t3],
                process=Process.sequential, verbose=True)
    return str(crew.kickoff(inputs={"transcript": transcript}).raw).strip()


@app.post("/agent3", response_model=AgentResponse, tags=["Agents"])
async def agent3_endpoint(request: AgentRequest):
    """Meeting Notes — summarizes transcripts and extracts action items.

    **prompt**: paste the meeting transcript directly, or pass it via `context.transcript`.
    """
    return await _execute(3, _run_agent3, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — Financial & Sales Analytics
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent4(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool

    llm = _get_llm(parameters, default_temp=0.2, default_tokens=1200)

    default_csv = (
        "product,region,units_sold,unit_price,revenue\n"
        "Cloud Backup,North America,120,50,6000\n"
        "API Gateway,Europe,80,120,9600\n"
        "Auth Service,North America,200,30,6000\n"
        "Cloud Backup,Europe,90,50,4500\n"
        "API Gateway,Asia-Pacific,110,120,13200\n"
        "Auth Service,Europe,150,30,4500\n"
        "Monitoring,North America,70,80,5600\n"
        "Monitoring,Asia-Pacific,60,80,4800"
    )
    csv_data = (context or {}).get("csv_data", default_csv)

    @tool("ComputeSalesMetrics")
    def compute_metrics() -> str:
        """Compute total revenue, units, top product, and top region from CSV."""
        df = pd.read_csv(io.StringIO(csv_data))
        return (
            f"Total Revenue: ${df['revenue'].sum():,.2f}\n"
            f"Total Units Sold: {df['units_sold'].sum():,}\n"
            f"Top Product by Revenue: {df.groupby('product')['revenue'].sum().idxmax()}\n"
            f"Top Region by Revenue: {df.groupby('region')['revenue'].sum().idxmax()}"
        )

    analyst = Agent(
        role="Data Analyst",
        goal=f"Compute sales metrics for request: {prompt}",
        backstory="Quantitative analyst computing accurate financial and volume totals.",
        llm=llm, tools=[compute_metrics], verbose=True,
    )
    insight = Agent(
        role="Business Insights Specialist",
        goal="Interpret metrics and draft strategic observations",
        backstory="Business strategist transforming metrics into executive narratives.",
        llm=llm, verbose=True,
    )
    reviewer = Agent(
        role="Executive Editor",
        goal="Format the final sales report in Markdown",
        backstory="Operations director verifying report clarity and data precision.",
        llm=llm, verbose=True,
    )

    t1 = Task(description=f"Use ComputeSalesMetrics to compute totals for: '{prompt}'.",
              expected_output="Numeric metrics: revenue, units, top product, top region.",
              agent=analyst)
    t2 = Task(description="Interpret the metrics and outline market trends.",
              expected_output="Bullet list of analytical insights.", agent=insight, context=[t1])
    t3 = Task(description="Compile metrics and insights into a formal executive sales report.",
              expected_output="Markdown report: Metrics Table | Key Observations | Next Steps.",
              agent=reviewer, context=[t1, t2])

    crew = Crew(agents=[analyst, insight, reviewer], tasks=[t1, t2, t3],
                process=Process.sequential, verbose=True)
    return str(crew.kickoff().raw).strip()


@app.post("/agent4", response_model=AgentResponse, tags=["Agents"])
async def agent4_endpoint(request: AgentRequest):
    """Sales Analytics — computes metrics and generates an executive report.

    Pass CSV data as a string via `context.csv_data`. Default sample data used otherwise.
    """
    return await _execute(4, _run_agent4, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — Customer Support Ticket Triage Flow
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent5(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai.flow.flow import Flow, listen, router, start

    llm = _get_llm(parameters, default_temp=0.3, default_tokens=800)
    ticket = prompt.strip() or "I was charged twice for subscription invoice #9823 on my Visa card."

    class CustomerSupportFlow(Flow[SupportState]):
        @start()
        def receive_ticket(self):
            self.state.ticket_text = ticket

        @listen(receive_ticket)
        def categorize_ticket(self):
            resp = llm.call(
                f"Classify this support ticket into exactly one word: billing, technical, account, or other.\n"
                f"Ticket: {self.state.ticket_text}\nRespond with ONLY the category word in lowercase."
            ).strip().lower()
            if "billing" in resp:
                self.state.category, self.state.priority = "billing", "high"
            elif "tech" in resp:
                self.state.category, self.state.priority = "technical", "high"
            elif "account" in resp:
                self.state.category, self.state.priority = "account", "medium"
            else:
                self.state.category, self.state.priority = "other", "low"

        @router(categorize_ticket)
        def route(self):
            return self.state.category

        @listen("billing")
        def billing(self):
            self.state.resolution = llm.call(
                f"Draft a professional billing support response for: {self.state.ticket_text}. "
                f"State that refund review takes 24 hours."
            )

        @listen("technical")
        def technical(self):
            self.state.resolution = llm.call(
                f"Draft an empathetic technical support response for: {self.state.ticket_text}. "
                f"Request reproduction steps."
            )

        @listen("account")
        def account(self):
            self.state.resolution = llm.call(
                f"Draft an account security response for: {self.state.ticket_text}. "
                f"Provide secure password reset steps."
            )

        @listen("other")
        def other(self):
            self.state.resolution = llm.call(
                f"Draft a polite customer support response acknowledging: {self.state.ticket_text}."
            )

    flow = CustomerSupportFlow()
    flow.kickoff()
    return (
        f"**Category:** {flow.state.category.upper()}\n"
        f"**Priority:** {flow.state.priority.upper()}\n\n"
        f"**Resolution:**\n{flow.state.resolution}"
    )


@app.post("/agent5", response_model=AgentResponse, tags=["Agents"])
async def agent5_endpoint(request: AgentRequest):
    """Support Triage — classifies ticket, assigns priority, drafts resolution.

    **prompt**: the support ticket text.
    """
    return await _execute(5, _run_agent5, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 6 — Resume Screening & Technical Evaluation Flow
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent6(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew
    from crewai.flow.flow import Flow, listen, router, start

    llm = _get_llm(parameters, default_temp=0.3, default_tokens=1000)

    default_resume = (
        "Candidate: Jordan Lee | Experience: 6 years as Full Stack Engineer\n"
        "Skills: Python, FastAPI, Docker, PostgreSQL, React, Kubernetes, LLM integration\n"
        "Achievements: Built real-time streaming pipeline reducing latency by 45%;"
        " deployed production CrewAI agents."
    )
    resume = prompt if len(prompt.strip()) > 30 else default_resume
    candidate = (context or {}).get("candidate_name", "Jordan Lee")

    class ResumeFlow(Flow[ScreeningState]):
        @start()
        def ingest(self):
            self.state.candidate = candidate
            self.state.resume_text = resume

        @listen(ingest)
        def score(self):
            raw = llm.call(
                f"Score this resume for a Senior AI/Backend Engineer role (1-10).\n"
                f"Resume: {self.state.resume_text}\nReply with ONLY an integer 1-10."
            ).strip()
            digits = "".join(c for c in raw if c.isdigit())
            self.state.score = int(digits[:2]) if digits else 6

        @router(score)
        def decide(self):
            if self.state.score >= 7:
                self.state.decision = "shortlist"
                return "shortlist"
            self.state.decision = "reject"
            return "reject"

        @listen("shortlist")
        def shortlist(self):
            ag = Agent(
                role="Technical Interview Lead",
                goal="Highlight candidate strengths and recommend interview focus areas",
                backstory="Engineering director assembling high-performing teams.",
                llm=llm, verbose=False,
            )
            t = Task(
                description=f"Draft interview summary for {self.state.candidate}. Resume: {self.state.resume_text}",
                expected_output="Shortlist memo with suggested interview topics.",
                agent=ag,
            )
            self.state.evaluation = str(Crew(agents=[ag], tasks=[t], verbose=False).kickoff().raw).strip()

        @listen("reject")
        def reject(self):
            self.state.evaluation = llm.call(
                f"Write a courteous rejection email to {self.state.candidate} for a Software Engineer role."
            )

    flow = ResumeFlow()
    flow.kickoff()
    return (
        f"**Candidate:** {flow.state.candidate}\n"
        f"**Score:** {flow.state.score}/10\n"
        f"**Decision:** {flow.state.decision.upper()}\n\n"
        f"**Evaluation:**\n{flow.state.evaluation}"
    )


@app.post("/agent6", response_model=AgentResponse, tags=["Agents"])
async def agent6_endpoint(request: AgentRequest):
    """Resume Screening — scores, routes (shortlist/reject), drafts interview memo.

    **prompt**: the resume text. Pass `context.candidate_name` for the candidate's name.
    """
    return await _execute(6, _run_agent6, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 7 — Bug & Issue Triage Flow
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent7(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew
    from crewai.flow.flow import Flow, listen, router, start

    llm = _get_llm(parameters, default_temp=0.2, default_tokens=1000)

    title = prompt.strip() or "App crashes with 500 on checkout when using PayPal"
    body = (context or {}).get("body",
        "Traceback indicates KeyError 'order_id' in payments/paypal.py line 42 on Safari mobile.")

    class TriageFlow(Flow[BugTriageState]):
        @start()
        def receive(self):
            self.state.issue_title = title
            self.state.issue_body = body

        @listen(receive)
        def classify(self):
            resp = llm.call(
                f"Classify this GitHub issue as exactly one word: bug, feature, or question.\n"
                f"Title: {self.state.issue_title}\nBody: {self.state.issue_body}\n"
                f"Reply with ONLY the lowercase word."
            ).strip().lower()
            self.state.issue_type = (
                "feature" if "feature" in resp
                else "question" if "question" in resp
                else "bug"
            )

        @router(classify)
        def route(self):
            return self.state.issue_type

        @listen("bug")
        def handle_bug(self):
            eng = Agent(
                role="QA Triager",
                goal="Assess severity, blast radius, and reproduction steps",
                backstory="Reliability engineer diagnosing and triaging defects.",
                llm=llm, verbose=False,
            )
            t = Task(
                description=(
                    f"Analyze defect: {self.state.issue_title}. "
                    f"Details: {self.state.issue_body}. "
                    f"Format as Severity | Steps to Reproduce | Root Cause Hypothesis."
                ),
                expected_output="Formatted bug triage memo.",
                agent=eng,
            )
            self.state.resolution = str(Crew(agents=[eng], tasks=[t], verbose=False).kickoff().raw).strip()

        @listen("feature")
        def handle_feature(self):
            pm = Agent(
                role="Product Manager",
                goal="Convert feature request into user story with acceptance criteria",
                backstory="Technical PM defining stories and product requirements.",
                llm=llm, verbose=False,
            )
            t = Task(
                description=f"Create feature spec for: {self.state.issue_title}. Context: {self.state.issue_body}.",
                expected_output="User story with Given/When/Then acceptance criteria.",
                agent=pm,
            )
            self.state.resolution = str(Crew(agents=[pm], tasks=[t], verbose=False).kickoff().raw).strip()

        @listen("question")
        def handle_question(self):
            self.state.resolution = llm.call(
                f"Answer this developer question with code or doc references:\n"
                f"{self.state.issue_title}\n{self.state.issue_body}"
            )

    flow = TriageFlow()
    flow.kickoff()
    return f"**Issue Type:** {flow.state.issue_type.upper()}\n\n{flow.state.resolution}"


@app.post("/agent7", response_model=AgentResponse, tags=["Agents"])
async def agent7_endpoint(request: AgentRequest):
    """Bug Triage — classifies issue as bug/feature/question and drafts resolution.

    **prompt**: the issue title. Pass `context.body` for the issue description.
    """
    return await _execute(7, _run_agent7, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 8 — Multi-Channel Content Repurposing Flow
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent8(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew
    from crewai.flow.flow import Flow, and_, listen, start

    llm = _get_llm(parameters, default_temp=0.6, default_tokens=1500)

    default_article = (
        "Generative AI in 2026 is moving from chat interfaces directly into background agent workflows. "
        "Companies are implementing autonomous multi-agent orchestrations where specialist agents validate data, "
        "query internal APIs, and execute end-to-end business processes with reliability and schema validations."
    )
    article = prompt if len(prompt.strip()) > 30 else default_article

    class RepurposeFlow(Flow[RepurposingState]):
        @start()
        def load(self):
            self.state.article = article

        @listen(load)
        def linkedin(self):
            ag = Agent(
                role="LinkedIn Content Creator",
                goal="Draft an engaging professional LinkedIn post",
                backstory="Growth strategist creating hook-driven thought leadership content.",
                llm=llm, verbose=False,
            )
            t = Task(
                description=f"Repurpose into a LinkedIn post with hook, key points, and discussion question:\n{self.state.article}",
                expected_output="LinkedIn post with hashtags.",
                agent=ag,
            )
            self.state.linkedin_post = str(Crew(agents=[ag], tasks=[t], verbose=False).kickoff().raw).strip()

        @listen(load)
        def twitter(self):
            ag = Agent(
                role="X/Twitter Strategist",
                goal="Draft a punchy 4-tweet thread",
                backstory="Social media creator skilled in concise hooks and takeaways.",
                llm=llm, verbose=False,
            )
            t = Task(
                description=f"Repurpose into a numbered 4-tweet thread:\n{self.state.article}",
                expected_output="4-tweet thread (1/4 → 4/4).",
                agent=ag,
            )
            self.state.twitter_thread = str(Crew(agents=[ag], tasks=[t], verbose=False).kickoff().raw).strip()

        @listen(load)
        def newsletter(self):
            ag = Agent(
                role="Newsletter Editor",
                goal="Draft a weekly newsletter section with TL;DR and action items",
                backstory="Curator of a tech newsletter with 50k+ subscribers.",
                llm=llm, verbose=False,
            )
            t = Task(
                description=f"Repurpose into a newsletter section (TL;DR, Core Takeaway, Action Item):\n{self.state.article}",
                expected_output="Structured newsletter section in Markdown.",
                agent=ag,
            )
            self.state.newsletter_snippet = str(Crew(agents=[ag], tasks=[t], verbose=False).kickoff().raw).strip()

        @listen(and_(linkedin, twitter, newsletter))
        def done(self):
            pass

    flow = RepurposeFlow()
    flow.kickoff()
    return (
        f"### LinkedIn Post\n\n{flow.state.linkedin_post}\n\n"
        f"---\n### Twitter/X Thread\n\n{flow.state.twitter_thread}\n\n"
        f"---\n### Newsletter Snippet\n\n{flow.state.newsletter_snippet}"
    )


@app.post("/agent8", response_model=AgentResponse, tags=["Agents"])
async def agent8_endpoint(request: AgentRequest):
    """Content Repurposing — generates LinkedIn post, tweet thread & newsletter from any article.

    **prompt**: the source article text.
    """
    return await _execute(8, _run_agent8, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 9 — B2B Lead Qualification Flow
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent9(prompt: str,
                context: Optional[Dict[str, Any]] = None,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew
    from crewai.flow.flow import Flow, listen, router, start

    llm = _get_llm(parameters, default_temp=0.3, default_tokens=1000)

    company = prompt.strip() or "Notion"
    url = (context or {}).get("website_url", "https://www.notion.so")

    class LeadFlow(Flow[LeadState]):
        @start()
        def capture(self):
            self.state.company_name = company
            self.state.website_url = url

        @listen(capture)
        def enrich(self):
            ag = Agent(
                role="Sales Intelligence Researcher",
                goal=f"Extract value proposition and target audience for {self.state.company_name}",
                backstory="B2B SDR skilled in researching target accounts.",
                llm=llm, verbose=False,
            )
            t = Task(
                description=f"Analyze market profile for {self.state.company_name} (URL: {self.state.website_url}).",
                expected_output="Concise summary of target market and product offerings.",
                agent=ag,
            )
            self.state.scraped_info = str(Crew(agents=[ag], tasks=[t], verbose=False).kickoff().raw).strip()

        @listen(enrich)
        def score(self):
            tier = llm.call(
                f"Classify this company as: hot, warm, or cold.\n"
                f"Company: {self.state.company_name}\nProfile: {self.state.scraped_info}\n"
                f"Reply with ONLY one word: hot, warm, or cold."
            ).strip().lower()
            self.state.qualification_tier = (
                "hot" if "hot" in tier else "warm" if "warm" in tier else "cold"
            )

        @router(score)
        def route(self):
            return self.state.qualification_tier

        @listen("hot")
        def hot(self):
            self.state.action_plan = llm.call(
                f"Draft a high-conversion enterprise outreach email to {self.state.company_name} "
                f"based on: {self.state.scraped_info}"
            )

        @listen("warm")
        def warm(self):
            self.state.action_plan = (
                f"Enrolled {self.state.company_name} in weekly product case-study nurture campaign."
            )

        @listen("cold")
        def cold(self):
            self.state.action_plan = (
                f"Logged {self.state.company_name} in CRM archive. Retarget in 6 months."
            )

    flow = LeadFlow()
    flow.kickoff()
    return (
        f"**Company:** {flow.state.company_name}\n"
        f"**Tier:** {flow.state.qualification_tier.upper()}\n\n"
        f"**Profile:**\n{flow.state.scraped_info}\n\n"
        f"**Action Plan:**\n{flow.state.action_plan}"
    )


@app.post("/agent9", response_model=AgentResponse, tags=["Agents"])
async def agent9_endpoint(request: AgentRequest):
    """Lead Qualification — scores prospect (Hot/Warm/Cold) and generates outreach.

    **prompt**: company name. Pass `context.website_url` for the company's URL.
    """
    return await _execute(9, _run_agent9, request)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 10 — Autonomous Deep Research & Report Flow
# ─────────────────────────────────────────────────────────────────────────────
def _run_agent10(prompt: str,
                 context: Optional[Dict[str, Any]] = None,
                 parameters: Optional[Dict[str, Any]] = None) -> str:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool

    llm = _get_llm(parameters, default_temp=0.4, default_tokens=2000)

    topic = prompt.strip() or "Multi-Agent AI Systems in Production 2026"
    audience = (context or {}).get("audience", "Software Engineers & Tech Leaders")

    @tool("WebSearch")
    def web_search(query: str) -> str:
        """Search the web for information."""
        return _safe_search(query)

    analyst = Agent(
        role="Research Analyst",
        goal="Find key facts and figures about {topic} for {audience}",
        backstory="Turns raw search results into clear, structured technical insights.",
        llm=llm, tools=[web_search], verbose=True,
    )
    researcher = Agent(
        role="Reports Researcher",
        goal="Find reliable, recent sources about {topic}",
        backstory="Skilled at identifying authoritative sources and key breakthroughs.",
        llm=llm, tools=[web_search], verbose=True,
    )
    checker = Agent(
        role="Fact Checker",
        goal="Remove weak, duplicate or unsupported claims",
        backstory="Strict about accuracy — always preserves source links.",
        llm=llm, verbose=True,
    )
    writer = Agent(
        role="Report Writer",
        goal="Write a clear, comprehensive report for {audience}",
        backstory="Prepares publication-ready whitepapers and executive briefs.",
        llm=llm, verbose=True,
    )

    t1 = Task(description="Gather core facts and sources about {topic}.",
              expected_output="List of facts with sources.", agent=analyst)
    t2 = Task(description="Extract key trends and use cases about {topic} for {audience}.",
              expected_output="Bullet list of insights tied to examples.",
              agent=researcher, context=[t1])
    t3 = Task(description="Validate findings, remove duplicates, ensure accuracy.",
              expected_output="Clean, validated research notes.",
              agent=checker, context=[t1, t2])
    t4 = Task(
        description="Write a comprehensive Markdown report on {topic} for {audience} using validated findings.",
        expected_output="Publication-ready Markdown report: Executive Summary | Findings | Strategic Outlook.",
        agent=writer, context=[t3],
    )

    crew = Crew(
        agents=[analyst, researcher, checker, writer],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff(inputs={"topic": topic, "audience": audience}).raw).strip()


@app.post("/agent10", response_model=AgentResponse, tags=["Agents"])
async def agent10_endpoint(request: AgentRequest):
    """Deep Research — multi-agent research, fact-checking, and full written report.

    **prompt**: research topic. Pass `context.audience` to tailor the report.
    """
    return await _execute(10, _run_agent10, request)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
