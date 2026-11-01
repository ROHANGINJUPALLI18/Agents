"""
test_main.py — FastAPI unit tests for the 10-agent CrewAI backend.

Strategy:
  - Each _run_agentN function is mocked so no real LLM calls are made.
  - We test that:
      1. The HTTP status is 200.
      2. The response contains the correct agent_id.
      3. The status field is "success".
      4. The result field contains the mocked string.
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(endpoint: str, prompt: str = "test prompt", context: dict = None, parameters: dict = None):
    payload = {"prompt": prompt}
    if context:
        payload["context"] = context
    if parameters:
        payload["parameters"] = parameters
    return client.post(endpoint, json=payload)


def _assert_ok(response, agent_num: int, mocked_result: str):
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["agent_id"] == f"agent{agent_num}"
    assert data["status"] == "success"
    assert mocked_result in data["result"]


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "agents" in r.json()


# ---------------------------------------------------------------------------
# Agent 1 — Science Explainer
# ---------------------------------------------------------------------------

def test_agent1():
    with patch("main._run_agent1", return_value="Science mocked"):
        r = _post("/agent1", "Explain quantum entanglement")
    _assert_ok(r, 1, "Science mocked")


# ---------------------------------------------------------------------------
# Agent 2 — Competitor Intelligence
# ---------------------------------------------------------------------------

def test_agent2():
    with patch("main._run_agent2", return_value="Competitor mocked"):
        r = _post("/agent2", "Notion, ClickUp")
    _assert_ok(r, 2, "Competitor mocked")


# ---------------------------------------------------------------------------
# Agent 3 — Meeting Notes
# ---------------------------------------------------------------------------

def test_agent3():
    with patch("main._run_agent3", return_value="Meeting mocked"):
        r = _post("/agent3", "Alex: Let's sync. Sarah: Agreed.")
    _assert_ok(r, 3, "Meeting mocked")


# ---------------------------------------------------------------------------
# Agent 4 — Sales Analytics
# ---------------------------------------------------------------------------

def test_agent4():
    with patch("main._run_agent4", return_value="Sales mocked"):
        r = _post("/agent4", "Q3 revenue breakdown")
    _assert_ok(r, 4, "Sales mocked")


def test_agent4_with_csv():
    csv = "product,region,units_sold,unit_price,revenue\nWidget,NA,10,5,50"
    with patch("main._run_agent4", return_value="CSV mocked"):
        r = _post("/agent4", "analyze", context={"csv_data": csv})
    _assert_ok(r, 4, "CSV mocked")


# ---------------------------------------------------------------------------
# Agent 5 — Customer Support Triage
# ---------------------------------------------------------------------------

def test_agent5():
    with patch("main._run_agent5", return_value="Billing mocked"):
        r = _post("/agent5", "I was charged twice for invoice #9823")
    _assert_ok(r, 5, "Billing mocked")


# ---------------------------------------------------------------------------
# Agent 6 — Resume Screening
# ---------------------------------------------------------------------------

def test_agent6():
    with patch("main._run_agent6", return_value="Resume mocked"):
        r = _post("/agent6", "Candidate: Jane Doe | Skills: Python")
    _assert_ok(r, 6, "Resume mocked")


def test_agent6_with_candidate_name():
    with patch("main._run_agent6", return_value="Jane mocked"):
        r = _post("/agent6", "resume text here", context={"candidate_name": "Jane Doe"})
    _assert_ok(r, 6, "Jane mocked")


# ---------------------------------------------------------------------------
# Agent 7 — Bug Triage
# ---------------------------------------------------------------------------

def test_agent7():
    with patch("main._run_agent7", return_value="Bug mocked"):
        r = _post("/agent7", "App crashes on checkout")
    _assert_ok(r, 7, "Bug mocked")


def test_agent7_with_body():
    with patch("main._run_agent7", return_value="Bug body mocked"):
        r = _post("/agent7", "500 error", context={"body": "KeyError in payments.py"})
    _assert_ok(r, 7, "Bug body mocked")


# ---------------------------------------------------------------------------
# Agent 8 — Content Repurposing
# ---------------------------------------------------------------------------

def test_agent8():
    with patch("main._run_agent8", return_value="Content mocked"):
        r = _post("/agent8", "AI agents are transforming software in 2026 with autonomous workflows.")
    _assert_ok(r, 8, "Content mocked")


# ---------------------------------------------------------------------------
# Agent 9 — Lead Qualification
# ---------------------------------------------------------------------------

def test_agent9():
    with patch("main._run_agent9", return_value="Lead mocked"):
        r = _post("/agent9", "Notion")
    _assert_ok(r, 9, "Lead mocked")


def test_agent9_with_url():
    with patch("main._run_agent9", return_value="Lead URL mocked"):
        r = _post("/agent9", "Acme Corp", context={"website_url": "https://acme.com"})
    _assert_ok(r, 9, "Lead URL mocked")


# ---------------------------------------------------------------------------
# Agent 10 — Deep Research
# ---------------------------------------------------------------------------

def test_agent10():
    with patch("main._run_agent10", return_value="Research mocked"):
        r = _post("/agent10", "Multi-Agent AI in Production 2026")
    _assert_ok(r, 10, "Research mocked")


def test_agent10_with_audience():
    with patch("main._run_agent10", return_value="Audience mocked"):
        r = _post("/agent10", "AI Trends", context={"audience": "CTOs"})
    _assert_ok(r, 10, "Audience mocked")


# ---------------------------------------------------------------------------
# Error handling — missing prompt
# ---------------------------------------------------------------------------

def test_missing_prompt_returns_422():
    r = client.post("/agent1", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------

def test_custom_parameters_passed():
    with patch("main._run_agent1", return_value="Params mocked") as mock:
        _post("/agent1", "test", parameters={"temperature": 0.9, "max_tokens": 500})
        _, _, kwargs = mock.mock_calls[0]
        # Parameters are passed as positional args: (prompt, context, parameters)
        params_arg = mock.call_args[0][2]
        assert params_arg.get("temperature") == 0.9
        assert params_arg.get("max_tokens") == 500
