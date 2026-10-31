from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import datetime

app = FastAPI(
    title="CrewAI Multi-Agent Backend",
    description="FastAPI backend server featuring 10 POST routes for agent tasks and orchestration.",
    version="1.0.0",
)


class AgentRequest(BaseModel):
    prompt: str = Field(..., description="Task prompt or input instruction for the agent")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional contextual data")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution parameters or hyper-parameters")


class AgentResponse(BaseModel):
    agent_id: str
    status: str
    input_received: AgentRequest
    result: str
    timestamp: str


def handle_agent_task(agent_num: int, request: AgentRequest) -> AgentResponse:
    return AgentResponse(
        agent_id=f"agent{agent_num}",
        status="success",
        input_received=request,
        result=f"Agent {agent_num} executed task successfully for prompt: '{request.prompt}'",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@app.get("/")
def root():
    return {
        "message": "CrewAI Multi-Agent API is running.",
        "routes": [f"/agent{i}" for i in range(1, 11)],
    }


@app.post("/agent1", response_model=AgentResponse, tags=["Agents"])
def agent1_endpoint(request: AgentRequest):
    """POST route for Agent 1."""
    return handle_agent_task(1, request)


@app.post("/agent2", response_model=AgentResponse, tags=["Agents"])
def agent2_endpoint(request: AgentRequest):
    """POST route for Agent 2."""
    return handle_agent_task(2, request)


@app.post("/agent3", response_model=AgentResponse, tags=["Agents"])
def agent3_endpoint(request: AgentRequest):
    """POST route for Agent 3."""
    return handle_agent_task(3, request)


@app.post("/agent4", response_model=AgentResponse, tags=["Agents"])
def agent4_endpoint(request: AgentRequest):
    """POST route for Agent 4."""
    return handle_agent_task(4, request)


@app.post("/agent5", response_model=AgentResponse, tags=["Agents"])
def agent5_endpoint(request: AgentRequest):
    """POST route for Agent 5."""
    return handle_agent_task(5, request)


@app.post("/agent6", response_model=AgentResponse, tags=["Agents"])
def agent6_endpoint(request: AgentRequest):
    """POST route for Agent 6."""
    return handle_agent_task(6, request)


@app.post("/agent7", response_model=AgentResponse, tags=["Agents"])
def agent7_endpoint(request: AgentRequest):
    """POST route for Agent 7."""
    return handle_agent_task(7, request)


@app.post("/agent8", response_model=AgentResponse, tags=["Agents"])
def agent8_endpoint(request: AgentRequest):
    """POST route for Agent 8."""
    return handle_agent_task(8, request)


@app.post("/agent9", response_model=AgentResponse, tags=["Agents"])
def agent9_endpoint(request: AgentRequest):
    """POST route for Agent 9."""
    return handle_agent_task(9, request)


@app.post("/agent10", response_model=AgentResponse, tags=["Agents"])
def agent10_endpoint(request: AgentRequest):
    """POST route for Agent 10."""
    return handle_agent_task(10, request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
