# FastAPI Multi-Agent Backend

A FastAPI backend server exposing 10 agent POST endpoints: `/agent1` through `/agent10`.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
or:
```bash
python main.py
```

- **Interactive API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Endpoints

- `GET /`: Health status and list of all 10 agent routes.
- `POST /agent1` to `POST /agent10`: Agent execution endpoints.

### Example Request Body:
```json
{
  "prompt": "Analyze dataset and generate summary",
  "context": {
    "user_id": "123"
  },
  "parameters": {
    "temperature": 0.7
  }
}
```

### Example Response:
```json
{
  "agent_id": "agent1",
  "status": "success",
  "input_received": {
    "prompt": "Analyze dataset and generate summary",
    "context": {
      "user_id": "123"
    },
    "parameters": {
      "temperature": 0.7
    }
  },
  "result": "Agent 1 executed task successfully for prompt: 'Analyze dataset and generate summary'",
  "timestamp": "2026-10-24T12:52:00.000000+00:00"
}
```

## Running Tests

```bash
pytest -v
```
