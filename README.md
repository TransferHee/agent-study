# agent-study

A study repository for learning AI agent with `langchain` and `langgraph`.

## Structure

- `langchain_agents/` — langchain-based examples (starting from `01_`, numbered in learning order)
- `langgraph_agents/` — langgraph-based examples (starting from `01_`, numbered in learning order)

Each file is a standalone script covering a single concept or pattern.

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate

pip install langchain langchain-openai langchain-google-genai langgraph openai pydantic python-dotenv requests
```

Create a `.env` file in the project root and fill in the required API keys.

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

## Usage

Run each script individually.

```bash
python3 langchain_agents/01_responses_api_basic.py
python3 langgraph_agents/01_langgraph_basic.py
```

## Contents

### langchain_agents

| File | Topic |
|---|---|
| `01_responses_api_basic.py` | Basic usage of the OpenAI Responses API |
| `02_langchain_basic.py` | langchain basics |
| `03_structured_output.py` | Structured output |
| `04_memory.py` | Memory basics |
| `11_agent.py` | Basic agent structure |
| `12_agent_with_memory.py` | Short-term memory with checkpointer |
| `13_agent_with_middleware.py` | Applying middleware |
| `14_agent_with_middleware_pii.py` | PII-masking middleware |
| `15_agent_with_structured.py` | Agent with structured output |
| `16_agent_with_hook.py` | Using built-in hooks |
| `17_agent_with_custom_hook.py` | Implementing a custom hook |
| `18_agent_with_guardrail.py` | Applying a guardrail |
| `19_agent_with_long_memory.py` | Long-term memory with store |

### langgraph_agents

| File | Topic |
|---|---|
| `01_langgraph_basic.py` | langgraph basics |
| `02_practice_email.py` | Email processing practice |
| `03_practice_email2.py` | Email processing practice (advanced) |
| `04_principles_of_state.py` | Principles of state design |
| `11_prompt_chaining.py` | Prompt chaining pattern |
| `12_parallelization.py` | Parallelization pattern |
| `13_routing.py` | Routing pattern |
| `14_loop.py` | Loop pattern |
| `15_orchestrator.py` | Orchestrator pattern |

## Acknowledgements

- https://wikidocs.net/book/19240
