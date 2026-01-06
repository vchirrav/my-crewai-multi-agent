# CrewAI Multi-Agent Router System

A practical example of building intelligent multi-agent systems using CrewAI with local LLM integration. This project demonstrates an agentic router pattern that classifies user intent and delegates tasks to specialized agents.

## Author

**Viswanath S Chirravuri**
Email: vchirrav@gmail.com

## Overview

This project showcases a multi-agent architecture where:
- A **Router Agent** classifies incoming requests as either FILE or MATH operations
- A **File Agent** handles file existence checks and content reading
- A **Math Agent** performs mathematical calculations

All agents run on a local Llama 3.1 model via Ollama, demonstrating how to build cost-effective AI systems without relying on cloud APIs.

## Key Features

- **Local LLM Integration**: Uses Ollama with Llama 3.1 model (no API costs)
- **Agentic Router Pattern**: Intelligent request classification and routing
- **Custom Tools**: Simplified file checker and calculator tools
- **Hallucination Prevention**: Strict temperature (0.0) and stop tokens to prevent AI rambling
- **Stateless Architecture**: Each request is processed independently without memory
- **Interactive CLI**: User-friendly command-line interface with graceful exit handling

## Architecture

```
User Input
    ↓
Router Agent (Classifier)
    ↓
    ├─→ FILE Intent → File Agent → CheckLocalFile Tool
    └─→ MATH Intent → Math Agent → Calculate Tool
```

## Dependencies

This project uses the following main dependencies:

- **crewai** (>=1.7.2): Multi-agent orchestration framework
- **crewai-tools** (>=1.7.2): Tool utilities for CrewAI agents
- **litellm** (>=1.75.3): LLM integration layer
- **apscheduler** (>=3.11.2): Task scheduling
- **fastapi** (>=0.128.0): Web framework (for potential API extensions)
- **fastapi-sso** (>=0.17.0): SSO integration
- **email-validator** (>=2.3.0): Email validation utilities

For the complete list, see [pyproject.toml](pyproject.toml).

## Prerequisites

1. **Python 3.12+** installed on your system
2. **Ollama** installed and running locally
3. **Llama 3.1 model** pulled in Ollama

### Installing Ollama and Llama 3.1

```bash
# Install Ollama (visit https://ollama.ai for your platform)

# Pull Llama 3.1 model
ollama pull llama3.1

# Verify Ollama is running (default: http://localhost:11434)
ollama serve
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/vchirrav/my-crewai-multi-agent.git
cd my-crewai-multi-agent
```

2. Install dependencies using pip or uv:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

## Usage

Run the interactive CLI:

```bash
python main.py
```

### Example Interactions

**File Operation:**
```
Enter request (e.g. 'read main.py', '50 * 2'): Does the file README.md exist in the folder?

🤖 [Router] Analyzing: 'Does the file README.md exist in the folder?'...
👉 Intent: FILE

[Agent processing...]

📝 RESULT: YES, File Found. Content (first 500 chars): # my-crewai-multi-agent
```

**Math Operation:**
```
Enter request (e.g. 'read main.py', '50 * 2'): Can you tell me how much is forty times fifty?

🤖 [Router] Analyzing: 'Can you tell me how much is forty times fifty?'...
👉 Intent: MATH

[Agent processing...]

📝 RESULT: 2000
```

**Exit:**
```
Enter request (e.g. 'read main.py', '50 * 2'): bye

👋 Exiting program. Goodbye!
```

## Code Structure

### 1. LLM Configuration ([main.py:8-13](main.py#L8-L13))

```python
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    temperature=0.0,
    stop=["<|eot_id|>", "Observation:"]
)
```

Strict configuration prevents hallucinations and keeps responses focused.

### 2. Custom Tools ([main.py:19-59](main.py#L19-L59))

**SimpleFileTool**: Checks file existence in the current directory and reads content (truncated to 500 chars).

**SimpleMathTool**: Evaluates mathematical expressions safely using Python's `eval()`.

### 3. Agent Functions ([main.py:63-127](main.py#L63-L127))

- `get_router_decision()`: Classifies intent as FILE or MATH
- `run_file_crew()`: Executes file checking workflow
- `run_math_crew()`: Executes calculation workflow

### 4. Main Loop ([main.py:130-167](main.py#L130-L167))

Interactive CLI with graceful exit handling and error management.

## Sample Output

The system provides detailed agent traces showing:
- Agent role and task
- Tool execution with input/output
- Final answers

### Detailed Trace Example

When `verbose=True` is set on agents, you'll see comprehensive execution details:

```
╭─────────────────── 🤖 Agent Started ───────────────────╮
│  Agent: FileChecker                                     │
│  Task: User Input: Does the file README.md exist...    │
╰─────────────────────────────────────────────────────────╯

╭─────────────────── 🔧 Agent Tool Execution ────────────╮
│  Agent: FileChecker                                     │
│  Thought: I need to check if the file README.md exists │
│  Using Tool: CheckLocalFile                            │
╰─────────────────────────────────────────────────────────╯

╭─────────────────── Tool Input ─────────────────────────╮
│  {                                                      │
│    "file_name": "README.md"                            │
│  }                                                      │
╰─────────────────────────────────────────────────────────╯

╭─────────────────── Tool Output ────────────────────────╮
│  YES, File Found. Content (first 500 chars):           │
│  # my-crewai-multi-agent                               │
╰─────────────────────────────────────────────────────────╯

╭─────────────────── ✅ Agent Final Answer ──────────────╮
│  Agent: FileChecker                                     │
│  Final Answer:                                          │
│  YES, File Found. Content (first 500 chars):...        │
╰─────────────────────────────────────────────────────────╯
```

This verbose output is invaluable for:
- Debugging agent reasoning and tool usage
- Understanding how the LLM interprets tasks
- Validating that agents are using tools correctly
- Learning how to improve prompts and tool descriptions

See [sample.output](sample.output) for complete execution traces with both FILE and MATH operations.

## Design Principles

1. **Hallucination Prevention**: Zero temperature and stop tokens prevent AI from generating unreliable outputs
2. **Path Safety**: File tool restricts operations to current directory only
3. **Stateless Design**: `memory=False` ensures each request is independent
4. **Clear Separation**: Each agent has a single, focused responsibility
5. **Truncation**: File content limited to 500 chars to prevent context overflow

## Extending the System

To add new capabilities:

1. **Create a new custom tool** inheriting from `BaseTool`
2. **Define a new agent function** similar to `run_file_crew()` or `run_math_crew()`
3. **Update the router** to recognize the new intent type
4. **Add routing logic** in the main loop

Example:
```python
# New tool
class WebSearchTool(BaseTool):
    name: str = "SearchWeb"
    description: str = "Searches the web for information"

    def _run(self, query: str) -> str:
        # Implementation
        pass

# New agent
def run_search_crew(input_prompt):
    search_agent = Agent(
        role='WebSearcher',
        goal='Search for information',
        tools=[web_search_tool],
        llm=local_llm
    )
    # ... task and crew setup
```

## Troubleshooting

**Ollama Connection Error:**
- Ensure Ollama is running: `ollama serve`
- Check if Llama 3.1 is installed: `ollama list`

**Slow Response:**
- Local LLM processing depends on hardware
- Consider using a smaller model for faster inference

**Agent Not Using Tools:**
- Check tool descriptions are clear
- Verify temperature is low enough
- Review stop tokens

## License

This project is provided as a reference implementation for learning CrewAI and multi-agent systems.

## Contributing

Feel free to fork this repository and experiment with different:
- LLM models (GPT-4, Claude, other local models)
- Agent configurations
- Custom tools
- Routing strategies

## References

- [CrewAI Documentation](https://docs.crewai.com/)
- [Ollama](https://ollama.ai/)
- [Llama Models](https://ai.meta.com/llama/)

## Contact

For questions or feedback, reach out to:
- **Viswanath S Chirravuri**: vchirrav@gmail.com
