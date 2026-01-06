import os
import sys
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# 1. SETUP LOCAL LLM
# We use a very strict configuration to prevent hallucinations
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    temperature=0.0,
    stop=["<|eot_id|>", "Observation:"] # Stop it from rambling
)

# 2. DEFINE CUSTOM SIMPLIFIED TOOLS

# --- CUSTOM TOOL 1: Simple File Checker ---
# We wrap the logic here so the AI has an easier job.
class SimpleFileTool(BaseTool):
    name: str = "CheckLocalFile"
    description: str = (
        "Checks if a file exists in the current working directory and reads it. "
        "Input should be ONLY the filename (e.g., 'readme.md' or 'data.txt'). "
        "Do not provide full paths."
    )

    def _run(self, file_name: str) -> str:
        # Clean input
        file_name = file_name.replace('"', '').replace("'", "").strip()
        
        # Force strict current directory usage to prevent path hallucinations
        current_dir = os.getcwd()
        full_path = os.path.join(current_dir, file_name)
        
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Truncate if too long to prevent context overflow
                    return f"YES, File Found. Content (first 500 chars):\n{content[:500]}"
            except Exception as e:
                return f"Error reading file: {e}"
        else:
            return f"NO, File does not exist at: {full_path}"

# --- CUSTOM TOOL 2: Math ---
class SimpleMathTool(BaseTool):
    name: str = "Calculate"
    description: str = "Evaluates math. Input is a string equation (e.g. '5 * 5')."
    
    def _run(self, operation: str) -> str:
        try:
            return str(eval(operation.replace('"', '').strip()))
        except Exception as e:
            return f"Error: {e}"

# Instantiate tools
file_tool = SimpleFileTool()
math_tool = SimpleMathTool()

# 3. EXECUTION FUNCTIONS (STATELESS)

def get_router_decision(user_input):
    """Decides if the intent is FILE or MATH"""
    print(f"\n🤖 [Router] Analyzing: '{user_input}'...")
    
    router_agent = Agent(
        role='Classifier',
        goal='Classify intent',
        backstory='You output ONLY the word "FILE" or "MATH".',
        verbose=False,
        llm=local_llm,
        memory=False
    )
    
    task = Task(
        description=f"Classify this input: '{user_input}'. Return ONLY 'FILE' or 'MATH'.",
        expected_output='A single word.',
        agent=router_agent
    )
    
    crew = Crew(agents=[router_agent], tasks=[task], verbose=False)
    return crew.kickoff().raw.strip().upper()

def run_file_crew(input_prompt):
    """Runs the File Agent"""
    file_agent = Agent(
        role='FileChecker',
        goal='Check for files',
        backstory='You check for files using the CheckLocalFile tool.',
        tools=[file_tool],
        verbose=True,
        llm=local_llm,
        memory=False
    )
    
    task = Task(
        description=(
            f"User Input: {input_prompt}\n"
            "Extract the filename. Use 'CheckLocalFile' to see if it exists. "
            "If the tool says YES, report it. If NO, report it."
        ),
        expected_output='A yes/no confirmation.',
        agent=file_agent
    )
    
    return Crew(agents=[file_agent], tasks=[task], process=Process.sequential).kickoff()

def run_math_crew(input_prompt):
    """Runs the Math Agent"""
    math_agent = Agent(
        role='Calculator',
        goal='Calculate numbers',
        backstory='You solve math.',
        tools=[math_tool],
        verbose=True,
        llm=local_llm,
        memory=False
    )
    
    task = Task(
        description=f"Solve this: {input_prompt}",
        expected_output='The number.',
        agent=math_agent
    )
    
    return Crew(agents=[math_agent], tasks=[task], process=Process.sequential).kickoff()

# 4. MAIN LOOP
if __name__ == "__main__":
    print("\n-------------------------------------------")
    print(f"   Agentic Router (Working Dir: {os.getcwd()})")
    print("-------------------------------------------\n")
    
    # List of phrases that indicate the user wants to quit
    EXIT_PHRASES = ["exit", "quit", "stop", "bye", "terminate"]

    while True:
        try:
            prompt = input("Enter request (e.g. 'read main.py', '50 * 2'): ")
            
            # --- IMPROVED EXIT CHECK ---
            # Checks if any exit phrase appears in the input OR if the input is empty
            if not prompt.strip() or any(phrase in prompt.lower() for phrase in EXIT_PHRASES):
                print("\n👋 Exiting program. Goodbye!")
                break
            # ---------------------------

            # 1. Router
            intent = get_router_decision(prompt)
            print(f"👉 Intent: {intent}")
            
            # 2. Execution
            if "FILE" in intent:
                result = run_file_crew(prompt)
            elif "MATH" in intent:
                result = run_math_crew(prompt)
            else:
                result = "Could not classify request. Try 'read [filename]' or 'math [equation]'."
            
            print(f"\n📝 RESULT: {result}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Force Exit detected.")
            sys.exit()
        except Exception as e:
            print(f"Error: {e}")