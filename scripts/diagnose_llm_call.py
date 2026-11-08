"""One-off diagnostic (not part of the submission): runs a minimal CrewAI
agent+task with a throwaway LLM that just dumps what call() receives, so we
can see the exact shape of `messages`, `tools`, and `available_functions`
before writing the real MockLLM.
"""
import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool


@tool("dummy_tool")
def dummy_tool(query: str) -> str:
    """A dummy tool that echoes the query."""
    return f"ECHO: {query}"


_CALL_COUNT = [0]


class DiagnosticLLM(BaseLLM):
    model: str = "diagnostic"

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
              from_task=None, from_agent=None, response_model=None):
        _CALL_COUNT[0] += 1
        n = _CALL_COUNT[0]
        print(f"\n========== CALL #{n} ==========")
        print("messages type:", type(messages))
        if isinstance(messages, list):
            for i, m in enumerate(messages):
                role = m.get("role")
                content = m.get("content", "")
                print(f"--- message[{i}] role={role!r} len={len(content)} ---")
                print(content)
        else:
            print(messages)
        print("tools:", tools)
        print("available_functions:", available_functions)
        print("from_agent role:", getattr(from_agent, "role", None))
        print("from_task:", getattr(from_task, "description", None))

        if n == 1:
            return "Thought: I should use the tool\nAction: dummy_tool\nAction Input: {\"query\": \"hello\"}"
        else:
            return "Thought: I have my answer\nFinal Answer: done"


llm = DiagnosticLLM(model="diagnostic")

agent = Agent(
    role="Diagnostic Agent",
    goal="Diagnose the prompt format",
    backstory="You exist only to reveal the prompt format.",
    tools=[dummy_tool],
    llm=llm,
    allow_delegation=False,
    verbose=False,
)

task = Task(
    description="Look up the word 'banana' using your tool.",
    expected_output="A short confirmation.",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
crew.kickoff()
