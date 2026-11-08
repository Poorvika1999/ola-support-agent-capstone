import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM

_CALL_COUNT = [0]


class DiagnosticLLM(BaseLLM):
    model: str = "diagnostic"

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
              from_task=None, from_agent=None, response_model=None):
        _CALL_COUNT[0] += 1
        role = getattr(from_agent, "role", "?")
        print(f"\n===== CALL #{_CALL_COUNT[0]} (agent role={role}) =====")
        for i, m in enumerate(messages):
            print(f"--- [{i}] {m.get('role')} ---")
            print(m.get("content", ""))
        return "Thought: done\nFinal Answer: OK from " + role


llm = DiagnosticLLM(model="diagnostic")

agent1 = Agent(role="Agent One", goal="g1", backstory="b1", llm=llm, allow_delegation=False, verbose=False)
agent1b = Agent(role="Agent OneB", goal="g1b", backstory="b1b", llm=llm, allow_delegation=False, verbose=False)
agent2 = Agent(role="Agent Two", goal="g2", backstory="b2", llm=llm, allow_delegation=False, verbose=False)

task1 = Task(description="Produce fact A.", expected_output="fact A", agent=agent1)
task1b = Task(description="Produce fact B.", expected_output="fact B", agent=agent1b)
task2 = Task(description="Compose final reply using upstream facts.", expected_output="final reply",
             agent=agent2, context=[task1, task1b])

crew = Crew(agents=[agent1, agent1b, agent2], tasks=[task1, task1b, task2], process=Process.sequential, verbose=False)
crew.kickoff()
