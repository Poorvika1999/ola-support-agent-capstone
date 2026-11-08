import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ModelInfo
from autogen_ext.models.replay import ReplayChatCompletionClient
from pydantic import BaseModel


class VerdictModel(BaseModel):
    approved: bool
    risk_level: str
    rationale: str


async def main():
    reviewer = AssistantAgent(
        name="policy_reviewer",
        model_client=ReplayChatCompletionClient(["The response cites KB10 correctly and is grounded. No concerns."]),
        system_message="You review support responses.",
    )

    verdict_client = ReplayChatCompletionClient(
        ['{"approved": true, "risk_level": "low", "rationale": "Grounded and consistent with policy."}'],
        model_info=ModelInfo(vision=False, function_calling=False, json_output=True, structured_output=True, family="unknown"),
    )
    verdict_agent = AssistantAgent(
        name="verdict_agent",
        model_client=verdict_client,
        system_message="You output the final structured verdict.",
        output_content_type=VerdictModel,
    )

    team = RoundRobinGroupChat(
        participants=[reviewer, verdict_agent],
        max_turns=2,
        custom_message_types=[StructuredMessage[VerdictModel]],
    )

    result = await team.run(task="Review this Ola support response for compliance.")
    for msg in result.messages:
        print(type(msg).__name__, "-", getattr(msg, "source", "?"), "->", getattr(msg, "content", None))


asyncio.run(main())
