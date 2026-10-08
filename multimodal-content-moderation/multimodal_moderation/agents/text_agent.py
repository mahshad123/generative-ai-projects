from pydantic_ai import Agent
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import ModerationResult, TextModerationResult


MODERATION_INSTRUCTIONS = """
<context>
At ACME enterprise we strive for a friendly but professional interaction with our customers.
</context>

<role>
You are a customer service reviewer at ACME enterprise. You make sure that the customer
service interactions are friendly and professional.
</role>

<input>
You will receive a message from the customer representative towards the customer.
</input>

<instructions>
Detect if:
- the tone of the message is unfriendly
- the tone of the message is unprofessional
- the message contains any personally-identifiable information (PII)
</instructions>

<output>
Provide a detailed rationale for your choices as well as a confidence score between 0 and 1 on your assessment.
</output>
"""


text_moderation_agent = Agent(
    instructions=MODERATION_INSTRUCTIONS,
    output_type=TextModerationResult,
)


async def moderate_text(model_choice: ModelChoice, text: str) -> TextModerationResult:
    result = await text_moderation_agent.run(
        f"Please moderate the following customer service message:\n\n{text}",
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )
    return result.output
