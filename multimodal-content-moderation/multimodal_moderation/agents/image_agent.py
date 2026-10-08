from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import ImageModerationResult


MODERATION_INSTRUCTIONS = """
CONTEXT
At ACME enterprise we strive for a friendly but professional interaction with our customers.

ROLE
You are a customer service reviewer at ACME enterprise. You make sure that the customer 
service interactions are friendly and professional.

TASK
You will receive an image. Determine if that image is appropriate for a professional customer service setting
according to the instructions below.

INSTRUCTIONS
Detect if:
- the image is disturbing in any way
- the image contains a person, a part of a person, or any other personally-identifiable information (PII). If yes, set
    contains_pii to True.
- the image is of low quality (blurry, pixelated, underexposed, overexposed, etc.)


OUTPUT
Provide a detailed rationale for your choices.
"""

image_moderation_agent = Agent(
    instructions=MODERATION_INSTRUCTIONS,
    output_type=ImageModerationResult,
)



async def moderate_image(
    model_choice: ModelChoice,
    image_source: bytes,
    media_type: str
) -> ImageModerationResult:
    image_input = BinaryContent(data=image_source, media_type=media_type)

    result = await image_moderation_agent.run(
        ["Please moderate the following image.", image_input],
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )
    return result.output