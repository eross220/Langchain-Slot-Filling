from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


_DEFAULT_SLOT_EXTRACTION_TEMPLATE = """
You are an AI assistant, reading the transcript of a conversation between an AI and a human.
From the last line of the conversation, extract all proper named entity(here denoted as slots) that match about car.
The named entity tag required for a restaurant_reservation are type_of_car, fuel_of_car, and color_of_car.

The output should be returned in the following json format.
{{
    "type_of_car": "Define car type identified from the conversation."
    "fuel_of_car": "Define the fuel of car identified from the conversation."
    "color_of_car": "Define color  identified from the conversation."
}}

If there is no match for each slot, assume null.(e.g., user is simply saying hello or having a brief conversation).

EXAMPLE
Conversation history:
Person #1: a new project has started
AI: "Good luck!"
Current Slots: {{"type_of_car": null, "fuel_of_car": null, "color_of_car": null}}
Last line:
Person #1: We are considering confirm the car information with project members as a way to get to know each other better.
Output Slots: {{"type_of_car": null, "fuel_of_car": null, "color_of_car": null}}
END OF EXAMPLE

EXAMPLE
Conversation history:
Person #1: A new project has started.
AI: "Good luck!"
Person #1: We are considering confirm the car information as a way to get to know each other better.
AI: " What sort of car are you looking for?"
Current Slots: {{"type_of_car": null, "fuel_of_car": null, "color_of_car": null}}
Last line:
Person #1: truck
Output Slots: {{"type_of_car": "truck", "fuel_of_car": null, "color_of_car": null}}
END OF EXAMPLE

EXAMPLE
Conversation history:
Person #1: Red is good
AI: What sort of fuel type do you want?
Current Slots: {{"type_of_car": null, "fuel_of_car": null, "color_of_car": null}}
Last line:
Person #1: Petrol
Output Slots:  {{"type_of_car": null, "fuel_of_car": "Petrol", "color_of_car": "Red"}}
END OF EXAMPLE

Output Slots must be in json format!

Begin!
Conversation history (for reference only):
{history}
Current Slots:
{slots}
Last line of conversation (for extraction):
Human: {input}


Output Slots:
"""

SLOT_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input", "slots"],
    template=_DEFAULT_SLOT_EXTRACTION_TEMPLATE,
)

__DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context.
If the AI does not know the answer to a question, it truthfully says it does not know.

If type_of_car is null with respect to the Current Slots value, ask a question about the type of car.
However, do not use the word "type_of_car" directly. Use expressions that are natural conversational expressions.

If fuel_of_car is null with respect to the Current Slots value, ask a question about the fuel of car.
However, do not use the word "fuel_of_car" directly. Use expressions that are natural conversational expressions.

If color_of_car is null with respect to the Current Slots value, ask a question about the color of car.
However, do not use the word "color_of_car" directly. Use expressions that are natural conversational expressions.

If current slots does not contain null, AI output should be output Finish[yes].
EXAMPLE
Conversation history:
Human:Truck is ok
AI: Okay.  what color do you want?
Current Slots: {{"type_of_car": Truck, "fuel_of_car": Petrol, "color_of_car": Red }}
Last line:
Human:Red
AI: Finish[yes]
END OF EXAMPLE

You don't have to output about Human's answer.
You don't have to output about Current Slots.

Begin!
Current conversation:
{history}
Current Slots:
{slots}
Human: {input}
AI:"""

CHAT_PROMPT = PromptTemplate(input_variables=["history", "input", "slots"], template=__DEFAULT_TEMPLATE)
