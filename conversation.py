from langchain_openai import ChatOpenAI
from chains.slot_filling import SlotFilling
from configs.slot_memory import SlotMemory
import langchain
from langchain.chains.conversation.base import ConversationChain
from configs.prompt import CHAT_PROMPT, SLOT_EXTRACTION_PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import get_buffer_string, BaseMessage
from langchain.chains.llm import LLMChain
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory

# langchain.debug = True
store = {}

llm = ChatOpenAI(temperature=0.7, api_key= "" )
slot_memory = SlotMemory(llm = llm)
chain = LLMChain(llm=llm, prompt=SLOT_EXTRACTION_PROMPT)

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    # print("memory variable count:",len(slot_memory.memory_variables))
    # key = slot_memory.memory_variables[1]
    # print(slot_memory.load_memory_variables({"input":"", "history":""}))
    # messages = slot_memory.load_memory_variables({})[key]
    # store[session_id] = InMemoryChatMessageHistory(messages=messages)
    return store[session_id]

query = "Hi"
input_dict = slot_memory.load_memory_variables({"input":query, "slots":slot_memory.current_slots})
print("input_dict",input_dict)

runnable = CHAT_PROMPT | llm
chain=  RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


inputs = dict( {"input":query}, **input_dict)
print(inputs)
response= chain.invoke(
    inputs,
    config = {"configurable": {"session_id": "foo"}}
)
outputs = { "response": response.content }
slot_memory.save_context(inputs, outputs)




query1 = "I am looking for truck of red color"

input_dict1 = slot_memory.load_memory_variables({"input": query1,"slots":""})
print("input_dict1",input_dict1)
inputs1 = dict( {"input":query1}, **input_dict1 )
response1 = chain.invoke(
    inputs1,
    config = {"configurable": {"session_id": "foo"}}
)
outputs1 = { "response": response1.content }
slot_memory.save_context(inputs, outputs1)
print(slot_memory.current_slots)













# print(chain.predict(input = "Hi"))
# print(chain.predict(input="I am looking for truck of red color"))
# print(chain.predict(input="Petrol"))
