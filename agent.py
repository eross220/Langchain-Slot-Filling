from langchain_openai import ChatOpenAI
from chains.slot_filling import SlotFilling
from configs.slot_memory import SlotMemory
import langchain
# langchain.debug = True

llm = ChatOpenAI(temperature=0.7, api_key= "" )
memory = SlotMemory(llm=llm, return_messages= True)
slot_filling = SlotFilling(memory=memory, llm=llm)

slot_filling.invoke( {"input": "hi", "slots": memory.current_slots} )
slot_filling.invoke( {"input": "I am looking for truck of red color", "slots": memory.current_slots} )
slot_filling.invoke( {"input": "Petrol", "slots": memory.current_slots} )
slot_filling.log()
print(slot_filling.memory.current_slots)
print(slot_filling.memory.inform_check)

if slot_filling.memory.inform_check == True:
    slot_filling.store = {}

    slot_filling.invoke( {"input": "hi", "slots": memory.current_slots} )
    slot_filling.invoke( {"input": "I am looking for truck of red color", "slots": memory.current_slots} )
    slot_filling.invoke( {"input": "Petrol", "slots": memory.current_slots} )
    slot_filling.log()

