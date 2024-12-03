from langchain.chains import ConversationChain
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from configs.prompt import CHAT_PROMPT
from configs.slot_memory  import SlotMemory

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from typing import Any, Dict, Union

           
class SlotFilling():
    memory: SlotMemory
    llm: BaseChatModel
    store: dict = {} 
    chain: RunnableWithMessageHistory
    def __init__(self, memory: SlotMemory, llm: BaseChatModel):
        self.memory = memory
        self.llm = llm
        runnable = CHAT_PROMPT | self.llm
        self.chain = RunnableWithMessageHistory(
            runnable,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    
    class Config:
        arbitrary_types_allowed = True
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    
    def create(self) -> RunnableWithMessageHistory:
        runnable = CHAT_PROMPT | self.llm
        return RunnableWithMessageHistory(
            runnable,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    def prep_input(self, inputs: Union[ Dict[str, Any], Any]) -> Dict[str, str]:
        input_dict = self.memory.load_memory_variables(inputs)
        print(input_dict)
        return input_dict
    
    def invoke(
        self,
        input:Dict[str, Any],
    ):
        print("Store:", self.store)
        # Prepare input for slot_filling_chain input which has slots, history and input dict 
        inputs = self.prep_input(input)

        response = self.chain.invoke(
            inputs,
            config = {"configurable": {"session_id": "foo"}}
        )
        outputs = { "response": response.content }
        self.memory.save_context(inputs, outputs)
        return response

    def log(self):
        print(f"【Slot】: {self.memory.current_slots}")