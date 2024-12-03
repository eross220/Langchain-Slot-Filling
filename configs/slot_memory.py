import copy
import json
from typing import Any, Dict, List, ClassVar
from pydantic import Field
from datetime import datetime
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.entity import BaseEntityStore, InMemoryEntityStore
from langchain.memory.utils import get_prompt_input_key
from langchain.prompts.base import BasePromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import get_buffer_string, BaseMessage
from  configs import prompt
# define the slots dict and assign to null
SLOT_DICT = {"type_of_car": "null", "fuel_of_car": "null", "color_of_car": "null"}


class SlotMemory(BaseChatMemory):
    
    llm: BaseChatModel
    slot_extraction_prompt: BasePromptTemplate = prompt.SLOT_EXTRACTION_PROMPT
    k: int = 10
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    chat_history_key: str = "history"
    slot_key: str = "slots"
    return_messages: bool = False
    default_slots :  ClassVar[Dict[str, str]] = SLOT_DICT 
    current_slots : ClassVar[Dict[str, str]] = copy.deepcopy( SLOT_DICT )
    entity_store: BaseEntityStore = Field(default_factory=InMemoryEntityStore)
    inform_check: bool = False


    @property
    def buffer(self) -> list[BaseMessage]:
        """String buffer of memory."""
        if self.return_messages:
            return self.chat_memory.messages
        else:
            return get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.slot_key, self.chat_history_key]
    def information_check(self):
        self.inform_check = True
        for value in self.current_slots.values():
            if value == "null":
                self.inform_check = False
                break

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return history buffer."""
        print("======== Execute load_memory_variables!!! =======")
        buffer_string = get_buffer_string(
            self.chat_memory.messages[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        print("buffer_string", buffer_string)
        
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        slots = self.current_slots
        chain = LLMChain(llm=self.llm, prompt=self.slot_extraction_prompt)
        output = chain.predict(
            history=buffer_string, input=inputs[prompt_input_key], slots=slots
        )
        print("Slot Output:", output)
        output = output.replace("None", "null")
        try:
            output_json = json.loads(output)
        except Exception:
            print(f"error output: {output}")
            output_json = slots
        for k, v in output_json.items():
            if v is not None and v != "null":
                self.current_slots[k] = v
        # print(f"current slots: {self.current_slots}")
                
        self.information_check()
        
        return {
            self.chat_history_key: buffer_string,
            self.slot_key: str(self.current_slots),
            prompt_input_key: inputs[prompt_input_key]
        }

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        print("============ Save Context ==========", inputs, outputs)
        super().save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear memory contents."""
        print("============ Clear Context ==========")

        self.chat_memory.clear()
        self.entity_store.clear()
        self.current_slots = copy.deepcopy(self.default_slots)
