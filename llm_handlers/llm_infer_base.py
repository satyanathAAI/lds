from abc import ABC,abstractmethod


class LlmHandler(ABC):
    
    @abstractmethod
    def get_llm_params(self):
        pass
    @abstractmethod
    def format_prompt(self,prompt_list):
        pass
    @abstractmethod
    def preproces_input(self,formated_prompt):
        pass
    @abstractmethod
    def call_llm(self,prompt_list,filename_list):
        pass
 


