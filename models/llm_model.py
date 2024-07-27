import os
from types import SimpleNamespace
from llama_cpp import Llama

from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)
class LLMModel():
    def __init__(self, args):
        with new_cd(parent_dir):
            self.llm = Llama(
                model_path = f"../checkpoints/{args.llm_version}",
                chat_format = "llama-3",
                n_ctx = args.max_length,
                verbose = False
                )
        self.args = args
        
        
    def chat(self, prompt, system_message="You are an helpful assistant who help users with their questions."):
        response = self.llm.create_chat_completion(
            messages = [
                {"role": "system", 
                 "content": f"{system_message}"},
                {"role": "user",
                 "content": f"{prompt}"}]
            )
        response = SimpleNamespace(content = response['choices'][0]['message']['content'])
        return response
        
