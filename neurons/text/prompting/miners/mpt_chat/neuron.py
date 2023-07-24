# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

class Mpt_chatMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--mpt_chat.model_name', type=str, required=False, help='Name/path of model to load' , default="mosaicml/mpt-7b-chat")
        parser.add_argument( '--mpt_chat.tokenizer_name', type=str, required=False, help='Name/path of model to load' , default="EleutherAI/gpt-neox-20b")
        parser.add_argument( '--mpt_chat.device', type=str, help='Device to load model', default="cuda" )
        parser.add_argument( '--mpt_chat.max_new_tokens', type=int, help='Max tokens for model output.', default=256 ) 
        parser.add_argument( '--mpt_chat.temperature', type=float, help='Sampling temperature of model', default=0.5 )
        parser.add_argument( '--mpt_chat.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).' )
        parser.add_argument( '--mpt_chat.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
        parser.add_argument( '--mpt_chat.system_prompt', type=str, help='What prompt to replace the system prompt with', default= "A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. " )
        parser.add_argument( '--mpt_chat.use_triton', action='store_true', default=False, help='Whether to use a triton to speed up inference' )

    def __init__( self ):
        super( Mpt_chatMiner, self ).__init__()
        print ( self.config )
        
        bittensor.logging.info( 'Loading ' + str(self.config.mpt_chat.model_name))
        self.tokenizer = AutoTokenizer.from_pretrained( self.config.mpt_chat.tokenizer_name )
        if self.config.mpt_chat.use_triton:
            config = AutoConfig.from_pretrained(
            'mosaicml/mpt-7b-chat',
            trust_remote_code=True
            )
            config.attn_config['attn_impl'] = 'triton'
            model = AutoModelForCausalLM.from_pretrained(
                self.config.mpt_chat.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained( self.config.mpt_chat.model_name, torch_dtype = torch.float16, low_cpu_mem_usage=True , trust_remote_code=True )
        bittensor.logging.info( 'Model loaded!' )

        if self.config.mpt_chat.device != "cpu":
            self.model = self.model.to( self.config.mpt_chat.device )

    def _process_history(self, history: List[str]) -> str:
        processed_history = """<|im_start|>system\n- You are a helpful assistant AI chatbot excited to help the user.<|im_end|>"""

        if self.config.mpt_chat.do_prompt_injection:
            processed_history += self.config.mpt_chat.system_prompt

        for message in history:
            if message['role'] == 'system':
                if not self.config.mpt_chat.do_prompt_injection or message != history[0]:
                    processed_history += '<|im_start|>system\n-' + message['content'].strip() + '<|im_end|>'

            if message['role'] == 'Assistant':
                processed_history += 'assistant\n' + message['content'].strip() + '<|im_end|>'
            if message['role'] == 'user':
                processed_history += 'user\n' + message['content'].strip() + '<|im_end|>'
        return processed_history

    def forward(self, messages: List[Dict[str, str]]) -> str:

        history = self._process_history(messages)
        prompt = history + "\n### Response:"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.mpt_chat.device)

        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + self.config.mpt_chat.max_new_tokens,
            temperature=self.config.mpt_chat.temperature,
            do_sample=self.config.mpt_chat.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generation = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False).strip()
        generation = generation.split("<|endoftext|>")[0]
        # Logging input and generation if debugging is active
        bittensor.logging.debug("Message: " + str(messages))
        bittensor.logging.debug("Prompt: " + str(prompt))
        bittensor.logging.debug("Generation: " + str(generation.replace("<","-")))
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    Mpt_chatMiner().run()