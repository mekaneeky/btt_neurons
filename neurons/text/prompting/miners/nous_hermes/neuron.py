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
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import neuron_config
class HermesMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass


    def __init__( self ):
        super( HermesMiner, self ).__init__()
        print ( self.config )

        bittensor.logging.info( 'Loading ' + str(neuron_config["model_name"]))
        self.tokenizer = AutoTokenizer.from_pretrained( neuron_config["model_name"], use_fast=False )
        self.model = AutoModelForCausalLM.from_pretrained( neuron_config["model_name"], torch_dtype = torch.float16, low_cpu_mem_usage=True )
        bittensor.logging.info( 'Model loaded!' )

        if neuron_config["device"] != "cpu":
            self.model = self.model.to( neuron_config["device"] )

    def _process_history(self, history: List[str]) -> str:
        processed_history = ''

        if neuron_config["do_prompt_injection"]:
            processed_history += neuron_config["system_prompt"]

        for message in history:
            if message['role'] == 'system':
                if not neuron_config["do_prompt_injection"] or message != history[0]:
                    processed_history += '' + message['content'].strip() + ' '

            if message['role'] == 'Assistant':
                processed_history += '### Response:\n' + message['content'].strip() + '[/INST]' #not needed w hermes?
            if message['role'] == 'user':
                processed_history += '### Instruction:\n' + message['content'].strip() + ' '
        return processed_history

    def forward(self, messages: List[Dict[str, str]]) -> str:

        history = self._process_history(messages)
        prompt = history + "ASSISTANT:"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(neuron_config["device"])

        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + neuron_config["max_new_tokens"],
            temperature=neuron_config["temperature"],
            do_sample=neuron_config["do_sample"],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generation = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Logging input and generation if debugging is active
        bittensor.logging.debug("Message: " + str(messages))
        bittensor.logging.debug("Generation: " + str(generation))
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    HermesMiner().run()