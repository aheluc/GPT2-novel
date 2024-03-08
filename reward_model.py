import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm.tokenization_chatglm import ChatGLMTokenizer

class Model(nn.Module):
    def __init__(self, model_path, pre_seq_len, device):
        super(Model, self).__init__()
        self.device = device
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.pre_seq_len = pre_seq_len
        config.prefix_projection = False
        self.model = ChatGLMForConditionalGeneration.from_pretrained('./chatglm', config=config, trust_remote_code=True)
        self.model.half().quantize(4).to(device=device) #
        self.model.transformer.prefix_encoder.float()
        
        self.tokenizer = ChatGLMTokenizer.from_pretrained('./chatglm', trust_remote_code=True)
        self.good_ids = [self.tokenizer.convert_tokens_to_ids('并'), self.tokenizer.convert_tokens_to_ids('不')]
        self.bad_ids = [self.tokenizer.convert_tokens_to_ids('难'), self.tokenizer.convert_tokens_to_ids('有')]
        #self.rate = nn.Linear(self.model.transformer.output_layer.in_features, 2, bias=False)
        #with torch.no_grad():
        #    self.rate.weight.data[0] = self.model.transformer.output_layer.weight.clone()[None, self.good_id].float()
        #    self.rate.weight.data[1] = self.model.transformer.output_layer.weight.clone()[None, self.bad_id].float()
        
    def forward(self, inputs1, inputs2):
        return1 = self.model(input_ids=inputs1.input_ids.to(device=self.device), 
            position_ids=inputs1.position_ids.to(device=self.device), 
            attention_mask=inputs1.attention_mask.to(device=self.device),
            return_dict=True,
            output_hidden_states=True)
        return2 = self.model(input_ids=inputs2.input_ids.to(device=self.device), 
            position_ids=inputs2.position_ids.to(device=self.device), 
            attention_mask=inputs2.attention_mask.to(device=self.device),
            return_dict=True,
            output_hidden_states=True)
        #result1 = torch.softmax(return1.logits[:, -1, [*self.good_ids, *self.bad_ids]], dim=-1)[:, :-1].sum(dim=-1).float() # / (4096 ** 0.5
        #result2 = torch.softmax(return2.logits[:, -1, [*self.good_ids, *self.bad_ids]], dim=-1)[:, :-1].sum(dim=-1).float()
        result1 = torch.cat((return1.logits[:, -1, [*self.good_ids]].sum(dim=-1, keepdim=True), 
            return1.logits[:, -1, [*self.bad_ids]].sum(dim=-1, keepdim=True)), dim=-1) / 2
        result2 = torch.cat((return2.logits[:, -1, [*self.good_ids]].sum(dim=-1, keepdim=True), 
            return2.logits[:, -1, [*self.bad_ids]].sum(dim=-1, keepdim=True)), dim=-1) / 2
        result1 = torch.softmax(result1, dim=-1)[:, :1]
        result2 = torch.softmax(result2, dim=-1)[:, :1]
        return result1, result2
    
    def get_reward(self, inputs):
        return_ = self.model(input_ids=inputs1.input_ids.to(device=self.device), 
            position_ids=inputs1.position_ids.to(device=self.device), 
            attention_mask=inputs1.attention_mask.to(device=self.device),
            return_dict=True,
            output_hidden_states=True)
        return return_.logits[:, -1, [*self.good_ids]].sum(dim=-1, keepdim=True) / 2