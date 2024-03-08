from itertools import permutations
import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoConfig
from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm.tokenization_chatglm import ChatGLMTokenizer
from reward_model import Model
import json

device = 'cuda:1'

'''
config = AutoConfig.from_pretrained('./chatglm', trust_remote_code=True)
config.pre_seq_len = 1
config.prefix_projection = False
model = ChatGLMForConditionalGeneration.from_pretrained('./chatglm', config=config, trust_remote_code=True)
model = model.half().quantize(4).to(device=device) #
model.transformer.prefix_encoder.float()
#model.transformer.last_embedding.float()
tokenizer = ChatGLMTokenizer.from_pretrained('./chatglm', trust_remote_code=True)
'''

model = Model('./chatglm', pre_seq_len=64, device=device).to(device=device)
optim = optim.AdamW(model.parameters(), lr=1e-2)

data = []
K = 3
data_ids = list(range(K))
pairs = list(permutations(data_ids, 2))

with open('./ptuning-data/ptuning-raw-train-data.txt', 'r', encoding='utf-8') as f:
    batch_str = ''
    for i, line in enumerate(f.readlines()):
        batch_str += line
        if i % 2 == 1:
            batch_obj = eval(batch_str)
            data.append(batch_obj)
            batch_str = ''

test_data = []
with open('./ptuning-data/ptuning-raw-test-data.txt', 'r', encoding='utf-8') as f:
    batch_str = ''
    for i, line in enumerate(f.readlines()):
        batch_str += line
        if i % 2 == 1:
            batch_obj = eval(batch_str)
            test_data.append(batch_obj)
            batch_str = ''

prompt = "“{}”\n以上文字难以理解吗？" # 从语法、信息量和连贯性的角度来说  难以理解
batch_size = 8
loss_fn = nn.BCELoss()
grad_accum = 4
iter_ = 0

def test():
    model.eval()
    start = 0
    n_correct = 0
    n_total = 0
    with open('./test_result.txt', 'w', encoding='utf-8') as f:
        while start < len(test_data):
            end = start + batch_size if start + batch_size < len(test_data) else len(test_data)
            batch = test_data[start:end]
            n_total += 2 * len(batch)
            inputs1 = [prompt.format(l) for l, _ in batch]
            inputs1 = [model.tokenizer.build_prompt(input_) + '以上文字' for input_ in inputs1]
            inputs1 = model.tokenizer(inputs1, padding=True, return_tensors='pt')
            inputs2 = [prompt.format(r) for _, r in batch]
            inputs2 = [model.tokenizer.build_prompt(input_) + '以上文字' for input_ in inputs2]
            inputs2 = model.tokenizer(inputs2, padding=True, return_tensors='pt')
            with torch.no_grad():
                probs1, probs2 = model(inputs1, inputs2)
            n_correct += (probs1 < 0.5).sum().item()
            n_correct += (probs2 > 0.5).sum().item()
            start += batch_size
            for i, s in enumerate(batch):
                s1, s2 = s
                f.write('[{}, {},\n'.format(probs1.view(-1)[i], json.dumps(s1, ensure_ascii=False)))
                f.write('    {}, {},]\n'.format(probs2.view(-1)[i], json.dumps(s2, ensure_ascii=False)))
    acc = n_correct / n_total
    print('acc', acc)
    return acc

best_acc = test()
for epoch in range(3):
    print('epoch', epoch + 1)
    random.shuffle(data)
    start = 0
    model.train()
    while start < len(data):
        iter_ += 1
        end = start + batch_size if start + batch_size < len(data) else len(data)
        batch = data[start:end]
        inputs1 = [prompt.format(l) for l, _ in batch]
        inputs1 = [model.tokenizer.build_prompt(input_) + '以上文字' for input_ in inputs1]
        inputs1 = model.tokenizer(inputs1, padding=True, return_tensors='pt')
        inputs2 = [prompt.format(r) for _, r in batch]
        inputs2 = [model.tokenizer.build_prompt(input_) + '以上文字' for input_ in inputs2]
        inputs2 = model.tokenizer(inputs2, padding=True, return_tensors='pt')
        probs1, probs2 = model(inputs1, inputs2)
        #print(probs1, probs2)
        labels_neg = torch.zeros(len(batch)).float().to(device=device)
        labels_pos = torch.ones(len(batch)).float().to(device=device)
        loss1 = loss_fn(probs1.view(-1), labels_neg)
        loss2 = loss_fn(probs2.view(-1), labels_pos)
        loss = (loss1 + loss2) / 2
        loss = loss / grad_accum
        print(iter_, loss.item())
        loss.backward()
        if iter_ % grad_accum == 0:
            optim.step()
            optim.zero_grad()
        start += batch_size
    acc = test()
    if acc > best_acc:
        print('save')
        best_acc = acc
        torch.save(model.model.transformer.prefix_encoder.state_dict(), 'ptuning-pre64-4tk-1e-2-logits-sum.param')
    else:
        print('early stopping...')
        break