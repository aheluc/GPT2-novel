import transformers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json
import random
import numpy as np
import argparse
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
#from tokenizations.bpe_tokenizer import get_encoder
import modeling_gpt_2jigen
from transformers import AutoTokenizer, AutoConfig
from reward_model import Model
from safetensors.torch import load_file
import math

glm_prompt = "“{}”\n以上文字难以理解吗？"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='./model/config.json', type=str, required=False,
                        help='选择模型参数')
    #parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    #parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    #parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率') #1.5e-4
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    #parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='./checkpoints/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='./model', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--prefix_path', default='./ptuning.param', type=str, help='P-Tuning微调后的前缀存放路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    #parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    #parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")
    parser.add_argument('--n_samples', default=4, type=int, required=False, help='一次采样的batch数')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    model_config = transformers.models.gpt2.configuration_gpt2.GPT2Config.from_json_file(args.model_config)
    n_ctx = model_config.n_ctx
    from transformers import BertTokenizer
    full_tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    full_tokenizer.max_len = 999999
    device = 'cuda:2' #if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    tokenized_data_path = args.tokenized_data_path
    #raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    #num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    model = modeling_gpt_2jigen.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    ref_model = modeling_gpt_2jigen.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    params = load_file('./model/model.safetensors', device="cpu")
    model.load_state_dict(params, strict=False)
    model.init_qk_lora()
    ref_model.load_state_dict(params, strict=False)
    model.train()
    model.to(device)
    ref_model.to(device)
    
    reward_model = Model('./chatglm', pre_seq_len=64, device=device).to(device=device) #, prefix_path=args.prefix_path
    reward_optim = optim.AdamW(reward_model.parameters(), lr=1e-2) #None #

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    with open('./data.txt', 'r') as f:
        lines = f.readlines()
        samples = [[int(token) for token in line.strip().split()] for line in lines]
    total_steps = int((len(samples) * epochs) / (batch_size * gradient_accumulation))
    print('total steps = {}'.format(total_steps))
    
    #for name, params in model.named_parameters():
    #    if params.requires_grad:
    #        print(name)
    #optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                                      num_training_steps=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    
    print('starting training')
    overall_step = 0
    running_loss = 0
    running_ml_loss = 0
    n_epoch = 1
    #ref_model = model.clone()
    ref_model.eval()
    reward_model.eval()
    n_samples = args.n_samples
    n_iteration = 1
    rm_train_time = 0
    miscls_neg, miscls_pos = train_reward_model(samples, batch_size, reward_optim, reward_model, model, full_tokenizer, device)
    for iteration in range(n_iteration):
        now = datetime.now()
        training_samples = samples.copy()
        random.shuffle(training_samples)
        model.train()
        for step, batches in generate_batch(training_samples, batch_size, n_samples):
            rl_data = []
            for batch in batches:
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)
                random_n = random.randint(0, 64)
                random_m = random.randint(0, 64)
                random_k = random.randint(5, 12)
                prompt = batch_inputs[:, random_n:random_n + random_k]
                with torch.no_grad():
                    _, sample_sentences = generate(model, full_tokenizer, prompt, 86 - random_k, do_sample=True) #
                    #print(sample_sentences)
                    #print(sample_sentences)
                    inputs = full_tokenizer(sample_sentences, padding=True, return_tensors='pt')
                    reward_prompt = [reward_model.tokenizer.build_prompt(glm_prompt.format(sample_sentence)) + '以上文字' for sample_sentence in sample_sentences]
                    reward_inputs = reward_model.tokenizer(reward_prompt, padding=True, return_tensors='pt')
                    reward = reward_model.get_reward(reward_inputs)
                    logits = model(inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device)).logits
                    ref_logits = ref_model(inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device)).logits
                    reward_values = model.critic(inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device))
                    reward = reward.squeeze(dim=-1)
                    for i, r in enumerate(reward.tolist()):
                        if r > 0:
                            miscls_neg.append(sample_sentences[i])
                    #miscls_neg.append(sample_sentences[reward.argmax().item()])
                    '''
                    for sent_reward, sent in zip(reward, sample_sentences):
                        if sent_reward.item() > 0:
                            miscls_neg.append(sent)
                    '''
                    reward_values = reward_values.squeeze(dim=-1)[:, :-1]
                    log_probs = gather_log_probabilities(logits[:, :-1], inputs.input_ids.to(device)[:, 1:])
                    ref_log_probs = gather_log_probabilities(logits[:, :-1], inputs.input_ids.to(device)[:, 1:])
                    batch = {
                        'prompt': prompt,
                        'log_probs': log_probs,
                        'ref_log_probs': ref_log_probs,
                        'reward': reward,
                        'reward_values': reward_values,
                        'inputs': inputs,
                        'ptx_inputs': batch_inputs[:, random_m:random_m + 86],
                        'sentences': sample_sentences,
                    }
                rl_data.append(batch)
            torch.cuda.empty_cache()
            if step % 2 == 0:
                print('iter: {}, step: {}, sample: {}'.format(iteration, step, rl_data[0]['sentences'][0]))
                print('reward', sum([batch['reward'].mean().item() for batch in rl_data]) / n_samples)
            for i in range(n_epoch):
                #random.shuffle(rl_data)
                for batch in rl_data:
                    input_ids = batch['inputs'].input_ids.to(device)
                    attention_mask = batch['inputs'].attention_mask.to(device)
                    #position_ids = batch['inputs'].position_ids.to(device)
                    prompt = batch['prompt']
                    reward = batch['reward']
                    old_reward_values = batch['reward_values']
                    old_log_probs = batch['log_probs']
                    ref_log_probs = batch['ref_log_probs']
                    start = prompt.size(-1) - 1
                    ptx_inputs = batch['ptx_inputs']
                    #sequence_mask = attention_mask[:, 1:]
                    #print(reward)
                    with torch.no_grad():
                        old_rewards = add_kl_divergence_regularization(
                            reward,
                            prompt,
                            old_log_probs,
                            ref_log_probs,
                            attention_mask[:, 1:],
                        )
                        reward_advantages, reward_returns = get_advantages_and_returns(
                            old_reward_values,
                            old_rewards,
                            attention_mask[:, 1:],
                            start,
                        )
                        #print(reward_advantages.mean(dim=-1))
                    outputs = model.forward(input_ids=ptx_inputs[:, :86], labels=ptx_inputs[:, :86])
                    ptx_loss = outputs[0].mean()
                    logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
                    log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
                    actor_loss = actor_loss_fn(
                        log_probs[:, start:],
                        old_log_probs[:, start:],
                        reward_advantages,
                        attention_mask[:, 1:][:, start:],
                    )
                    reward_values = model.critic(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    reward_values = reward_values.squeeze(dim=-1)[:, :-1]
                    reward_critic_loss = critic_loss_fn(
                        reward_values[:, start:],
                        old_reward_values[:, start:],
                        reward_returns,
                        attention_mask[:, 1:][:, start:],
                    )
                    # lr 1e-5
                    loss = (0.6 * ptx_loss + actor_loss + 0.1 * reward_critic_loss) / n_samples # 0.8 best?
                    loss.backward()
                    if step % 2 == 0:
                        print('ptx_loss: {:.5f}, actor_loss: {:.5f}, reward_critic_loss: {:.5f}, total_loss: {:.5f}.'.format(ptx_loss.item(), 
                            actor_loss.item(), 
                            reward_critic_loss.item(),
                            loss.item()))
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % 20 == 0: # 16
                print('saving model for step {}'.format(step + 1))
                if not os.path.exists(output_dir + 'model_step_{}'.format(step + 1)):
                    os.mkdir(output_dir + 'model_step_{}'.format(step + 1))
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir + 'model_step_{}'.format(step + 1))
                rm_train_time += 1.
                # retrain reward model to avoid reward hacking
                miscls_neg, miscls_pos = train_reward_model(samples, batch_size, reward_optim, reward_model, model, full_tokenizer, device, miscls_neg, miscls_pos, rm_train_time)
        print('iteration {} finished'.format(iteration + 1))
        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one iteration: {}'.format(then - now))

def generate_batch(training_samples, batch_size, n_samples):
    for step in range(len(training_samples) // (batch_size * n_samples)):
        batches = [training_samples[(step + i) * batch_size: (step + i + 1) * batch_size] for i in range(n_samples)]
        yield step, batches

def train_reward_model(training_samples, batch_size, reward_optim, reward_model, model, full_tokenizer, device, miscls_neg=[], miscls_pos=[], n_time=0):
    reward_model.unfreeze()
    #reward_model.reinit()
    reward_model.train()
    model.eval()
    grad_accum = 4
    # ??? should create?
    #reward_optim = optim.AdamW(reward_model.parameters(), lr=1e-2)
    corpus = training_samples.copy()
    random.shuffle(corpus)
    loss_fn = nn.BCELoss()
    n_correct = 0
    n_total = 0
    if n_time == 0:
        n_train_data = 256 #+ int(n_expand) * batch_size * grad_accum
    else:
        n_train_data = 64
    n_test_data = 56
    n_data = n_train_data + n_test_data
    n_train_step = n_train_data // batch_size
    n_step = n_data // batch_size
    max_n_old_data = n_train_data // 4
    neg_data, pos_data = [] + miscls_neg[-max_n_old_data:], [] + miscls_pos[-max_n_old_data:]
    generating_batch_size = batch_size * 4
    n_reuired_data = max(n_data - len(neg_data), n_data - len(pos_data))
    n_generating_step = math.ceil(n_reuired_data / generating_batch_size)
    batches = []
    miscls_neg, miscls_pos = [], []
    with torch.no_grad():
        for step, batch_ in generate_batch(corpus, generating_batch_size, 1):
            batch_inputs = []
            for ids in batch_[0]:
                int_ids = [int(x) for x in ids]
                batch_inputs.append(int_ids)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)
            n_added_neg = min(n_data - len(neg_data), generating_batch_size)
            if n_added_neg > 0:
                random_n = random.randint(0, 64)
                random_k = random.randint(5, 12)
                prompt = batch_inputs[:, random_n:random_n + random_k]
                _, sample_sentences = generate(model, full_tokenizer, prompt, 86 - random_k, do_sample=True)
                neg_data += sample_sentences[:n_added_neg]
            n_added_pos = min(n_data - len(pos_data), generating_batch_size)
            if n_added_pos > 0:
                sentences = ids_to_text(full_tokenizer, batch_inputs[:, random_n:random_n + 86])
                pos_data += sentences[:n_added_pos]
            if step > n_generating_step:
                break
    batches = [(l, r) for l, r in zip(neg_data, pos_data)]
    new_batches = []
    for step in range(n_step):
        new_batches.append(batches[step * batch_size:(step + 1) * batch_size])
    #print(len(new_batches), new_batches[0])
    for step, batch in enumerate(new_batches):
        glm_prompt = "“{}”\n以上文字难以理解吗？"
        inputs1 = [glm_prompt.format(l) for l, _ in batch]
        inputs1 = [reward_model.tokenizer.build_prompt(input_) + '以上文字' for input_ in inputs1]
        inputs1 = reward_model.tokenizer(inputs1, padding=True, return_tensors='pt')
        inputs2 = [glm_prompt.format(r) for _, r in batch]
        inputs2 = [reward_model.tokenizer.build_prompt(input_) + '以上文字' for input_ in inputs2]
        inputs2 = reward_model.tokenizer(inputs2, padding=True, return_tensors='pt')
        if step < n_train_step:
            probs1, probs2 = reward_model(inputs1, inputs2)
            labels_neg = torch.zeros(batch_size).float().to(device=device)
            labels_pos = torch.ones(batch_size).float().to(device=device)
            loss1 = loss_fn(probs1.view(-1), labels_neg)
            loss2 = loss_fn(probs2.view(-1), labels_pos)
            loss = (loss1 + loss2) / 2
            loss = loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                print(step, loss.item())
                reward_optim.step()
                reward_optim.zero_grad()
                torch.cuda.empty_cache()
        elif step >= n_train_step and step <= n_step:
            reward_model.eval()
            n_total += batch_size * 2
            with torch.no_grad():
                probs1, probs2 = reward_model(inputs1, inputs2)
            n_correct += (probs1 < 0.5).sum().item()
            n_correct += (probs2 > 0.5).sum().item()
            for i, (prob1, prob2) in enumerate(torch.cat((probs1, probs2), dim=-1).tolist()):
                if prob1 > 0.5:
                    miscls_neg.append(batch[i][0])
                if prob2 < 0.5:
                    miscls_pos.append(batch[i][1])
    #del reward_optim
    torch.cuda.empty_cache()
    acc = n_correct / n_total
    print('acc', acc)
    reward_model.freeze()
    print('mis neg', len(miscls_neg), 'mis pos', len(miscls_pos))
    return miscls_neg, miscls_pos

def masked_mean(
    x: torch.Tensor,  # size = (B, L)
    mask: torch.BoolTensor = None,  # size = (B, L)
) -> torch.Tensor:  # size = ()
    """Compute the mean of a tensor with a mask."""
    if mask is None:
        return x.mean()
    return ((x * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()

def actor_loss_fn(
        log_probs: torch.Tensor,  # size = (B, L - S)
        old_log_probs: torch.Tensor,  # size = (B, L - S)
        advantages: torch.Tensor,  # size = (B, L - S)
        mask: torch.BoolTensor,  # size = (B, L - S)
        clip_range_ratio: int = 0.2
    ) -> torch.Tensor:  # size = ()
        # size = (B, L - S)
        ratios = torch.exp(log_probs - old_log_probs)
        surrogate1 = advantages * ratios
        surrogate2 = advantages * torch.clamp(
            ratios,
            1.0 - clip_range_ratio,
            1.0 + clip_range_ratio,
        )
        surrogate = torch.minimum(surrogate1, surrogate2)
        return -masked_mean(surrogate, mask)

def critic_loss_fn(
        values: torch.Tensor,  # size = (B, L - S)
        old_values: torch.Tensor,  # size = (B, L - S)
        returns: torch.Tensor,  # size = (B, L - S)
        mask: torch.BoolTensor,  # size = (B, L - S)
        clip_range_value = 5.0
    ) -> torch.Tensor:  # size = ()
        """Compute critic loss."""
        # size = (B, L - S)
        values_clipped = torch.clamp(
            values,
            old_values - clip_range_value,
            old_values + clip_range_value,
        )
        vf_loss1 = torch.square(values - returns)
        vf_loss2 = torch.square(values_clipped - returns)
        return 0.5 * masked_mean(torch.maximum(vf_loss1, vf_loss2), mask)

def get_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        sequence_mask: torch.BoolTensor,
        start: int,
        gamma: int = 1.0,
        gae_lambda: int = 0.99, #0.95
        
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
        # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        last_gae_lambda = 0.0
        advantages_reversed = []
        values = values * sequence_mask
        rewards = rewards * sequence_mask
        length = rewards.size(-1)
        for t in reversed(range(start, length)):  # pylint: disable=invalid-name
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + gamma * next_values - values[:, t]
            last_gae_lambda = delta + gamma * gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        #print(returns.sum)
        return advantages.detach(), returns

def add_kl_divergence_regularization(
        reward: torch.Tensor,  # size = (B,)
        prompt: torch.LongTensor,  # size = (B, S) # pylint: disable=unused-argument
        log_probs: torch.Tensor,  # size = (B, L)
        ref_log_probs: torch.Tensor,  # size = (B, L)
        sequence_mask: torch.BoolTensor,  # size = (B, L)
        kl_coeff: int = 0.02,
        clip_range_score: int = 50
    ) -> torch.Tensor:  # size = (B, L)
    end_index = torch.cat([m.nonzero()[-1] for m in sequence_mask])  # size = (B,)

    # size = (B, L)
    kl_divergence_estimate = log_probs - ref_log_probs
    #print(log_probs, ref_log_probs)
    kl_penalty_rewards = -kl_coeff * kl_divergence_estimate
    rewards = torch.scatter_add(
        kl_penalty_rewards,
        dim=-1,
        index=end_index.unsqueeze(dim=-1),
        src=reward.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
    )
    return torch.clamp(rewards, min=-clip_range_score, max=clip_range_score)

def gather_log_probabilities(
    logits: torch.Tensor,  # size = (B, L, V)
    labels: torch.LongTensor,  # size = (B, L)
) -> torch.Tensor:  # size = (B, L)
    """Gather log probabilities of the given labels from the logits."""
    log_probs = F.log_softmax(logits, dim=-1)  # size = (B, L, V)
    gathered_log_probs = torch.gather(  # size = (B, L, 1)
        log_probs,
        dim=-1,
        index=labels.unsqueeze(dim=-1),
    )
    return gathered_log_probs.squeeze(dim=-1)

def top_k_filtering(logits, top_k=0, filter_value=-float('Inf')):
    top_values = logits.topk(k=top_k)[0][..., -1, None]
    indices_to_remove = logits < top_values
    logits[indices_to_remove] = filter_value
    return logits

def generate(model, tokenizer, inputs, length, do_sample=False):
    batch_size = inputs.size(0)
    sentences = []
    prev, past = inputs, None
    for i in range(length):
        output = model(prev, past_key_values=past)
        output, past = output[:2]
        output = output[:, -1, :]
        output[:, tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        if do_sample:
            filtered_logits = top_k_filtering(output, top_k=8, filter_value=-1e10) #8
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1).reshape(-1)
        else:
            next_token = output.argmax(dim=-1)
        sentences.append(next_token.detach())
        prev = next_token.view(-1, 1)
    # [SL, B]
    ids = torch.stack(sentences)
    ids = torch.cat([inputs, ids.transpose(1, 0)], dim=-1)
    sentences = [tokenizer.convert_ids_to_tokens(sentence) for sentence in ids]
    sentences = [''.join(sentence).replace('[MASK]', '').replace('[CLS]', '').replace('[SEP]', '\n').replace('##', '') for sentence in sentences]
    return ids, sentences

def ids_to_text(tokenizer, ids):
    sentences = [tokenizer.convert_ids_to_tokens(sentence) for sentence in ids]
    sentences = [''.join(sentence).replace('[MASK]', '').replace('[CLS]', '').replace('[SEP]', '\n').replace('##', '') for sentence in sentences]
    return sentences

if __name__ == '__main__':
    main()
