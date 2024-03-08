import transformers
import torch
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

from transformers import AutoTokenizer, AutoConfig
from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm.tokenization_chatglm import ChatGLMTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='model/config.json', type=str, required=False,
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
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    #parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    #parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    model_config = transformers.models.gpt2.configuration_gpt2.GPT2Config.from_json_file(args.model_config)
    n_ctx = model_config.n_ctx
    from transformers import BertTokenizer
    full_tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    full_tokenizer.max_len = 999999
    device = 'cuda:3' #if torch.cuda.is_available() else 'cpu'
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
    #tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    import modeling_gpt_2jigen
    model = modeling_gpt_2jigen.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)
    
    ref_model = nn.clone_module(model)
    
    glm_config = AutoConfig.from_pretrained('./chatglm', trust_remote_code=True)
    glm_config.pre_seq_len = 1
    glm_config.prefix_projection = False
    glm_model = ChatGLMForConditionalGeneration.from_pretrained('./chatglm', config=glm_config, trust_remote_code=True)
    glm_model = glm_model.half().quantize(4).to(device=device) #
    glm_model.transformer.prefix_encoder.float()
    glm_model.transformer.prefix_encoder.load_state_dict(torch.load('./ptuning.params'))
    glm_tokenizer = ChatGLMTokenizer.from_pretrained('./chatglm', trust_remote_code=True)
    glm_model.eval()
    for param in glm_model.parameters():
        param.requires_grad = False

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

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                          num_training_steps=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    def generate_batch(training_samples, batch_size, n_samples):
        for step in range(len(training_samples) // (batch_size * n_samples)):
            batches = [training_samples[step * batch_size: (step + 1) * batch_size],
                        training_samples[(step + 1) * batch_size: (step + 2) * batch_size],
                        training_samples[(step + 2) * batch_size: (step + 3) * batch_size],
                        training_samples[(step + 3) * batch_size: (step + 4) * batch_size],
                    ]
            yield batches
    
    print('starting training')
    overall_step = 0
    running_loss = 0
    running_ml_loss = 0
    n_epoch = 1
    for iteration in range(n_iteration):
        now = datetime.now()
        training_samples = samples.copy()
        random.shuffle(training_samples)
        n_samples = 4
        for batches in generate_batch(training_samples, batch_size, n_samples):
            rl_data = []
            for batch in batches:
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)
                random_k = random.randint(5, 10)
                prompt = batch_inputs[:, :random_k]
                with torch.no_grad():
                    _, sample_sentences = generate(model, full_tokenizer, prompt, 86 - random_k, do_sample=True)
                    inputs = full_tokenizer(sample_sentences)
                    reward_inputs = reward_model.tokenizer(sample_sentences)
                    
                    logits = model(inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device)).logits
                    ref_logits = ref_model(inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device)).logits
                    
                    reward = reward_model.get_reward(reward_inputs)
                    reward_values = model.critic(inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device))
                    
                    reward = reward.squeeze(dim=-1)
                    reward_values = reward_values.squeeze(dim=-1)[:, :-1]
                    
                    log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
                    ref_log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
                batch = {
                    'prompt': prompt,
                    'log_probs': log_probs,
                    'ref_log_probs': ref_log_probs,
                    'reward': reward,
                    'reward_values': reward_values,
                    'reward': reward,
                    'inputs': inputs,
                    'ptx_inputs': batch_inputs,
                    'sentences': sample_sentences,
                }
                rl_data.append(batch)
            print('sample', rl_data[0]['sentences'][0])
            model.train()
            for i in range(n_epoch):
                #random.shuffle(rl_data)
                for batch in rl_data:
                    input_ids = batch['inputs'].input_ids.to(device)
                    attention_mask = batch['inputs'].attention_mask.to(device)
                    position_ids = batch['inputs'].position_ids.to(device)
                    prompt = batch['prompt']
                    reward_values = batch['reward_values']
                    start = prompt.size(-1) - 1
                    ptx_inputs = batch['ptx_inputs']
                    with torch.no_grad():
                        old_rewards = add_kl_divergence_regularization(
                            reward,
                            prompt,
                            old_log_probs,
                            ref_log_probs,
                            sequence_mask,
                        )
                        reward_advantages, reward_returns = get_advantages_and_returns(
                            old_reward_values,
                            old_rewards,
                            sequence_mask,
                            start,
                        )
                    outputs = model.forward(input_ids=ptx_inputs, labels=ptx_inputs)
                    ptx_loss = outputs[0].mean()
                    logits = model(input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=False).logits
                    log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
                    actor_loss = actor_loss_fn(
                        log_probs[:, start:],
                        old_log_probs[:, start:],
                        reward_advantages,
                        sequence_mask[:, start:],
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
                        sequence_mask[:, start:],
                    )
                    # lr 1e-5
                    loss = (10 * ptx_loss + actor_loss + 0.1 * reward_critic_loss) / n_samples
                    print('ptx_loss: {}, actor_loss: {}, reward_critic_loss: {}, total_loss: {}.'.format())
                optimizer.step()
                optimizer.zero_grad()
        print('saving model for iteration {}'.format(iteration + 1))
        if not os.path.exists(output_dir + 'model_iter_{}'.format(iteration + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(iteration + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(iteration + 1))
        print('iteration {} finished'.format(iteration + 1))
        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one iteration: {}'.format(then - now))
    
def get_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        sequence_mask: torch.BoolTensor,
        start: int,
        gamma: int = 1.0,
        gae_lambda: int = 0.95,
        
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
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae_lambda = delta + gamma * gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

def masked_mean(
    x: torch.Tensor,  # size = (B, L)
    mask: torch.BoolTensor | None = None,  # size = (B, L)
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

def add_kl_divergence_regularization(
        reward: torch.Tensor,  # size = (B,)
        prompt: torch.LongTensor,  # size = (B, S) # pylint: disable=unused-argument
        log_probs: torch.Tensor,  # size = (B, L)
        ref_log_probs: torch.Tensor,  # size = (B, L)
        sequence_mask: torch.BoolTensor,  # size = (B, L)
        kl_coeff: int = 0.02
        clip_range_score: int = 50
    ) -> torch.Tensor:  # size = (B, L)
    end_index = torch.cat([m.nonzero()[-1] for m in sequence_mask])  # size = (B,)

    # size = (B, L)
    kl_divergence_estimate = log_probs - ref_log_probs
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
            filtered_logits = top_k_filtering(output, top_k=8)
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

if __name__ == '__main__':
    main()
