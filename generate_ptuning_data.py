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
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    #parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
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
    device = 'cuda:1' #if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    #raw_data_path = args.raw_data_path
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

    import modeling_gpt_2jigen
    from safetensors.torch import load_file
    model = modeling_gpt_2jigen.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    params = load_file('./model/model.safetensors', device="cpu")
    model.load_state_dict(params, strict=False)
    model.to(device)

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

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    print('starting generating')
    output_training_data = 350
    output_n_data = 400
    n_output = 0
    test_data_file = open('./ptuning-raw-test-data.txt', 'w', encoding='utf-8')
    with open('./ptuning-raw-train-data.txt', 'w', encoding='utf-8') as f:
        now = datetime.now()
        training_samples = samples.copy()
        random.shuffle(training_samples)
        model.eval()
        with torch.no_grad():
            for step in range(len(training_samples) // batch_size):
                batch = training_samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)
                
                random_k = random.randint(5, 10)
                random_n = random.randint(0, 32)
                _, sample_sentences1 = generate(model, full_tokenizer, batch_inputs[:, :random_k], 86 - random_k, do_sample=True) # 64
                sentences = ids_to_text(full_tokenizer, batch_inputs[:, random_n:random_n + 86])
                
                for s1, s2 in zip(sample_sentences1, sentences):
                    n_output += 1
                    if n_output <= output_training_data:
                        f.write('[{},\n'.format(json.dumps(s1, ensure_ascii=False)))
                        f.write('    {},]\n'.format(json.dumps(s2, ensure_ascii=False)))
                    else:
                        test_data_file.write('[{},\n'.format(json.dumps(s1, ensure_ascii=False)))
                        test_data_file.write('    {},]\n'.format(json.dumps(s2, ensure_ascii=False)))
                
                print(n_output, '/', output_n_data)
                if n_output > output_n_data:
                    break
        test_data_file.close()
        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('generating finished')

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

def ids_to_text(tokenizer, ids):
    sentences = [tokenizer.convert_ids_to_tokens(sentence) for sentence in ids]
    sentences = [''.join(sentence).replace('[MASK]', '').replace('[CLS]', '').replace('[SEP]', '\n').replace('##', '') for sentence in sentences]
    return sentences

if __name__ == '__main__':
    main()
