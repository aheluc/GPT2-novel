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
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
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
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    #parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    #parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    #if args.segment:
    #    from tokenizations import tokenization_bert_word_level as tokenization_bert
    #else:
    #    from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    model_config = transformers.models.gpt2.configuration_gpt2.GPT2Config.from_json_file(args.model_config)
    #print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    #if args.bpe_token:
    #    full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    #else:
    #    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    from transformers import BertTokenizer
    full_tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    full_tokenizer.max_len = 999999
    device = 'cuda:3' #if torch.cuda.is_available() else 'cpu'
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

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    import modeling_gpt_2jigen
    model = modeling_gpt_2jigen.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)
    
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

    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
    #    multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0
    running_ml_loss = 0
    for epoch in range(epochs):
        now = datetime.now()
        #random.shuffle(samples)
        '''
        if epoch > 0:
            losses = []
            model.eval()
            with torch.no_grad():
                for step in range(len(samples) // batch_size):
                    batch = samples[step * batch_size:(step + 1) * batch_size]
                    batch_inputs = []
                    for ids in batch:
                        int_ids = [int(x) for x in ids]
                        batch_inputs.append(int_ids)
                    batch_inputs = torch.tensor(batch_inputs).long().to(device)
                    # forward pass
                    outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                    loss, _ = outputs[:2]
                    losses += loss.mean(dim=-1).cpu().tolist()
            threshold = sorted(losses)[-int(len(samples) * 0.5)]
            training_samples = []
            for loss, sample in zip(losses, samples):
                if loss >= threshold:
                    training_samples.append(sample)
        else:
            training_samples = samples.copy()
        '''
        training_samples = samples.copy()
        random.shuffle(training_samples)
        model.train()
        for step in range(len(training_samples) // batch_size):
            batch = training_samples[step * batch_size: (step + 1) * batch_size]
            batch_inputs = []
            for ids in batch:
                int_ids = [int(x) for x in ids]
                batch_inputs.append(int_ids)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)

            #  forward pass
            outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
            # MLE loss
            loss, logits = outputs[:2]
            MLE_loss = loss.mean()
            
            # RL
            random_k = random.randint(4, 7)
            _, greedy_sentences = generate(model, full_tokenizer, batch_inputs[:, :random_k])
            ids, sample_sentences = generate(model, full_tokenizer, batch_inputs[:, :random_k], do_sample=True)
            prompt = "句1：“{}”\n句2：“{}”\n句1句2哪个更好？"
            inputs = [prompt.format(g, s) for g, s in zip(greedy_sentences, sample_sentences)]
            inputs = [glm_tokenizer.build_prompt(input_) + '更好的是句' for input_ in inputs]
            inputs = glm_tokenizer(inputs, padding=True, return_tensors='pt')
            with torch.no_grad():
                return_obj = glm_model(input_ids=inputs.input_ids.to(device=device), position_ids=inputs.position_ids.to(device=device), attention_mask=inputs.attention_mask.to(device=device)
                    , labels=inputs.input_ids.to(device=device))
            probs = torch.softmax(return_obj.logits[:, -1, [30939, 30943]], dim=-1)
            reward = probs[:, 1] - probs[:, 0]
            outputs = model.forward(input_ids=ids, labels=ids)
            # RL loss
            loss, logits = outputs[:2]
            RL_loss = loss[:, random_k:].mean(dim=-1) * reward.detach()
            RL_loss = RL_loss.mean()
            
            alpha = 0.8
            loss = alpha * MLE_loss + (1 - alpha) * RL_loss
            
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            #optimizer.step()
            #optimizer.zero_grad()
            #scheduler.step()
            if (overall_step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                running_ml_loss += MLE_loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                #print(sample_sentences[0])

            if (overall_step + 1) % log_step == 0:
                print('reward {}, greedy_sentence {}, sample_sentence {}'.format(reward[0].item(), greedy_sentences[0], sample_sentences[0]))
                #tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                print('now time: {}:{}. Step {} of epoch {}, loss {}, ML Loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        epoch + 1,
                    running_loss * gradient_accumulation / (log_step / gradient_accumulation),
                    running_ml_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                running_loss = 0
                running_ml_loss = 0
            overall_step += 1

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')

def top_k_filtering(logits, top_k=0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_values = logits.topk(k=top_k)[0][..., -1, None]
    indices_to_remove = logits < top_values
    logits[indices_to_remove] = filter_value
    return logits

def generate(model, tokenizer, inputs, do_sample=False):
    batch_size = inputs.size(0)
    sentences = []
    prev, past = inputs, None
    for i in range(80):
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
