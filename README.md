It aims to train a GPT2-novel, capable of generating sentences in the style of light novels and sci-fin novels.

Place the GPT-2 weights and code in the 'model' folder.

Place the ChatGLM2 weights and code in the 'chatglm' folder.

The corpus of novels are tokenized in 'data.txt'.

# 1. Fine-tune GPT2 with LoRA on the corpus of novels

```
python lora_tuning.py --pretrained_model ./model --lr 5e-3 --model_config ./model/config.json --warmup_steps 10 --batch_size 16 --output_dir checkpoints_lora/ --log_step 16 --gradient_accumulation 4 --epochs 3
```

# 2. Generate data to train the reward model

```
python generate_ptuning_data.py
```

# 3. Train the ChatGLM-based reward model with P-Tuning v2.

```
python ptuning2.py
```

Let ChatGLM discriminate true sentences and generated sentences. And the logit will be used as the reward. The reward model rates sentences by adversarial training rather than pairwise ranking.

# 4. Perform PPO-PTX reinforcement learning training on fine-tuned GPT2

```
python train_RLLF.py --pretrained_model ./model --lr 5e-3 --model_config ./model/config.json --batch_size 8 --output_dir checkpoints_lora/ 
```

RLLF means Reinforcement Learning from LLM feedback.
