import os
from gemma.config import get_config_for_7b, get_config_for_2b
from gemma.model import GemmaForCausalLM
import torch
import shutil
import time
# 解压文件
# ckpt_path = 'F:\\GemmaLLM\\ChekPoints\\7b-it'
# shutil.unpack_archive('F:\\GemmaLLM\\7B-it.tar.gz', ckpt_path)

# 设置模型  笔记本别跑7B，会卡死
VARIANT = '2b-it'
# 设置设备信息，如果只有cpu的话，改成下面这个
# MACHINE_TYPE = 'cpu'
MACHINE_TYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置最大生成长度
max_lens = 200
# 设置温度值，过高会胡思乱想（0-1）
temps = 0.77




# Set up model config.
model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
model_config.tokenizer = 'F:\\GemmaLLM\\ChekPoints\\2b-it\\tokenizer.model'
model_config.quant = 'quant' in VARIANT

# Instantiate the model and load the weights.
torch.set_default_dtype(model_config.get_dtype())
device = torch.device(MACHINE_TYPE)
model = GemmaForCausalLM(model_config)
model.load_weights('F:\\GemmaLLM\\ChekPoints\\2b-it\\gemma-2b-it.ckpt')
model = model.to(device).eval()
# Generate with one request in chat mode

# a = ('1'+'2')
# 生成一个持续的问题和回答：

USER_CHAT_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n'
MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n{prompt}<end_of_turn>\n'

user_input_all = ''
# 在控制台等待用户输入命令
while True:
    # newspeak, newout = chats(user_input)
    user_input = input("Enter command: ")
    user_input_f = USER_CHAT_TEMPLATE.format(prompt=user_input)
    # print(1)
    if user_input_all == '':
        # 第一次输入
        user_input_all = user_input_f
        out = model.generate(
            prompts=user_input_all,
            device=device,
            output_len=max_lens,
            temperature=temps
        )
    else:
        # a = (user_input_all)
        user_input_all = user_input_all+user_input_f
        out = model.generate(
            prompts=user_input_all,
            device=device,
            output_len=max_lens,
            temperature=temps
        )
    out_f = MODEL_CHAT_TEMPLATE.format(prompt=out)
    user_input_all = user_input_all + out_f
    # print(user_input_all)   # 查看输出是否正确

    # 输出结果：
    for char in out_f[20:-14]+'\n':
        print(char, end='', flush=True)  # 使用 end='' 避免换行，flush=True 确保立即打印到控制台
        # 添加适当的延迟，以便观察每个字符的打印过程
        time.sleep(0.05)  # 如果需要添加延迟，需要导入 time 模块
# 测试问题：
# 我有一个二维dataframe，我想根据其中的一列查找其中包含'het'字符串的所有行
