import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
# 第一个参数表示下载模型的型号，第二个参数是下载后存放的缓存地址，第三个表示版本号，默认 master
model_dir = snapshot_download('Qwen/Qwen2-1.5B-Instruct', cache_dir='qwen2chat_src', revision='master')

#推理模型
#对 qwen2 模型进行低精度量化至 int4 ，以减少计算资源的需求和提高系统的效率,它可以在硬件上实现快速、低功耗的推理，也可以加快模型加载的速度。

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import os
if __name__ == '__main__':
    model_path = os.path.join("qwen2chat_src/Qwen/Qwen2-1___5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.save_low_bit('qwen2chat_int4')
    tokenizer.save_pretrained('qwen2chat_int4')

#嵌入式模型bge-base下载，用于将文本转换为向量表示
# 第一个参数表示下载模型的型号，第二个参数是下载后存放的缓存地址，第三个表示版本号，默认 master
model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir='qwen2chat_src', revision='master')