"""运行前复制 .env.example 为 .env，填写权重路径，然后执行 source .env。"""

import os

import pytrio as trio

# 权重路径由本地环境变量提供，避免将个人模型地址提交到仓库。
model_path = os.environ.get("TRIO_MODEL_PATH", "").strip()
if not model_path:
    raise SystemExit("请设置 TRIO_MODEL_PATH：填写 .env 中的权重路径并执行 source .env。")

# 1. 与 TRIO 建立连接
service_client = trio.ServiceClient()

# 2. 创建 1 个推理客户端
sampling_client = service_client.create_sampling_client(
    base_model="Qwen/Qwen3.5-4B",
    model_path=model_path
)

# 3. 获取 Tokenizer 并对输入文本进行预处理
print("Loading tokenizer...")
tokenizer = sampling_client.get_tokenizer()
messages = [
    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": "皇上万福金安，臣妾有事禀告。"},
]
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

input_ids = tokenizer.encode(input_text)
print("tokenizer finish")

# 4. 推理
params = trio.SamplingParams(max_tokens=4096, seed=42, temperature=0.7)
response = sampling_client.sample(
    prompt=trio.ModelInput.from_ints(input_ids),
    num_samples=2,
    sampling_params=params,
)
response = response.result()

for i, seq in enumerate(response.sequences):
    print(f"Sample {i+1}: {repr(seq.text)}")
