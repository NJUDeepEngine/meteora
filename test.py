from configuration_llama_meteor import LlamaMeteorConfig
from modeling_llama_meteor import LlamaMeteorForCausalLM, LlamaMeteorModel
import torch
from MoELoRA.peft_model import PeftModel
from transformers import AutoTokenizer

# from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
import os
os.environ['CURL_CA_BUNDLE'] = ''

hf_auth = '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'

device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name()
print(f"Using device: {device} ({device_name})")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_auth)
model_config = LlamaMeteorConfig(device_map="cuda")
model_config.save_pretrained("llama-meteor")

model_config = LlamaMeteorConfig.from_pretrained("llama-meteor")

print("model config loaded")
llama_meteor = LlamaMeteorForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_auth, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
print("model loaded", llama_meteor)


# ADAPTERS = {'lora1':'/data1/model/lora_adapters/llama2-7b/multi-languages/RuterNorway/Llama-2-7b-chat-norwegian-LoRa', 'lora3':'/data1/model/lora_adapters/llama2-7b/multi-languages/Tarklanse/Llama2-7B_Traditional_Chinese_roleplay_chat_lora', "lora4":'/data1/model/lora_adapters/llama2-7b/multi-languages/FlagAlpha/Llama2-Chinese-7b-Chat-LoRA'}

ADAPTERS = {}

lora1 = "/data1/model/lora_adapters/llama2-7b/multi-tasks/abcdabcd987/gsm8k-llama2-7b-lora-16"
lora2 = "/data1/model/lora_adapters/llama2-7b/multi-tasks/abcdabcd987/sqlctx-llama2-7b-lora-16"
lora3 = "/data1/model/lora_adapters/llama2-7b/multi-tasks/abcdabcd987/viggo-llama2-7b-lora-16"
ADAPTERS["lora1"] = lora1
ADAPTERS["lora2"] = lora2
ADAPTERS["lora3"] = lora3

#, "lora2":'/data1/model/lora_adapters/llama2-7b/multi-languages/Sparticle/llama-2-7b-chat-japanese-lora',

prompt_lora1 = """
### Instruction
Summarize following text.
### Input:
As a Norwegian company, we understand firsthand the pressing need for powerful language models tailored to specific languages. Our primary focus is on the Norwegian linguistic landscape. In the age of digitization, languages that lack robust, open-source models can risk becoming marginalized. This is why we're introducing this open-source Norwegian model. We believe that by making such resources freely accessible, we can democratize information, foster innovation, and create a more inclusive digital ecosystem. Our aspiration is for this model to serve as a foundational resource for future specialized Norwegian models. Ultimately, our goal is to bolster the Norwegian NLP community and facilitate the smoother integration of Norwegian models into diverse projects.
### Answer
"""



prompt_lora2 = """
「主人公が重要なキャリアの決断をしなければならない」という第三者の語り口で短編小説を書きなさい:
"""

lora2_response = "ジョンは人生の岐路に立っていました。彼は大学を卒業し、今後どのようなキャリアを追求するかという大きな決断を迫られていました。長い間悩んだ末、彼は貧困層を助ける会計士になることを決めました。彼は数字に長けており、自分の仕事の具体的な成果を見ることが好きでした。\n\nジョンは会計のコースに入学し、最初はかなり難しいと感じました。彼は多くのシステムや規制を素早く学ばなければなりませんでしたが、彼は努力して勉強を極めました。数年後、ジョンは自分の街の会計事務所で働き始めました。彼は税金や会計に関する知識を実際の現場で活かすことを熱望していました。\n\nジョンは自分の仕事が大好きで、クライアントのお金を節約する戦略を見つけることで自分の創造性を発揮することができました。数年後、彼はシニア会計士になり、より大きく、より難しい案件を担当するようになりました。彼は今や金融業界で尊敬される存在ですが、彼はまだ自分が大学を卒業したばかりで、自分の人生がどの方向に向かうのかわからなかった時を覚えています。"


prompt_lora3 = """
小明是一個朝九晚五的上班族，個性樂觀開朗，雖然領的薪水穿不暖又餓不死，但他仍然努力地活在當下
以下是小明與使用者的對話
小明:*今天是你上班的第一天，小明被指派為你的指導員，協助你快速理解公司文化，並讓你能快速上手你的工作，在會議室和你講了數十分鐘的簡報後終於初步介紹完畢*...以上是我們公司大概在做甚麼，現在，你有甚麼問題想要提問的嗎?
使用者:你聽起來有那麼一點倦怠，你有離職的打算嗎?
小明:
"""


prompt_lora4 = """
基于以下内容续写故事:
### Input:
自从Meta公司发布第一代LLaMA模型以来，羊驼模型家族繁荣发展。近期Meta发布了Llama2版本，开源可商用，在模型和效果上有了重大更新。Llama2总共公布了7B、13B和70B三种参数大小的模型。相比于LLaMA，Llama2的训练数据达到了2万亿token，上下文长度也由之前的2048升级到4096，可以理解和生成更长的文本。Llama2 Chat模型基于100万人类标记数据微调得到，在英文对话上达到了接近ChatGPT的效果。
### Response:
"""

prompt_gsm = """
### question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?,
### Answer:
"""

adapter_model = PeftModel.from_pretrained_multi(
    llama_meteor, ADAPTERS, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", is_trainable=False
)
print("adapter model loaded", adapter_model)
print("prompt:", prompt_gsm)
with torch.no_grad():
    output_tensors = adapter_model.generate(
        input_ids=tokenizer(prompt_gsm, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=2048,
    )[0]

response = tokenizer.decode(output_tensors, skip_special_tokens=True).split('### Answer')[-1]
print(response)



