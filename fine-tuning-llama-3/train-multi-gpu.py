# Authors: Fluid Numerics
#          Garrett Byrd             (garrett@fluidnumerics.com)
#          Dr. Joseph Schoonover    (joe@fluidnumerics.com)

# Fine-Tuning Llama-3 on AMD Radeon GPU
# with Fully-Sharded Data Parellel for Multi-GPU Training

# https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora

import torch
from numpy import argmax
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    pipeline,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import evaluate
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState
import sys

print(f"Device name: {torch.cuda.get_device_name(0)}")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("No CUDA device found.")
    sys.exit()

print(f"Device: {device}")

path_to_model = ""  # Set path to your local model, or a model from Hugging Face

my_tokenizer = AutoTokenizer.from_pretrained(path_to_model)
my_tokenizer.pad_token = my_tokenizer.eos_token

fp4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.float16,  # Must be the same as compute_dtype, or FSDP will not work!
)

quantized_model = LlamaForCausalLM.from_pretrained(
    path_to_model,
    quantization_config=fp4_config,
    torch_dtype=torch.float16,
    device_map={"": PartialState().process_index},  # Maps to both GPUs
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "up_proj",
        "down_proj",
        "gate_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    ],
)

# Apply the LoRA config
adapted_model = get_peft_model(quantized_model, lora_config)

# Load dataset
MetaMathQA = load_dataset(
    "json", data_files="MetaMathQA/MetaMathQA-395K.json", split="train[:100]"
)

# Split dataset into "test" and "train" columns
MetaMathQA = MetaMathQA.train_test_split(test_size=0.2)


# Format data
def instructify(qr_row):
    qr_json = [
        {
            "role": "user",
            "content": qr_row["query"],
        },
        {
            "role": "assistant",
            "content": qr_row["response"],
        },
    ]

    qr_row["text"] = my_tokenizer.apply_chat_template(qr_json, tokenize=False)
    return qr_row


my_tokenizer.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['content'] | trim + '\n' %}{{ content }}{% endfor %}"""
formatted_dataset = MetaMathQA.map(instructify)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = argmax(logits, axis=-1)
    return evaluate.metric.compute(predictions=predictions, references=labels)


training_arguments = TrainingArguments(
    output_dir="Llama-Math-Multi-GPU",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.25,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=1e-4,
    fp16=True,
    bf16=False,
    group_by_length=True,
    save_strategy="no",  # turn off checkpointing
    fsdp="full_shard auto_wrap",  # https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/trainer#transformers.TrainingArguments.fsdp
)

trainer = SFTTrainer(
    model=adapted_model,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=my_tokenizer,
    args=training_arguments,
    packing=False,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model()
