import torch

from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from argparse import Namespace


script_args = Namespace(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    max_grad_norm=0.3,
    weight_decay=0.01,
    lora_alpha=16,
    lora_dropout=0.0,
    lora_r=8,
    max_seq_length=256,
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    tokenizer_path="tokenizer.model",
    # dataset_name="tatsu-lab/alpaca",
    dataset_name="instruction-data.json",
    device_map="cpu",
    use_4bit=True,
    bnb_4bit_compute_dtype="float16",
    num_train_epochs=10,
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    max_steps=200,
    warmup_steps=50,
    group_by_length=True, # Group sequences into batches with same length
    eval_steps=10,
    save_steps=10,
    logging_steps=10, # Log every X updates steps
    report_to="wandb",
    output_dir="./results_packing",
)

def generate(model, prompt, tokenizer, max_new_tokens, context_size=256, temperature=0.0, top_k=1, eos_id=[128001,128009]):
    """ Generate till reaching max_new_tokens or till eos_id=<|eot_id|> or <|end_of_text|>"""
    formatted_prompt=(
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}"
        f"### Response:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    idx=tokenizer.encode(formatted_prompt)
    idx=torch.tensor(idx).unsqueeze(0).to(script_args.device_map) # add batch dimension
    _,num_tokens=idx.shape

    # Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            outputs = model.forward(idx_cond)
            logits=outputs.logits
        # last time step
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next in eos_id:  # Stop generating early if <|eot_id|> or<|end_of_text|> token is encountered 
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
        
    #print(f"Output tensor: {idx.shape}")
    # remove batch dimension
    idx_flat=idx.squeeze(0)
    generated_ids=idx_flat[num_tokens:] # take out the input prompt
    generated_text=tokenizer.decode(generated_ids)
    
    return generated_text


def load_fine_tune_model(base_model_id, saved_weights,args):
    """ Load the fine tuned model and tokenizer
        The fined model is LLAMA3.2 + LoRA layers. We use PeftModel to load it
    """
    # load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.to(args.device_map)


    # Load LoRA model
    peft_config=LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj'],)
    lora_model = PeftModel(base_model, peft_config)

    # Load and adapt state dict
    state_dict = torch.load(saved_weights,map_location=args.device_map)
    
    # Adapt keys to match expected format
    #   Expected keys have prefix: base_model.model.model.
    #   Saved weights have prefix: model.
    new_state_dict = {}
    for key, value in state_dict.items():
        # Add the expected prefix
        new_key = f"base_model.model.{key}"
        new_state_dict[new_key] = value

    # load finetuned lora weights
    lora_model.load_state_dict(new_state_dict, strict=False)
    
    # Ensure model is in eval mode for inference
    lora_model = lora_model.eval()
    
    # Clear cache to free up memory (only for cuda)
    #torch.cuda.empty_cache()

    return lora_model, tokenizer

# base_model_id="meta-llama/Llama-3.2-1B-Instruct"
# lora_weights="LLAMA32_fine_tuned.pth"
# model_ft,tokenizer=load_fine_tune_model(base_model_id,lora_weights,script_args)

# # Generate with ft model: adjust temperature and top_k for different outputs
# prompt="Explain the function of human heart"
# print(generate(model_ft, prompt, tokenizer, max_new_tokens=100,temperature=0.8,top_k=10))