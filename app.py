import torch
import chainlit
from utils_for_app import load_fine_tune_model, script_args, generate

# Load model and tokenizer
print(f"Loading finetuned model ...")
base_model_id="meta-llama/Llama-3.2-1B-Instruct"
lora_weights="LLAMA32_fine_tuned.pth"
model,tokenizer=load_fine_tune_model(base_model_id,lora_weights,script_args)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    torch.manual_seed(123)

    prompt = message.content

    response = generate(
        model=model, 
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.8,
        top_k=5
        )

    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface
    ).send()
