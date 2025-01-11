## Fine-tuning LLAMA 3.2-1B on Instruction Dataset

This repository demonstrates the process of fine-tuning the **LLAMA 3.2 1B** on an instruction dataset created by Sebastian Raschka. The dataset can be downloaded from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json.

### Training Details
- Training Framework: The training uses the SFTTrainer from the trl (Transformer Reinforcement Learning) library.
- Parameter Optimization: LoRA (Low-Rank Adaptation) is applied to reduce the number of parameters and improve efficiency during the fine-tuning process.

### Evaluation 
The fine-tuned model is evaluated using the **LLAMA 3 (8 billion)** model from **OLLAMA**. The evaluation yields an average score of 76.43.

### Interactive API with chainlit
Interact with the fine-tuned model through a web API by running the command **chainlit run app.py**. This will launch an interactive interface for the model.

