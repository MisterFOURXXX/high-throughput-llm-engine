from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import time

app = FastAPI(title="RAG LLM Inference API", version="1.0.0")

class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="User question or prompt")
    max_new_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.95, description="Top-p sampling")
    top_k: int = Field(50, description="Top-k sampling")

class InferenceResponse(BaseModel):
    generated_text: str
    model_name: str
    processing_time_ms: float

class ModelLoader:
    _instance = None
    _model = None
    _tokenizer = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_model(self, model_path: str):
        if self._model is None:
            print(f"Loading model from {model_path}...")
            base_model_name = "Qwen/Qwen3-0.6B"
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            
            self._model = PeftModel.from_pretrained(base_model, model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "left"
            
            self._model.eval()
            print("Model loaded successfully")
        
        return self._model, self._tokenizer

model_loader = ModelLoader()

@app.on_event("startup")
async def startup_event():
    model_path = os.getenv("MODEL_PATH", "./models/qwen-dpo-final")
    model_loader.load_model(model_path)

@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    model, tokenizer = model_loader.load_model(os.getenv("MODEL_PATH", "./models/qwen-dpo-final"))
    
    enhanced_prompt = f"""<thinking>
Let me solve this coding problem step by step:
1. Problem Analysis:
   - Understand the requirement: {request.prompt}
   - Identify input/output specifications
2. Solution Approach:
   - Choose optimal solution based on constraints
</thinking>

<solution>
```python
# Implementation
def solution():
    pass
</solution>

<explanation>
Complexity Analysis:
- Time Complexity: O(n)
- Space Complexity: O(1)
</explanation>"""
    
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are an expert coding assistant. Always use Chain-of-Thought reasoning."},
         {"role": "user", "content": enhanced_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            top_p=request.top_p,
            top_k=request.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    processing_time = (time.time() - start_time) * 1000
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return InferenceResponse(
        generated_text=generated_text,
        model_name="Qwen3-0.6B-DPO",
        processing_time_ms=processing_time
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_loader._model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)