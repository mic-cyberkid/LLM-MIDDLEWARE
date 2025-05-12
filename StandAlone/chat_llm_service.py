import os
import logging
from llama_cpp import Llama

MODEL_DIR = "models"
MODEL_FILENAME = "qwen1_5-0_5b-chat-q8_0.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

GENERATION_PARAMS = {
    "stop": ["<|im_end|>"],
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

class ChatLLMService:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = Llama(model_path=model_path, n_ctx=4096)
        logging.info("LLM model loaded from: %s", model_path)

    def format_prompt(self, message: str, history: list, curriculum_chunks: list) -> str:
        prompt = "<|im_start|>system\nYou are a helpful assistant with access to curriculum knowledge.<|im_end|>\n"

        if curriculum_chunks:
            prompt += f"<|im_start|>user\nCurriculum Context:\n" + "\n---\n".join(curriculum_chunks) + "<|im_end|>\n"

        for msg in history:
            role = msg['role']
            content = msg['content']
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def generate_response(self, prompt: str) -> str:
        try:
            output = self.model(prompt, **GENERATION_PARAMS)
            return output['choices'][0]['text'].strip()
        except Exception as e:
            logging.error("LLM generation error: %s", e)
            raise RuntimeError("Failed to generate LLM response.") from e
