from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import autopep8
import subprocess
import time
import re
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional

try:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
    _pygments_available = True
except ImportError:
    _pygments_available = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Auto Language Code Evaluation API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
    expose_headers=["*"]
)

CACHE_DIR = Path("./.cache/huggingface")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)

HF_TOKEN = os.getenv("HF_API_TOKEN")

MODEL_NAME = "Salesforce/codet5-small"
tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForSeq2SeqLM] = None

LANG_EXT = {
    'python': 'py',
    'java': 'java',
    'cpp': 'cpp',
    'javascript': 'js'
}

class CodeRequest(BaseModel):
    code: str
    language: Optional[str] = None

def detect_language(code: str) -> str:
    if not _pygments_available:
        logger.warning('Pygments not installed; defaulting to python')
        return 'python'
    try:
        lexer = guess_lexer(code)
        alias = lexer.aliases[0]
        if alias in ('python', 'py'):
            return 'python'
        elif alias == 'java':
            return 'java'
        elif alias in ('cpp', 'c++'):
            return 'cpp'
        elif alias in ('js', 'javascript', 'nodejs'):
            return 'javascript'
    except Exception as ex:
        logger.warning(f'Language detection failed ({ex}); defaulting to python')
    return 'python'

def evaluate_code(user_code: str, lang: str) -> dict:
    file_ext = LANG_EXT.get(lang, 'txt')
    filename = f"temp_script.{file_ext}"
    try:
        with open(filename, "w") as f:
            f.write(user_code)

        commands = {
            'python': ['python3', filename],
            'java': ['javac', filename, '&&', 'java', filename.replace('.java', '')],
            'cpp': ['g++', filename, '-o', 'temp_out', '&&', './temp_out'],
            'javascript': ['node', filename]
        }

        start_time = time.time()
        if lang in commands:
            proc = subprocess.run(' '.join(commands[lang]), capture_output=True,
                                  text=True, timeout=15, shell=True)
            exec_time = time.time() - start_time
            success = proc.returncode == 0
            stderr = proc.stderr.strip() if not success else None
        else:
            return {'status': 'error', 'message': 'Unsupported language', 'score': 0}

        score = 0
        score += 50 if success else 0
        score += 20 if len(user_code) < 200 else 10
        score += 30 if exec_time < 1 else 10
        score += 20 if not re.search(r"\b(eval|exec)\b", user_code) else 0
        total = max(0, min(score, 100))

        feedback = []
        if not success:
            feedback.append(f"Error: {stderr}")
        else:
            feedback.append("Execution successful.")
        if exec_time >= 1:
            feedback.append("Performance: consider optimizing loops.")
        if len(user_code) >= 200:
            feedback.append("Readability: refactor into functions.")
        if re.search(r"\b(eval|exec)\b", user_code):
            feedback.append("Security: avoid eval()/exec().")

        return {
            'status': 'success' if success else 'error',
            'execution_time': round(exec_time, 3) if success else None,
            'score': total,
            'feedback': feedback
        }

    except Exception as ex:
        logger.error(f"Evaluation error: {ex}")
        return {'status': 'error', 'message': str(ex), 'score': 0}

def fallback_optimize_code(code: str, lang: str) -> str:
    if lang == 'python':
        optimized = autopep8.fix_code(code)
        optimized += "\n# TIP: Break down large functions for readability."
        return optimized
    elif lang == 'java':
        optimized = re.sub(r'{', '{\n', code)
        optimized = re.sub(r';', ';\n', optimized)
        optimized += "\n// TIP: Use streams and avoid nested loops when possible."
        return optimized
    elif lang == 'cpp':
        optimized = re.sub(r'{', '{\n', code)
        optimized = re.sub(r';', ';\n', optimized)
        optimized += "\n// TIP: Consider STL algorithms and minimize pointer usage."
        return optimized
    return code + "\n# No optimization available for this language."

def optimize_code_ai(user_code: str, lang: str) -> str:
    global tokenizer, model
    try:
        if tokenizer is None or model is None:
            logger.info('Loading model for optimization...')
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, cache_dir=str(CACHE_DIR))
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME, token=HF_TOKEN, cache_dir=str(CACHE_DIR), torch_dtype=torch.float32
            ).to('cpu')
            logger.info('Model loaded on CPU')

        logger.info(f'Generating optimized code for language: {lang}')
        prompt = f"Improve the following {lang} code for readability, performance, and best practices:\n\n{user_code.strip()}\n"

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to('cpu')
        outputs = model.generate(**inputs, max_length=512, num_beams=3, early_stopping=True)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        logger.info(f"Decoded Output: {decoded}")

        if len(decoded) < 5 or decoded == user_code.strip():
            raise ValueError("Model returned empty or unhelpful response")

        return decoded

    except Exception as ex:
        logger.warning(f"LLM optimization failed: {ex} – Using fallback optimization.")
        return fallback_optimize_code(user_code, lang)

@app.post('/evaluate')
async def evaluate_endpoint(req: CodeRequest):
    lang = req.language or detect_language(req.code)
    logger.info(f'Evaluate request language: {lang}')
    result = evaluate_code(req.code, lang)
    return {'language': lang, 'result': result}

@app.post('/optimize')
async def optimize_endpoint(req: CodeRequest):
    lang = req.language or detect_language(req.code)
    logger.info(f'Optimize endpoint called; language: {lang}')
    optimized_code = optimize_code_ai(req.code, lang)
    return {'language': lang, 'optimized_code': optimized_code}

@app.options('/evaluate')
async def options_eval():
    return Response(status_code=200, headers={'Access-Control-Allow-Origin': '*'})

@app.options('/optimize')
async def options_opt():
    return Response(status_code=200, headers={'Access-Control-Allow-Origin': '*'})

@app.get('/health')
async def health():
    return {'status': 'ok' if model else 'loading'}

@app.get('/')
async def root():
    return {'message': 'Auto Language Code API running'}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run('auto_lang_code_api:app', host='0.0.0.0', port=port, workers=1)
