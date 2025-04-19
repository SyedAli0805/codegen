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
from typing import Optional, Dict, List
import tempfile
import shutil

# Configuration
try:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
    _pygments_available = True
except ImportError:
    _pygments_available = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

class CodeRequest(BaseModel):
    code: str
    language: Optional[str] = None
    timeout: Optional[int] = 15  # Default timeout in seconds

class EvaluationResult(BaseModel):
    status: str
    execution_time: Optional[float]
    score: int
    feedback: List[str]
    output: Optional[str]
    error: Optional[str]

class OptimizationResult(BaseModel):
    optimized_code: str
    improvements: List[str]
    warnings: List[str]

# Constants
LANG_EXT = {
    'python': 'py',
    'java': 'java',
    'cpp': 'cpp',
    'javascript': 'js',
    'c': 'c'
}

COMMAND_MAP = {
    'python': ['python3', '{filename}'],
    'java': ['javac', '{filename}', '&&', 'java', '{classname}'],
    'cpp': ['g++', '{filename}', '-o', '{output}', '&&', './{output}'],
    'javascript': ['node', '{filename}'],
    'c': ['gcc', '{filename}', '-o', '{output}', '&&', './{output}']
}

SECURITY_RISK_PATTERNS = {
    'python': [r'\b(eval|exec|subprocess|os\.system|_import_)\b'],
    'java': [r'Runtime\.exec', r'ProcessBuilder', r'Unsafe', r'Reflection'],
    'cpp': [r'system\(', r'exec[lv]?\(', r'popen\('],
    'javascript': [r'eval\(', r'Function\(', r'exec\(', r'spawn\('],
    'c': [r'system\(', r'exec[lv]?\(', r'popen\(']
}

app = FastAPI(
    title="Code Evaluation and Optimization API",
    description="API for evaluating and optimizing code in multiple languages",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_environment():
    """Initialize environment and model loading"""
    cache_dir = Path("./.cache/huggingface")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir)

def detect_language(code: str) -> str:
    """Detect programming language from code snippet"""
    if not _pygments_available:
        logger.warning('Pygments not available, defaulting to python')
        return 'python'
    
    try:
        lexer = guess_lexer(code)
        for alias in lexer.aliases:
            if alias in LANG_EXT:
                return alias
        return 'python'
    except Exception as ex:
        logger.warning(f'Language detection failed: {ex}')
        return 'python'

def create_temp_file(code: str, ext: str) -> Path:
    """Create temporary file with the given code"""
    temp_dir = Path(tempfile.mkdtemp())
    file_path = temp_dir / f"temp_code.{ext}"
    file_path.write_text(code)
    return file_path

def cleanup_temp_files(temp_dir: Path):
    """Clean up temporary directory"""
    try:
        shutil.rmtree(temp_dir)
    except Exception as ex:
        logger.warning(f"Failed to clean temp files: {ex}")

def check_security_risks(code: str, language: str) -> List[str]:
    """Check for potentially dangerous patterns in code"""
    warnings = []
    patterns = SECURITY_RISK_PATTERNS.get(language, [])
    
    for pattern in patterns:
        if re.search(pattern, code):
            warnings.append(f"Security warning: Found '{pattern}' in code")
    
    return warnings

def execute_code(file_path: Path, language: str, timeout: int) -> Dict:
    """Execute code and return results"""
    if language not in COMMAND_MAP:
        raise ValueError(f"Unsupported language: {language}")
    
    file_ext = LANG_EXT[language]
    parent_dir = file_path.parent
    filename = file_path.name
    
    # Prepare command based on language
    cmd_template = COMMAND_MAP[language]
    cmd = [part.format(
        filename=filename,
        classname=filename.replace(f'.{file_ext}', ''),
        output='temp_out'
    ) for part in cmd_template]
    
    try:
        start_time = time.time()
        proc = subprocess.run(
            ' '.join(cmd),
            cwd=str(parent_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True
        )
        exec_time = time.time() - start_time
        
        return {
            'success': proc.returncode == 0,
            'output': proc.stdout.strip(),
            'error': proc.stderr.strip(),
            'execution_time': exec_time
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f"Execution timed out after {timeout} seconds",
            'execution_time': timeout
        }
    except Exception as ex:
        return {
            'success': False,
            'error': str(ex),
            'execution_time': 0
        }

def calculate_score(results: Dict, code: str, security_warnings: List[str]) -> int:
    """Calculate evaluation score based on multiple factors"""
    score = 0
    
    # Execution success (50 points)
    score += 50 if results['success'] else 0
    
    # Code length (20 points max)
    score += max(0, 20 - (len(code) // 100))
    
    # Execution time (30 points max)
    if results['success']:
        score += max(10, 30 - int(results['execution_time'] * 5))
    
    # Security (20 points)
    score += 20 if not security_warnings else 0
    
    # Style (10 points for python if passes pep8)
    if detect_language(code) == 'python':
        try:
            fixed = autopep8.fix_code(code)
            if fixed == code:
                score += 10
        except:
            pass
    
    return min(100, max(0, score))

def evaluate_code(code: str, language: str, timeout: int = 15) -> EvaluationResult:
    """Evaluate code for execution, performance, and security"""
    language = language or detect_language(code)
    file_ext = LANG_EXT.get(language, 'txt')
    
    # Check for security risks first
    security_warnings = check_security_risks(code, language)
    
    # Create temp file and execute
    temp_file = create_temp_file(code, file_ext)
    try:
        exec_results = execute_code(temp_file, language, timeout)
        
        # Calculate score
        score = calculate_score(exec_results, code, security_warnings)
        
        # Generate feedback
        feedback = []
        if exec_results['success']:
            feedback.append("Execution successful")
            if exec_results['execution_time'] > 1:
                feedback.append("Performance: Consider optimizing long-running operations")
        else:
            feedback.append(f"Execution failed: {exec_results.get('error', 'Unknown error')}")
        
        if len(code) > 200:
            feedback.append("Readability: Consider breaking down into smaller functions")
        
        feedback.extend(security_warnings)
        
        return EvaluationResult(
            status='success' if exec_results['success'] else 'error',
            execution_time=round(exec_results['execution_time'], 3),
            score=score,
            feedback=feedback,
            output=exec_results.get('output'),
            error=exec_results.get('error')
        )
    finally:
        cleanup_temp_files(temp_file.parent)

def optimize_code_ai(code: str, language: str) -> OptimizationResult:
    """Optimize code using AI model with fallback to basic optimization"""
    global tokenizer, model
    
    # Initialize model if not loaded
    if tokenizer is None or model is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
            logger.info("AI model loaded successfully")
        except Exception as ex:
            logger.error(f"Failed to load AI model: {ex}")
            return optimize_code_fallback(code, language)
    
    try:
        prompt = f"Optimize this {language} code for performance and readability:\n\n{code}\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            num_beams=3,
            early_stopping=True
        )
        optimized = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        improvements = []
        if len(optimized) < len(code):
            improvements.append("Reduced code length")
        
        return OptimizationResult(
            optimized_code=optimized,
            improvements=improvements,
            warnings=[]
        )
    except Exception as ex:
        logger.warning(f"AI optimization failed: {ex}")
        return optimize_code_fallback(code, language)

def optimize_code_fallback(code: str, language: str) -> OptimizationResult:
    """Basic code optimization fallback"""
    improvements = []
    warnings = ["Used fallback optimization - limited improvements"]
    
    if language == 'python':
        optimized = autopep8.fix_code(code)
        improvements.append("Applied PEP8 formatting")
    elif language in ['java', 'cpp', 'c']:
        optimized = re.sub(r'([{};])', r'\1\n', code)
        improvements.append("Improved brace and semicolon formatting")
    else:
        optimized = code
    
    return OptimizationResult(
        optimized_code=optimized,
        improvements=improvements,
        warnings=warnings
    )

# API Endpoints
@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_code_endpoint(request: CodeRequest):
    """Evaluate code execution and quality"""
    try:
        return evaluate_code(
            code=request.code,
            language=request.language,
            timeout=request.timeout
        )
    except Exception as ex:
        logger.error(f"Evaluation error: {ex}")
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/optimize", response_model=OptimizationResult)
async def optimize_code_endpoint(request: CodeRequest):
    """Optimize code using AI or fallback methods"""
    try:
        language = request.language or detect_language(request.code)
        return optimize_code_ai(request.code, language)
    except Exception as ex:
        logger.error(f"Optimization error: {ex}")
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

if _name_ == "_main_":
    import uvicorn
    setup_environment()
    uvicorn.run(app, host="0.0.0.0", port=8000)