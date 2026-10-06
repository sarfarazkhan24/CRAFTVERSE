import os
import sys
import json
import subprocess
import shutil
import zipfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI(title="Abstainity API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def run_worker(payload: dict):
    """Executes the Abstainity pipeline in a sandboxed subprocess."""
    backend_dir = os.path.dirname(__file__)
    python_exe = sys.executable 
    try:
        proc = subprocess.run(
            [python_exe, "-m", "abstain.worker"],
            cwd=backend_dir,
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            timeout=300 # 5 min timeout
        )
        output = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")

        # User adapters may print diagnostics, so find the last JSON line rather
        # than assuming the subprocess's final byte is the worker response.
        res = None
        for line in reversed(output.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                res = candidate
                break

        if res is None:
            detail = stderr.strip() or output.strip() or "Worker produced no diagnostic output."
            raise RuntimeError(f"Worker crashed (exit code {proc.returncode}): {detail}")
            
        if res.get("status") == "error":
            raise RuntimeError(f"Worker Error: {res.get('message')}\nTrace: {res.get('trace')}")

        if proc.returncode != 0:
            detail = stderr.strip() or "Worker exited with a non-zero status."
            raise RuntimeError(f"Worker crashed (exit code {proc.returncode}): {detail}")
            
        return res
    except subprocess.TimeoutExpired:
        raise RuntimeError("Sandbox worker timed out after 5 minutes.")

from inspect_model import inspect_model_script
from internal_adapter_gen import generate_adapter

@app.get("/")
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Abstainity API"}

@app.post("/api/inspect_model")
async def inspect_model(script: UploadFile = File(...)):
    content = (await script.read()).decode("utf-8")
    classes = inspect_model_script(content)
    return {"classes": classes, "status": "success"}

@app.post("/api/upload/model")
async def upload_model(
    preset: str = Form("imagenet"),
    model_type: str = Form("full_model"),
    class_name: Optional[str] = Form(None),
    kwargs_json: Optional[str] = Form(None),
    custom_size: Optional[str] = Form(None),
    custom_mean: Optional[str] = Form(None),
    custom_std: Optional[str] = Form(None),
    model_file: Optional[UploadFile] = File(None),
    script_file: Optional[UploadFile] = File(None),
    class_names_file: Optional[UploadFile] = File(None)
):
    if model_type == "test_script":
        if not script_file:
            raise HTTPException(status_code=400, detail="Inference script (test.py) is required for auto-configuration.")
    elif model_type == "weights_and_script":
        if not model_file or not script_file:
            raise HTTPException(status_code=400, detail="Both model weights file and architecture script are required.")
    else: # full_model
        if not model_file:
            raise HTTPException(status_code=400, detail="PyTorch model file (.pt / .pth) is required.")

    if model_type == "test_script" and script_file:
        script_path = os.path.join(WORKSPACE_DIR, "user_test.py")
        script_text = (await script_file.read()).decode("utf-8")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_text)
            
        if model_file:
            model_bytes = await model_file.read()
            weights_path = os.path.join(WORKSPACE_DIR, model_file.filename)
            with open(weights_path, "wb") as f:
                f.write(model_bytes)
            # Also save as model.pt so generated scripts can reliably locate it
            default_model_path = os.path.join(WORKSPACE_DIR, "model.pt")
            with open(default_model_path, "wb") as f:
                f.write(model_bytes)
                
        # AST logic (skipped for now, assuming fail)
        ast_success = False
        log_messages = []
        if not ast_success:
            log_messages.append("AST detection failed -> asking LLM")
            from abstain.llm_adapter import generate_adapter as llm_gen
            
            error_hint = None
            success = False
            for attempt in range(3):
                adapter_code = llm_gen(script_text, error_hint)
                if not adapter_code:
                    log_messages.append("LLM fallback unavailable or API error.")
                    break
                    
                adapter_path = os.path.join(WORKSPACE_DIR, "adapter.py")
                with open(adapter_path, "w", encoding="utf-8") as f:
                    f.write(adapter_code)
                    
                # Dry run
                test_script = """
import sys
sys.path.insert(0, '.')
import adapter
import torch

model = adapter.load_model()
try:
    dev = next(model.parameters()).device
except Exception:
    dev = torch.device('cpu')

x = torch.randn(1, 3, 224, 224, device=dev)
out = adapter.predict(model, x)
print('SUCCESS_MARKER')
"""
                test_script_path = os.path.join(WORKSPACE_DIR, "test_adapter.py")
                with open(test_script_path, "w", encoding="utf-8") as f:
                    f.write(test_script)
                    
                try:
                    result = subprocess.run(
                        [sys.executable, "test_adapter.py"],
                        cwd=WORKSPACE_DIR,
                        capture_output=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=45
                    )
                    if "SUCCESS_MARKER" in result.stdout:
                        success = True
                        log_messages.append("dry-run passed")
                        break
                    else:
                        error_hint = f"Output: {result.stdout}\nError: {result.stderr}"
                        log_messages.append(f"dry-run failed -> retry {attempt+1}/2")
                except subprocess.TimeoutExpired:
                    error_hint = "Execution timed out."
                    log_messages.append(f"dry-run failed (timeout) -> retry {attempt+1}/2")
                    
            if not success:
                log_messages.append("fallback to manual")
                return {"status": "error", "message": " | ".join(log_messages) + " | Auto-configuration failed. Please use the manual preset-dropdown flow."}
            else:
                return {"status": "success", "message": " | ".join(log_messages) + " | Model auto-configured successfully using LLM."}

    elif model_type == "weights_and_script" and script_file:
        # Save weights
        weights_path = os.path.join(WORKSPACE_DIR, "weights.pt")
        with open(weights_path, "wb") as f:
            f.write(await model_file.read())
        # Save user script
        script_path = os.path.join(WORKSPACE_DIR, "user_model.py")
        with open(script_path, "wb") as f:
            f.write(await script_file.read())
            
        kwargs_dict = json.loads(kwargs_json) if kwargs_json else {}
        adapter_code = generate_adapter("weights_and_script", preset, class_name, kwargs_dict, custom_mean, custom_std, custom_size)
    else:
        # Save full model
        model_path = os.path.join(WORKSPACE_DIR, "model.pt")
        with open(model_path, "wb") as f:
            f.write(await model_file.read())
            
        adapter_code = generate_adapter("full_model", preset, None, None, custom_mean, custom_std, custom_size)
        
    if class_names_file:
        cn_path = os.path.join(WORKSPACE_DIR, class_names_file.filename)
        with open(cn_path, "wb") as f:
            f.write(await class_names_file.read())
        
    adapter_path = os.path.join(WORKSPACE_DIR, "adapter.py")
    with open(adapter_path, "w", encoding="utf-8") as f:
        f.write(adapter_code)
        
    # Test the generated adapter by running load_model() in a subprocess
    test_script = """
import sys
sys.path.insert(0, '.')
import adapter
model = adapter.load_model()
print('SUCCESS_MARKER')
"""
    test_script_path = os.path.join(WORKSPACE_DIR, "test_adapter.py")
    with open(test_script_path, "w", encoding="utf-8") as f:
        f.write(test_script)
        
    try:
        result = subprocess.run(
            [sys.executable, "test_adapter.py"],
            cwd=WORKSPACE_DIR,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30
        )
        if "SUCCESS_MARKER" not in result.stdout:
            return {"status": "error", "message": f"Adapter generation failed to load model. Output: {result.stdout} {result.stderr}"}
            
        # Extract the auto-detected layer line
        detected_line = [line for line in result.stdout.split('\n') if 'Auto-detected FEATURE_LAYER' in line]
        msg = detected_line[0] if detected_line else "Model loaded, but no FEATURE_LAYER detected."
        return {"status": "success", "message": msg}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Model load timed out."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/upload/reference")
async def upload_reference(file: UploadFile = File(...)):
    zip_path = os.path.join(WORKSPACE_DIR, "reference.zip")
    with open(zip_path, "wb") as f:
        f.write(await file.read())
        
    ref_dir = os.path.join(WORKSPACE_DIR, "reference_data")
    if os.path.exists(ref_dir):
        shutil.rmtree(ref_dir)
    os.makedirs(ref_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ref_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Reference file is not a valid ZIP archive.") from exc
        
    # Optional: flatten if zip contains a single root folder
    extracted_items = os.listdir(ref_dir)
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(ref_dir, extracted_items[0])):
        inner_dir = os.path.join(ref_dir, extracted_items[0])
        for item in os.listdir(inner_dir):
            shutil.move(os.path.join(inner_dir, item), ref_dir)
        os.rmdir(inner_dir)
        
    return {"status": "success", "message": "Reference data extracted."}

@app.post("/api/fit")
async def fit():
    script_path = os.path.join(WORKSPACE_DIR, "adapter.py")
    ref_dir = os.path.join(WORKSPACE_DIR, "reference_data")
    config_path = os.path.join(WORKSPACE_DIR, "config.json")

    if not os.path.isfile(script_path):
        raise HTTPException(status_code=400, detail="Configure a model before fitting.")
    if not os.path.isdir(ref_dir):
        raise HTTPException(status_code=400, detail="Upload reference data before fitting.")

    payload = {
        "command": "fit",
        "script_path": script_path,
        "reference_dir": ref_dir,
        "output_config": config_path
    }
    
    try:
        res = run_worker(payload)
        return {"status": "success", "message": res.get("message")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    script_path = os.path.join(WORKSPACE_DIR, "adapter.py")
    config_path = os.path.join(WORKSPACE_DIR, "config.json")
    
    if not os.path.isfile(script_path) or os.path.getsize(script_path) == 0:
        raise HTTPException(status_code=400, detail="Please upload and configure a model (Step 1) before testing images.")
    
    img_path = os.path.join(WORKSPACE_DIR, "test_input.jpg")
    with open(img_path, "wb") as f:
        f.write(await image.read())
        
    payload = {
        "command": "predict",
        "script_path": script_path,
        "image_path": img_path,
        "config_path": config_path
    }
    
    try:
        res = run_worker(payload)
        return res["result"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
