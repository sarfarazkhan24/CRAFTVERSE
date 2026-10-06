import sys
import json
import traceback

# Import from the abstain package
from abstain.adapter import load_adapter
from abstain.pipeline import AbstainityPipeline
from abstain.fit import fit_reference_set

def run():
    try:
        input_data = sys.stdin.read()
        req = json.loads(input_data)
        
        command = req.get("command")
        script_path = req.get("script_path")
        
        adapter = load_adapter(script_path)
        model = adapter.load_model()
        model.eval()
        
        if command == "fit":
            ref_dir = req.get("reference_dir")
            out_config = req.get("output_config")
            fit_reference_set(adapter, model, ref_dir, out_config)
            # Output final JSON result cleanly
            print(json.dumps({"status": "success", "message": f"Fitted params saved to {out_config}"}))
            
        elif command == "predict":
            img_path = req.get("image_path")
            config_path = req.get("config_path")
            
            x = adapter.preprocess(img_path)
            pipeline = AbstainityPipeline(config_path)
            result = pipeline.predict(adapter, model, x)
            print(json.dumps({"status": "success", "result": result}))
            
        else:
            print(json.dumps({"status": "error", "message": f"Unknown command {command}"}))
            
    except Exception as e:
        # Capture and print exception safely for API to read
        print(json.dumps({"status": "error", "message": str(e), "trace": traceback.format_exc()}))
        sys.exit(1)

if __name__ == "__main__":
    run()
