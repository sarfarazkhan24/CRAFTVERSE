import ast

def inspect_model_script(script_content: str):
    tree = ast.parse(script_content)
    classes = []
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Check if it inherits from nn.Module
            is_nn_module = False
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == 'Module':
                    is_nn_module = True
                elif isinstance(base, ast.Attribute) and base.attr == 'Module':
                    is_nn_module = True
                    
            if not is_nn_module:
                # Be lenient, maybe they imported differently
                is_nn_module = True # Let's assume all classes might be the model
                
            init_args = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    for arg in item.args.args:
                        if arg.arg != 'self' and arg.arg not in ['args', 'kwargs']:
                            init_args.append(arg.arg)
                            
            classes.append({
                "name": node.name,
                "args": init_args
            })
            
    return classes
