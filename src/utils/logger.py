import time
import inspect
import os

def log_step(message, start_time=None):
    """
    Traces execution by printing the relative file path and line number of the caller,
    along with the duration if a start_time is provided.
    """
    frame = inspect.stack()[1]
    abs_path = os.path.abspath(frame.filename)
    
    # Resolve relative path from project root 'ai_dw_nlp_poc'
    if 'ai_dw_nlp_poc' in abs_path:
        parts = abs_path.split('ai_dw_nlp_poc')
        rel_path = parts[-1].lstrip('\\/').replace('\\', '/')
    else:
        try:
            rel_path = os.path.relpath(abs_path).replace('\\', '/')
        except Exception:
            rel_path = os.path.basename(abs_path)
            
    lineno = frame.lineno
    duration_str = ""
    if start_time is not None:
        duration = time.time() - start_time
        duration_str = f" (took {duration:.4f}s)"
        
    print(f"[{rel_path}:{lineno}] {message}{duration_str}")
