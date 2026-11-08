"""Check released public API availability without loading implementation files."""
import warnings
warnings.filterwarnings("ignore")
import json
import transformers

result = {"version": transformers.__version__}
for name in ("MoonshineForConditionalGeneration", "MoonshineStreamingForConditionalGeneration", "AutoProcessor"):
    try:
        result[name] = getattr(transformers, name) is not None
    except Exception as error:
        result[name] = {"error_type": type(error).__name__}
print(json.dumps(result, sort_keys=True))
