import onnxruntime as ort

# Path to your ONNX model
model_path = "your_model.onnx"

# Create ONNX Runtime session with CPUExecutionProvider only
session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

# Verify the providers actually applied
print("ONNX Runtime Execution Providers available:", ort.get_available_providers())
print("ONNX Runtime Execution Providers in use:", session.get_providers())

# Now you can run inference as usual
# Example:
# inputs = {session.get_inputs()[0].name: your_input_array}
# outputs = session.run(None, inputs)
