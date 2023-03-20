import torch
from convert_data import FeatureGen
from model import ASLModel

sample_input = torch.rand((50, 543, 3))
onnx_feat_gen_path = 'feature_gen.onnx'
feature_converter = FeatureGen()
feature_converter.eval()

torch.onnx.export(
    feature_converter,                  # PyTorch Model
    sample_input,                    # Input tensor
    onnx_feat_gen_path,        # Output file (eg. 'output_model.onnx')
    opset_version=12,       # Operator support version
    input_names=['input'],   # Input tensor name (arbitary)
    output_names=['output'], # Output tensor name (arbitary)
    dynamic_axes={
        'input' : {0: 'input'}
    }
)

sample_input = torch.rand((1, 3258)).cuda()
onnx_model_path = 'asl_model.onnx'
model.eval()

torch.onnx.export(
    model,                  # PyTorch Model
    sample_input,                    # Input tensor
    onnx_model_path,        # Output file (eg. 'output_model.onnx')
    opset_version=12,       # Operator support version
    input_names=['input'],   # Input tensor name (arbitary)
    output_names=['output'], # Output tensor name (arbitary)
    dynamic_axes={
        'input' : {0: 'input'}
    }
)