import torch
from getModel import getEDVRSmallModel


torch_model = getEDVRSmallModel()

batch_size = 1
input_shape = (1, 5, 3, 64, 64)
device = torch.device('cuda')



torch_model.eval()
torch_model.to(device)

x = torch.randn(input_shape)
x = x.to(device)
export_onnx_file = 'outNetwork.onnx'

# torch.onnx.export(torch_model,
# 				  x,
# 				  export_onnx_file,
# 				  opset_version=10,
# 				  do_constant_folding=True,
# 				  input_names=["input"],
# 				  output_names=["output"],
# 				  dynamic_axes=["input":{0:"batch_size"}, 
# 				  				"output":{0:batch_size}])

input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
output_names = ["output1"]

torch.onnx.export(torch_model,
				  x,
				  export_onnx_file,
                                  opset_version=11,
				  input_names = input_names,
				  output_names = output_names)

