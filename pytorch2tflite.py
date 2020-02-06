import numpy as np

import tensorflow as tf

import torch
import torch.nn as nn

import onnx
from onnx_tf.backend import prepare


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 5)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    
model = Net()
model.eval()

# Convert from PyTorch to ONNX
input_names = ['input']
output_names = ['output']
dummy_input = torch.randn(1, 10, device='cpu')
torch.onnx.export(model, dummy_input, 'onnx/linear.onnx', 
                  verbose=False, input_names=input_names, output_names=output_names)

# Convert From ONNX to TF
onnx_model = onnx.load("onnx/linear.onnx")
tf_rep = prepare(onnx_model, strict=False)
tf_rep.export_graph("onnx/linear.pb")

# Test TF model on random input data.
input_tensor = np.random.random_sample((1, 10))
graph_def = tf.GraphDef()
graph_def.ParseFromString(open('onnx/linear.pb', 'rb').read())
tf.import_graph_def(graph_def, name='')

graph = tf.get_default_graph()

input_node = graph.get_tensor_by_name('input:0')
output_node = graph.get_tensor_by_name('output:0')

with tf.Session() as sess:
        output = sess.run([output_node], 
                          {input_node: input_tensor})
        print(output)

# Save TFlite model
graph_def_file = 'onnx/linear.pb'
input_arrays = ['input']
output_arrays = ['output']
converter = tf.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
with open('onnx/linear.tflite', 'wb') as f:
    f.write(tflite_model)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='onnx/linear.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test TFLite model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
