import tensorflow as tf
from tensorflow.python.framework import ops
import os

if 'LMBSPECIALOPS_LIB' in os.environ:
    _lib_path = os.environ['LMBSPECIALOPS_LIB']
else: # try to find the lib in the build directory relative to this file
    _lib_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..', 'build','lib', 'lmbspecialops.so'))
if not os.path.isfile(_lib_path):
    raise ValueError('Cannot find lmbspecialops.so . Set the environment variable LMBSPECIALOPS_LIB to the path to lmbspecialops.so file')
lmbspecialopslib = tf.load_op_library(_lib_path)
print('Using {0}'.format(_lib_path), flush=True)

# create alias for each op
depth_to_flow = lmbspecialopslib.depth_to_flow
depth_to_normals = lmbspecialopslib.depth_to_normals
flow_to_depth = lmbspecialopslib.flow_to_depth
leaky_relu = lmbspecialopslib.leaky_relu
median3x3_downsample = lmbspecialopslib.median3x3_downsample
replace_nonfinite = lmbspecialopslib.replace_nonfinite
scale_invariant_gradient = lmbspecialopslib.scale_invariant_gradient
warp2d = lmbspecialopslib.warp2d

@ops.RegisterGradient("ScaleInvariantGradient")
def _scale_invariant_gradient_grad(op, grad):
    return lmbspecialopslib.scale_invariant_gradient_grad(
            gradients=grad, 
            input=op.inputs[0], 
            deltas=op.get_attr('deltas'),
            weights=op.get_attr('weights'),
            epsilon=op.get_attr('epsilon') )


@ops.RegisterGradient("ReplaceNonfinite")
def _replace_nonfinite_grad(op, grad):
    return lmbspecialopslib.replace_nonfinite_grad(
            gradients=grad, 
            input=op.inputs[0] )


@ops.RegisterGradient("LeakyRelu")
def _leaky_relu_grad(op, grad):
    return lmbspecialopslib.leaky_relu_grad(
            gradients=grad, 
            input=op.inputs[0],
            leak=op.get_attr('leak'))

