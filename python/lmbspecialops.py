#
#  lmbspecialops - a collection of tensorflow ops
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import warnings

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
leaky_relu = lmbspecialopslib.leaky_relu
median3x3_downsample = lmbspecialopslib.median3x3_downsample
replace_nonfinite = lmbspecialopslib.replace_nonfinite
scale_invariant_gradient = lmbspecialopslib.scale_invariant_gradient
warp2d = lmbspecialopslib.warp2d


# wrap deprecated functions
def flow_to_depth(flow, intrinsics, rotation, translation, rotation_format=None, inverse_depth=None, normalized_flow=None, name=None, nowarning=False):
    if not nowarning:
        warnings.warn("flow_to_depth has incorrect behaviour but is kept for compatibility. Please use flow_to_depth2", DeprecationWarning, stacklevel=2)
    return lmbspecialopslib.flow_to_depth(
            flow=flow, 
            intrinsics=intrinsics, 
            rotation=rotation, 
            translation=translation, 
            rotation_format=rotation_format, 
            inverse_depth=inverse_depth, 
            normalized_flow=normalized_flow, 
            name=name)
flow_to_depth.__doc__ = lmbspecialopslib.flow_to_depth.__doc__


# register gradient ops
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



