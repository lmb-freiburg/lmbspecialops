# lmbspecialops documentation

Op name | Summary
--------|--------
[depth_to_flow](#depth_to_flow) | Computes the optical flow for an image pair based on the depth map and camera motion.
[depth_to_normals](#depth_to_normals) | Computes the normal map from a depth map.
[flow_to_depth](#flow_to_depth) | Computes the depth from optical flow and the camera motion.
[leaky_relu](#leaky_relu) | Computes the leaky rectified linear unit activations y = max(leak*x,x).
[median3x3_downsample](#median3x3_downsample) | Downsamples an image with a 3x3 median filter with a stride of 2.
[replace_nonfinite](#replace_nonfinite) | Replaces all nonfinite elements.
[scale_invariant_gradient](#scale_invariant_gradient) | This op computes the scale invariant spatial gradient as described in the DeMoN paper.
[warp2d](#warp2d) | Warps the input with the given displacement vector field.

## depth_to_flow

```python
depth_to_flow(depth, intrinsics, rotation, translation, rotation_format='angleaxis3', inverse_depth=False, normalize_flow=False)
```

Computes the optical flow for an image pair based on the depth map and camera motion.

Takes the depth map of the first image and the relative camera motion of the
second image and computes the optical flow from the first to the second image.
The op assumes that the internal camera parameters are the same for both cameras.

*There is no corresponding gradient op*

#### Args

* ```depth```: depth map with absolute or inverse depth values
The depth values describe the z distance to the optical center.

* ```intrinsics```: camera intrinsics in the format [fx, fy, cx, cy].
fx,fy are the normalized focal lengths.
cx,cy is the normalized position of the principal point.

* ```rotation```: The relative rotation R of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera

* ```translation```: The relative translation vector t of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera

* ```rotation_format```: The format for the rotation.
Allowed values are {'matrix', 'quaternion', 'angleaxis3'}.
'matrix' is a 3x3 rotation matrix in column major order
'quaternion' is a quaternion given as [w,x,y,z], w is the coefficient for the real part.
'angleaxis3' is a 3d vector with the rotation axis. The angle is encoded as magnitude.

* ```inverse_depth```: If true then the input depth map must use inverse depth values.

* ```normalize_flow```: If true the returned optical flow will be normalized with respect to the
image dimensions.



#### Returns


A tensor with the optical flow from the first to the second image.
The format of the output tensor is NCHW with C=2; [batch, 2, height, width].

## depth_to_normals

```python
depth_to_normals(depth, intrinsics, inverse_depth=False)
```

Computes the normal map from a depth map.



*There is no corresponding gradient op*

#### Args

* ```depth```: depth map with absolute or inverse depth values
The depth values describe the z distance to the optical center.

* ```intrinsics```: camera intrinsics in the format [fx, fy, cx, cy].
fx,fy are the normalized focal lengths.
cx,cy is the normalized position of the principal point.

* ```inverse_depth```: If true then the input depth map must use inverse depth values.



#### Returns


Normal map in the coordinate system of the camera.
The format of the output tensor is NCHW with C=3; [batch, 3, height, width].

## flow_to_depth

```python
flow_to_depth(flow, intrinsics, rotation, translation, rotation_format='angleaxis3', inverse_depth=False, normalized_flow=False)
```

Computes the depth from optical flow and the camera motion.

Takes the optical flow and the relative camera motion from the second camera to
compute a depth map.
The layer assumes that the internal camera parameters are the same for both
images.

*There is no corresponding gradient op*

#### Args

* ```flow```: optical flow normalized or in pixel units.

* ```intrinsics```: camera intrinsics in the format [fx, fy, cx, cy].
fx,fy are the normalized focal lengths.
cx,cy is the normalized position of the principal point.

* ```rotation```: The relative rotation R of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera

* ```translation```: The relative translation vector t of the second camera.
RX+t transforms a point X to the camera coordinate system of the second camera

* ```rotation_format```: The format for the rotation.
Allowed values are {'matrix', 'quaternion', 'angleaxis3'}.
'angleaxis3' is a 3d vector with the rotation axis.
The angle is encoded as magnitude.

* ```inverse_depth```: If true then the output depth map uses inverse depth values.

* ```normalized_flow```: If true then the input flow is expected to be normalized with respect to the
image dimensions.



#### Returns


A tensor with the depth for the first image.
The format of the output tensor is NCHW with C=1; [batch, 1, height, width].

## leaky_relu

```python
leaky_relu(input, leak=0.1)
```

Computes the leaky rectified linear unit activations y = max(leak*x,x).



*This op has a corresponding gradient implementation*

#### Args

* ```input```: Input tensor of any shape.

* ```leak```: The leak factor.



#### Returns


A tensor with the activation.

## median3x3_downsample

```python
median3x3_downsample(input)
```

Downsamples an image with a 3x3 median filter with a stride of 2.



*There is no corresponding gradient op*

#### Args

* ```input```: Tensor with at least rank 2.
The supported format is NCHW [batch, channels, height, width].



#### Returns


Downsampled tensor.

## replace_nonfinite

```python
replace_nonfinite(input, value=0.0)
```

Replaces all nonfinite elements.

Replaces nonfinite elements in a tensor with a specified value.
The corresponding gradient for replaced elements is 0.

*This op has a corresponding gradient implementation*

#### Args

* ```input```: Input tensor of any shape.

* ```value```: The value used for replacing nonfinite elements.



#### Returns


Tensor with all nonfinite values replaced with 'value'.

## scale_invariant_gradient

```python
scale_invariant_gradient(input, deltas=[1], weights=[1.0], epsilon=0.001)
```

This op computes the scale invariant spatial gradient as described in the DeMoN paper.

The x component is computed as:
  grad_x = sum_delta w*(u(x+delta,y) - u(x,y))/(|u(x+delta,y)| + |u(x,y)| + eps)

Note that this op does not distinguish between channels and batch size of the
input tensor. If the input tensor has more than one channel, then the resulting
batch size will be the product of the input batch size and the channels.
E.g. (bi,ci,hi,wi) -> (bi*ci, 2, h, w).

*This op has a corresponding gradient implementation*

#### Args

* ```input```: An input tensor with at least rank 2.

* ```deltas```: The pixel delta for the difference.
This vector must be the same length as weight.

* ```weights```: The weight factor for each difference.
This vector must be the same length as delta.

* ```epsilon```: epsilon value for avoiding division by zero



#### Returns


Tensor with the scale invariant spatial gradient.
The format of the output tensor is NCHW with C=2; [batch, 2, height, width].
The first channel is the x (width) component.

## warp2d

```python
warp2d(input, displacements, normalized=False, border_mode='clamp', border_value=0.0)
```

Warps the input with the given displacement vector field.



*There is no corresponding gradient op*

#### Args

* ```input```: Input tensor in the format NCHW with a minimum rank of 2.
For rank 2 tensors C == 1 is assumed.
For rank 3 tensors N == 1 is assumed.

* ```displacements```: The tensor storing the displacement vector field.
The format is NCHW with C=2 and the rank is at least 3.
The first channel is the displacement in x direction (width).
The second channel is the displacement in y direction (height).

* ```normalized```: If true then the displacement vectors are normalized with the width and height of the input.

* ```border_mode```: Defines how to handle values outside of the image.
'clamp': Coordinates will be clamped to the valid range.
'value' : Uses 'border_value' outside the image borders.

* ```border_value```: The value used outside the image borders.



#### Returns


The warped input tensor.

