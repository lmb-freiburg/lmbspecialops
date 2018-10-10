//
//  lmbspecialops - a collection of tensorflow ops
//  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include "config.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "rotation_format.h"

using namespace tensorflow;


REGISTER_OP("RotationMatrixToAngleAxis")
  .Attr("T: {float, double}")
  .Input("in: T")
  .Output("out: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle input_shape, output_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));

      if( c->RankKnown(input_shape) )
      {
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input_shape,-1), 3, &d));
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input_shape,-2), 3, &d));


        int rank = c->Rank(input_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-2; ++i )
        {
          c->Multiply(first_dim, c->Dim(input_shape,i), &first_dim);
        }
        c->Concatenate(c->MakeShape({first_dim}), c->MakeShape({3}), &output_shape);
        c->set_output(0, output_shape);
      }
      else
      {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    })
  .Doc(R"doc(
This op transforms a 3x3 rotation matrix to an angle axis rotation representation
with 3 elements.

in: A 3x3 matrix. Tensors with rank > 2 represent batches of matrices.

out: A vector with 3 elements. The rotation angle is encoded in the length.
)doc");



template <class T>
class RotationMatrixToAngleAxisOp : public OpKernel 
{
public:
  explicit RotationMatrixToAngleAxisOp(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());
    const int input_rank = input_shape.dims();

    TensorShape output_shape;
    int64_t num = 1;
    for( int i = 0; i < input_rank-2; ++i )
      num *= input_shape.dim_size(i);
    output_shape.AddDim(num);
    output_shape.AddDim(3);
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,3,3> Mat3;

    T* out = output.data();
    const T* in = input.data();
    for( int64_t i = 0; i < num; ++i )
    {
      // Don't use Eigen::AngleAxis here. We want to match the gradient operation
      Eigen::Map<const Mat3> R(in+9*i);
      Eigen::Map<Vec3> aa(out+i*3);

      T tmp = (R.trace()-1)*T(0.5);
      T theta = std::acos(std::min(T(1),std::max(T(-1),tmp)));
      if( theta < T(1e-6) )
      {
        aa.setZero();
      }
      else
      {
        // row and column switched ??????? because of eigen column major?
        Vec3 w( R(1,2)-R(2,1), R(2,0)-R(0,2), R(0,1)-R(1,0) );
        w *= 1/(2*std::sin(theta));
        aa = w*theta;
       }
    }

    
  }


};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("RotationMatrixToAngleAxis")                                         \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    RotationMatrixToAngleAxisOp<type>);                                       
REG_KB(float)
REG_KB(double)
#undef REG_KB





REGISTER_OP("RotationMatrixToAngleAxisGrad")
  .Attr("T: {float, double}")
  .Input("gradients: T")
  .Input("in: T")
  .Output("backprops: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
  .Doc(R"doc(
This op implements the gradient for RotationMatrixToAngleAxis
)doc");


template <class T>
class RotationMatrixToAngleAxisGradOp : public OpKernel 
{
public:
  explicit RotationMatrixToAngleAxisGradOp(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& gradient_tensor = context->input(0);
    auto gradient = gradient_tensor.flat<T>();
    const TensorShape gradient_shape(gradient_tensor.shape());
    
    const Tensor& input_tensor = context->input(1);
    auto input = input_tensor.flat<T>();
    const TensorShape input_shape(input_tensor.shape());
    const int input_rank = input_shape.dims();

    TensorShape output_shape(input_shape);
    int64_t num = 1;
    for( int i = 0; i < input_rank-2; ++i )
      num *= input_shape.dim_size(i);
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();


    rotationmatrixtoangleaxis_grad_cpu(output.data(), input.data(), gradient.data(), num);
  }


  void rotationmatrixtoangleaxis_grad_cpu(
      T* out, const T* in, const T* gradient, int num )
  {
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,3,3> Mat3;

    for( int i = 0; i < num; ++i )
    {
      // Note that Eigen uses column major but gradient is row major.
      // In the following code we switch row and column indices.
      Eigen::Map<Mat3> R_backprop(out+9*i);
      R_backprop.setZero();
      Eigen::Map<const Mat3> R(in+i*9);
      Eigen::Map<const Vec3> grad(gradient+i*3);

      T tmp = R.trace()-1;
      T theta = std::acos(std::min(T(1),std::max(T(-1),T(0.5)*tmp)));
      if( theta < T(1e-6) ) // infinitesimal rotation
      {
        R_backprop(1,0) += T(-0.5)*grad(2);
        R_backprop(2,0) += T(0.5)*grad(1);
        R_backprop(0,1) += T(0.5)*grad(2);
        R_backprop(2,1) += T(-0.5)*grad(0);
        R_backprop(0,2) += T(-0.5)*grad(1);
        R_backprop(1,2) += T(0.5)*grad(0);
        continue;
      }

      T x1 = tmp;
      T x2 = -T(1.0/4.0)*x1*x1 + 1;
      T x3 = std::sqrt(x2);
      T x4 = 1.0/x3;
      T x5 = -R(2,1) + R(1,2);
      T x6 = 1.0/x2;
      T x7 = theta;
      T x8 = T(1.0/8.0)*x1*x4*x5*x6*x7 - T(1.0/4.0)*x4*x4*x5;
      T x9 = R(2,0) - R(0,2);
      T x10 = (1.0/8.0)*x1*x4*x6*x7*x9 - T(1.0/4.0)*x4*x4*x9;
      T x11 = -R(1,0) + R(0,1);
      T x12 = T(1.0/8.0)*x1*x11*x4*x6*x7 - T(1.0/4.0)*x11*x4*x4;
      T x13 = T(1.0/2.0)*x7/x3;
      T x14 = -x13;

      R_backprop(0,0) += (x8)*grad(0);
      R_backprop(0,0) += (x10)*grad(1);
      R_backprop(0,0) += (x12)*grad(2);
      R_backprop(1,0) += (x14)*grad(2);
      R_backprop(2,0) += (x13)*grad(1);
      R_backprop(0,1) += (x13)*grad(2);
      R_backprop(1,1) += (x8)*grad(0);
      R_backprop(1,1) += (x10)*grad(1);
      R_backprop(1,1) += (x12)*grad(2);
      R_backprop(2,1) += (x14)*grad(0);
      R_backprop(0,2) += (x14)*grad(1);
      R_backprop(1,2) += (x13)*grad(0);
      R_backprop(2,2) += (x8)*grad(0);
      R_backprop(2,2) += (x10)*grad(1);
      R_backprop(2,2) += (x12)*grad(2);

    }
  }


};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("RotationMatrixToAngleAxisGrad")                                     \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    RotationMatrixToAngleAxisGradOp<type>);                                   
REG_KB(float)
REG_KB(double)
#undef REG_KB



