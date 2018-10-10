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


REGISTER_OP("AngleAxisToRotationMatrix")
  .Attr("T: {float, double}")
  .Input("in: T")
  .Output("out: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle input_shape, output_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input_shape));

      if( c->RankKnown(input_shape) )
      {
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input_shape,-1), 3, &d));


        int rank = c->Rank(input_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-1; ++i )
        {
          c->Multiply(first_dim, c->Dim(input_shape,i), &first_dim);
        }
        c->Concatenate(c->MakeShape({first_dim}), c->MakeShape({3,3}), &output_shape);
        c->set_output(0, output_shape);
      }
      else
      {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    })
  .Doc(R"doc(
This op transforms an angle axis rotation representation with 3 elements into
a 3x3 rotation matrix.

in: A vector with 3 elements. Tensors with rank > 1 represent batches of vectors.
  The angle is encoded in the magnitude of the vector.

out: A tensor with a 3x3 rotation matrix.
)doc");



template <class T>
class AngleAxisToRotationMatrixOp : public OpKernel 
{
public:
  explicit AngleAxisToRotationMatrixOp(OpKernelConstruction* construction)
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
    for( int i = 0; i < input_rank-1; ++i )
      num *= input_shape.dim_size(i);
    output_shape.AddDim(num);
    output_shape.AddDim(3);
    output_shape.AddDim(3);
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    T* out = output.data();
    const T* in = input.data();
    for( int64_t i = 0; i < num; ++i )
    {
      Eigen::Map<Eigen::Matrix<T,3,3>> R(out+9*i);
      // transpose because eigen uses column major but we want to store the
      // matrix in row major
      R = convert_to_rotation_matrix(in+3*i, ANGLEAXIS3).transpose();
    }

    
  }


};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("AngleAxisToRotationMatrix")                                         \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    AngleAxisToRotationMatrixOp<type>);                                       
REG_KB(float)
REG_KB(double)
#undef REG_KB





REGISTER_OP("AngleAxisToRotationMatrixGrad")
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
This op implements the gradient for AngleAxisToRotationMatrix
)doc");


template <class T>
class AngleAxisToRotationMatrixGradOp : public OpKernel 
{
public:
  explicit AngleAxisToRotationMatrixGradOp(OpKernelConstruction* construction)
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
    for( int i = 0; i < input_rank-1; ++i )
      num *= input_shape.dim_size(i);
    Tensor* output_tensor = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();


    angleaxistorotationmatrix_grad_cpu(output.data(), input.data(), gradient.data(), num);
  }


  void angleaxistorotationmatrix_grad_cpu(
      T* out, const T* in, const T* gradient, int num )
  {
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,3,3> Mat3;

    for( int i = 0; i < num; ++i )
    {
      Eigen::Map<Vec3> aa_backprop(out+3*i);
      aa_backprop.setZero();

      // Note that Eigen uses column major but gradient is row major.
      // In the following code we switch row and column indices.
      Eigen::Map<const Mat3> grad(gradient+i*9);

      Eigen::Map<const Vec3> aa(in+i*3);
      T theta = aa.norm();
      if( theta < T(1e-6) ) // infinitesimal rotation
      {
        aa_backprop(0) = -grad(2,1) + grad(1,2);
        aa_backprop(1) = -grad(0,2) + grad(2,0);
        aa_backprop(2) = -grad(1,0) + grad(0,1);
        continue;
      }

      T inv_theta = 1/theta;
      T inv_theta3 = inv_theta*inv_theta*inv_theta;
      Vec3 w = aa/theta;

      T theta_backprop = 0;
      Vec3 w_backprop(0,0,0);

      T c = std::cos(theta);
      T s = std::sin(theta);

      // compute gradient with respect to angle theta_backprop, and axis w_backprop
      theta_backprop += (-s + s*w[0]*w[0])*grad(0,0);
      theta_backprop += (s*w[0]*w[1] - c*w[2])*grad(1,0);
      theta_backprop += (s*w[0]*w[2] + c*w[1])*grad(2,0);
      theta_backprop += (s*w[1]*w[0] + c*w[2])*grad(0,1);
      theta_backprop += (-s + s*w[1]*w[1])*grad(1,1);
      theta_backprop += (s*w[1]*w[2] - c*w[0])*grad(2,1);
      theta_backprop += (s*w[2]*w[0] - c*w[1])*grad(0,2);
      theta_backprop += (s*w[2]*w[1] + c*w[0])*grad(1,2);
      theta_backprop += (-s + s*w[2]*w[2])*grad(2,2);
      w_backprop(0) += (2*(-c + 1)*w[0])*grad(0,0);
      w_backprop(0) += ((-c + 1)*w[1])*grad(1,0);
      w_backprop(0) += ((-c + 1)*w[2])*grad(2,0);
      w_backprop(0) += ((-c + 1)*w[1])*grad(0,1);
      w_backprop(0) += (-s)*grad(2,1);
      w_backprop(0) += ((-c + 1)*w[2])*grad(0,2);
      w_backprop(0) += (s)*grad(1,2);
      w_backprop(1) += ((-c + 1)*w[0])*grad(1,0);
      w_backprop(1) += (s)*grad(2,0);
      w_backprop(1) += ((-c + 1)*w[0])*grad(0,1);
      w_backprop(1) += (2*(-c + 1)*w[1])*grad(1,1);
      w_backprop(1) += ((-c + 1)*w[2])*grad(2,1);
      w_backprop(1) += (-s)*grad(0,2);
      w_backprop(1) += ((-c + 1)*w[2])*grad(1,2);
      w_backprop(2) += (-s)*grad(1,0);
      w_backprop(2) += ((-c + 1)*w[0])*grad(2,0);
      w_backprop(2) += (s)*grad(0,1);
      w_backprop(2) += ((-c + 1)*w[1])*grad(2,1);
      w_backprop(2) += ((-c + 1)*w[0])*grad(0,2);
      w_backprop(2) += ((-c + 1)*w[1])*grad(1,2);
      w_backprop(2) += (2*(-c + 1)*w[2])*grad(2,2);

      // now compute the gradient with respect to the 3d angle axis vector
      aa_backprop(0) += aa(0)*inv_theta*theta_backprop;
      aa_backprop(1) += aa(1)*inv_theta*theta_backprop;
      aa_backprop(2) += aa(2)*inv_theta*theta_backprop;

      aa_backprop(0) += (inv_theta - aa(0)*aa(0)*inv_theta3)*w_backprop(0);
      aa_backprop(0) += - aa(0)*aa(1)*inv_theta3*w_backprop(1);
      aa_backprop(0) += - aa(0)*aa(2)*inv_theta3*w_backprop(2);

      aa_backprop(1) += - aa(1)*aa(0)*inv_theta3*w_backprop(0);
      aa_backprop(1) += (inv_theta - aa(1)*aa(1)*inv_theta3)*w_backprop(1);
      aa_backprop(1) += - aa(1)*aa(2)*inv_theta3*w_backprop(2);

      aa_backprop(2) += - aa(2)*aa(0)*inv_theta3*w_backprop(0);
      aa_backprop(2) += - aa(2)*aa(1)*inv_theta3*w_backprop(1);
      aa_backprop(2) += (inv_theta - aa(2)*aa(2)*inv_theta3)*w_backprop(2);
    }
  }


};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("AngleAxisToRotationMatrixGrad")                                     \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    AngleAxisToRotationMatrixGradOp<type>);                                   
REG_KB(float)
REG_KB(double)
#undef REG_KB

