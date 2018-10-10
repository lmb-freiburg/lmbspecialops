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
#include "helper.h"
#include "rotation_format.h"
#include "Eigen/Geometry"

// TODO 
// - different image size for output images


using namespace tensorflow;

REGISTER_OP("TransferKeyFrame2")
  .Attr("T: {float, double}")
  .Attr("rotation_format: {'matrix', 'quaternion', 'angleaxis3'} = 'angleaxis3'")
  .Attr("inverse_depth: bool = false")
  .Attr("depth_visible_threshold: float = 0.9")
  .Attr("pass_near: bool = true")
  .Input("image: T")
  .Input("depth: T")
  .Input("intrinsics: T")
  .Input("rotation: T")
  .Input("translation: T")
  .Output("image2: T")
  .Output("depth2: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
      using namespace ::tensorflow::shape_inference;
      ShapeHandle image_shape, depth_shape, intrinsics_shape, rotation_shape, translation_shape, output_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 3, &image_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &depth_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &intrinsics_shape));
      std::string rotation_format;
      c->GetAttr("rotation_format", &rotation_format);
      if( rotation_format == "matrix" )
      {
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 2, &rotation_shape));
      }
      else
      {
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &rotation_shape));
      }
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(4), 1, &translation_shape));

      if( c->RankKnown(intrinsics_shape) )
      {
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(intrinsics_shape,-1), 4, &d));
      }

      if( c->RankKnown(rotation_shape) )
      {
        DimensionHandle d;
        if( rotation_format == "matrix" )
        {
          TF_RETURN_IF_ERROR(c->WithValue(c->Dim(rotation_shape,-1), 3, &d));
          TF_RETURN_IF_ERROR(c->WithValue(c->Dim(rotation_shape,-2), 3, &d));
        }
        else if( rotation_format == "quaternion" )
        {
          TF_RETURN_IF_ERROR(c->WithValue(c->Dim(rotation_shape,-1), 4, &d));
        }
        else
        {
          TF_RETURN_IF_ERROR(c->WithValue(c->Dim(rotation_shape,-1), 3, &d));
        }
      }

      if( c->RankKnown(translation_shape) )
      {
        DimensionHandle d;
        TF_RETURN_IF_ERROR(c->WithValue(c->Dim(translation_shape,-1), 3, &d));
      }

      if( c->RankKnown(image_shape) && c->RankKnown(depth_shape))
      {
        // check if width and height are compatible
        ShapeHandle image_shape2d;
        c->Subshape(image_shape, -2, &image_shape2d);
        ShapeHandle depth_shape2d;
        c->Subshape(depth_shape, -2, &depth_shape2d);

        ShapeHandle merged_shape2d;
        TF_RETURN_IF_ERROR(c->Merge(image_shape2d, depth_shape2d, &merged_shape2d));
      }



      // check if N is compatible for all inputs
      DimensionHandle batch_dim = c->UnknownDim();

      {
        int rank = c->Rank(image_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-3; ++i )
        {
          c->Multiply(first_dim, c->Dim(image_shape,i), &first_dim);
        }
        TF_RETURN_IF_ERROR(c->Merge(first_dim, batch_dim, &batch_dim));
      }
      {
        int rank = c->Rank(depth_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-2; ++i )
        {
          c->Multiply(first_dim, c->Dim(depth_shape,i), &first_dim);
        }
        TF_RETURN_IF_ERROR(c->Merge(first_dim, batch_dim, &batch_dim));
      }
      {
        int rank = c->Rank(rotation_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-1; ++i )
        {
          c->Multiply(first_dim, c->Dim(rotation_shape,i), &first_dim);
        }
        TF_RETURN_IF_ERROR(c->Merge(first_dim, batch_dim, &batch_dim));
      }
      {
        int rank = c->Rank(translation_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-1; ++i )
        {
          c->Multiply(first_dim, c->Dim(translation_shape,i), &first_dim);
        }
        TF_RETURN_IF_ERROR(c->Merge(first_dim, batch_dim, &batch_dim));
      }
      {
        int rank = c->Rank(intrinsics_shape);
        DimensionHandle first_dim = c->MakeDim(1);
        for( int i = 0; i < rank-1; ++i )
        {
          c->Multiply(first_dim, c->Dim(intrinsics_shape,i), &first_dim);
        }
        TF_RETURN_IF_ERROR(c->Merge(first_dim, batch_dim, &batch_dim));
      }


      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    })
  .Doc(R"doc(
Transfers a key frame to a new viewpoint.

Forward warps an image and a depth map to another view point. 
The resulting image and depth map may contain holes.
The op assumes that the internal camera parameters are the same for both cameras.


image: 
  Image in NCHW format with at least one channel.

depth: 
  Depth map with absolute or inverse depth values
  The depth values describe the z distance to the optical center.

intrinsics:
  Camera intrinsics in the format [fx, fy, cx, cy].
  fx,fy are the normalized focal lengths.
  cx,cy is the normalized position of the principal point.

rotation:
  The relative rotation R of the second camera.
  RX+t transforms a point X to the camera coordinate system of the second camera

translation:
  The relative translation vector t of the second camera.
  RX+t transforms a point X to the camera coordinate system of the second camera

rotation_format:
  The format for the rotation. 
  Allowed values are {'matrix', 'quaternion', 'angleaxis3'}.
  'matrix' is a 3x3 rotation matrix in column major order
  'quaternion' is a quaternion given as [w,x,y,z], w is the coefficient for the real part.
  'angleaxis3' is a 3d vector with the rotation axis. The angle is encoded as magnitude.

inverse_depth:
  If true then the input depth map must use inverse depth values.
  The output uses the same format as the input

pass_near:
  If true depth samples pass that are closer to the camera.
  If false depth samples pass that are farther away from the camera.

image2: 
  Tensor with the same shape as the input image.

depth2:
  A tensor with the depth map.
)doc");


namespace {

template <class T, class VEC2T, class VEC3T, class MAT3T>
inline void compute_p2( 
    Eigen::Matrix<T,3,1>& p2,        // the flow vector
    const Eigen::MatrixBase<VEC2T>& p1,        // pixel coordinates in the first image with pixel centers at x.5, y.5
    const T depth,                         // depth of the point in the first image
    const Eigen::MatrixBase<VEC2T>& f,     // focal lengths
    const Eigen::MatrixBase<VEC2T>& inv_f, // reciprocal of focal lengths (1/f.x, 1/f.y)
    const Eigen::MatrixBase<VEC2T>& c,     // principal point coordinates, not pixel coordinates! pixel centers are shifted by 0.5
    const Eigen::MatrixBase<MAT3T>& R,     // rotation
    const Eigen::MatrixBase<VEC3T>& t      // translation
    ) 
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VEC2T, 2) 
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VEC3T, 3) 
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(MAT3T, 3, 3) 
  typedef Eigen::Matrix<T,2,1> Vec2;
  // compute the 3d point in the coordinate frame of the first camera
  Vec2 tmp2 = (p1-c).cwiseProduct(inv_f);

  // transform the point to the coordinate frame of the second camera
  p2 = R*(depth*tmp2.homogeneous()) + t;
  
  // project point to the image plane
  p2.x() = f.x()*(p2.x()/p2.z()) + c.x();
  p2.y() = f.y()*(p2.y()/p2.z()) + c.y();
}

} //namespace



template <class T>
class TransferKeyFrame2Op : public OpKernel 
{
public:
  explicit TransferKeyFrame2Op(OpKernelConstruction* construction)
    :OpKernel(construction)
  { 
    std::string R_format;
    OP_REQUIRES_OK(construction, construction->GetAttr("rotation_format", &R_format));
    if( R_format == "matrix" )
      rotation_format = MATRIX;
    else if( R_format == "quaternion" )
      rotation_format = QUATERNION;
    else
      rotation_format = ANGLEAXIS3;

    OP_REQUIRES_OK(construction, construction->GetAttr("inverse_depth", &inverse_depth));
    OP_REQUIRES_OK(construction, construction->GetAttr("depth_visible_threshold", &depth_visible_threshold));
    OP_REQUIRES_OK(construction, construction->GetAttr("pass_near", &pass_near));
  }

  void Compute( OpKernelContext* context ) override 
  {
    const Tensor& image_tensor = context->input(0);
    auto image = image_tensor.flat<T>();
    const TensorShape image_shape(image_tensor.shape());
    const int image_rank = image_shape.dims();

    const Tensor& depth_tensor = context->input(1);
    auto depth = depth_tensor.flat<T>();
    const TensorShape depth_shape(depth_tensor.shape());
    const int depth_rank = depth_shape.dims();

    const Tensor& intrinsics_tensor = context->input(2);
    auto intrinsics = intrinsics_tensor.flat<T>();
    const Tensor& rotation_tensor = context->input(3);
    auto rotation = rotation_tensor.flat<T>();
    const Tensor& translation_tensor = context->input(4);
    auto translation = translation_tensor.flat<T>();

    int64_t w_size = 1;
    for( int i = 0; i < depth_rank-2; ++i )
      w_size *= depth_shape.dim_size(i);

    Tensor* image2_tensor = 0; 
    OP_REQUIRES_OK(context, context->allocate_output(0, image_shape, &image2_tensor));
    auto image2 = image2_tensor->flat<T>();

    Tensor* depth2_tensor = 0; 
    OP_REQUIRES_OK(context, context->allocate_output(1, depth_shape, &depth2_tensor));
    auto depth2 = depth2_tensor->flat<T>();


    AllocatorAttributes attr;
    attr.set_on_host(true);
    attr.set_gpu_compatible(false);
    
    TensorShape sum_shape;
    sum_shape.AddDim(w_size);
    sum_shape.AddDim(image_shape.dim_size(image_rank-2));
    sum_shape.AddDim(image_shape.dim_size(image_rank-1));

    Tensor sum_tensor;
    OP_REQUIRES_OK(context, 
        context->allocate_temp( DataTypeToEnum<T>::v(), 
          sum_shape,
          &sum_tensor,
          attr));
    auto sum = sum_tensor.flat<T>();


    transferimage_cpu( 
        image2.data(),
        depth2.data(),
        sum.data(),
        image.data(),
        depth.data(),
        intrinsics.data(),
        rotation.data(),
        translation.data(),
        image_shape.dim_size(image_rank-1),
        image_shape.dim_size(image_rank-2),
        image_shape.dim_size(image_rank-3),
        w_size );
    
  }





  void transferimage_cpu( 
      T* image2, T* depth2, T* sum, const T* image, const T* depth, 
      const T* intrinsics,
      const T* rotation,
      const T* translation,
      int image_x_size, int image_y_size, int image_z_size, int image_w_size
      )
  {
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,4,1> Vec4;
    typedef Eigen::Matrix<T,3,3> Mat3;
    const T depth_threshold = depth_visible_threshold;
    const int xy_size = image_x_size*image_y_size;
    const int image_xyz_size = image_x_size*image_y_size*image_z_size;
    const int rotation_step = rotation_format_size(rotation_format);

    for( int i = 0; i < image_w_size*image_xyz_size; ++i )
      image2[i] = 0;

    for( int i = 0; i < image_w_size*xy_size; ++i )
      sum[i] = 0;

    if( inverse_depth )
    {
      for( int i = 0; i < image_w_size*xy_size; ++i )
        depth2[i] = (pass_near ? 0 : std::numeric_limits<T>::infinity());
    }
    else
    {
      for( int i = 0; i < image_w_size*xy_size; ++i )
        depth2[i] = (pass_near ? std::numeric_limits<T>::infinity() : 0);
    }


    for( int w = 0; w < image_w_size; ++w )
    {
      Vec2 f, c;
      f.x() = intrinsics[4*w+0]*image_x_size;
      f.y() = intrinsics[4*w+1]*image_y_size;
      c.x() = intrinsics[4*w+2]*image_x_size;
      c.y() = intrinsics[4*w+3]*image_y_size;
      Vec2 inv_f(1/f.x(), 1/f.y());

      Eigen::Map<const Vec3> t(translation+3*w);
      Mat3 R = convert_to_rotation_matrix(rotation+w*rotation_step, rotation_format);

      const T* image_ptr = image + w*image_xyz_size;
      const T* depth_ptr = depth + w*xy_size;
      T* image2_ptr = image2 + w*image_xyz_size;
      T* depth2_ptr = depth2 + w*xy_size;
      T* sum_ptr = sum + w*xy_size;
#define IMAGE(x,y,z) image_ptr[(z)*xy_size+(y)*image_x_size+(x)]
#define IMAGE2(x,y,z) image2_ptr[(z)*xy_size+(y)*image_x_size+(x)]
#define DEPTH(x,y) depth_ptr[(y)*image_x_size+(x)]
#define DEPTH2(x,y) depth2_ptr[(y)*image_x_size+(x)]
#define SUM(x,y) sum_ptr[(y)*image_x_size+(x)]
      for( int y = 0; y < image_y_size; ++y )
      for( int x = 0; x < image_x_size; ++x )
      {
        Vec3 p2;

        T d = DEPTH(x,y);
        if( inverse_depth )
          d = 1/d;
        if( d <= 0 || !std::isfinite(d) )
          continue;

        Vec2 p1(x+T(0.5),y+T(0.5));
        compute_p2(p2, p1, d, f, inv_f, c, R, t);
        if( !matrix_is_finite(p2) )
          continue;

        Eigen::Matrix<int,4,2> p2i;
        p2i.row(0) << p2.x()-T(0.5), p2.y()-T(0.5);
        p2i.row(1) << p2i(0,0)+1, p2i(0,1);
        p2i.row(2) << p2i(0,0), p2i(0,1)+1;
        p2i.row(3) << p2i(0,0)+1, p2i(0,1)+1;

        T d2 = (inverse_depth ? 1/p2.z() : p2.z());

        for( int i = 0; i < 4; ++i )
        if( p2i(i,0) >= 0 && p2i(i,0) < image_x_size && p2i(i,1) >= 0 && p2i(i,1) < image_y_size )
        {
          T old_d2 = DEPTH2(p2i(i,0),p2i(i,1));
          if( (inverse_depth && pass_near) || (!inverse_depth && !pass_near) )
          {
            DEPTH2(p2i(i,0),p2i(i,1)) = std::max(old_d2,d2);
          }
          else
          {
            DEPTH2(p2i(i,0),p2i(i,1)) = std::min(old_d2,d2);
          }
        }
      }
      for( int y = 0; y < image_y_size; ++y )
      for( int x = 0; x < image_x_size; ++x )
      {
        Vec3 p2;

        T d = DEPTH(x,y);
        if( inverse_depth )
          d = 1/d;
        if( d <= 0 || !std::isfinite(d) )
          continue;

        Vec2 p1(x+T(0.5),y+T(0.5));
        compute_p2(p2, p1, d, f, inv_f, c, R, t);
        if( !matrix_is_finite(p2) )
          continue;

        Eigen::Matrix<int,4,2> p2i;
        p2i.row(0) << p2.x()-T(0.5), p2.y()-T(0.5);
        p2i.row(1) << p2i(0,0)+1, p2i(0,1);
        p2i.row(2) << p2i(0,0), p2i(0,1)+1;
        p2i.row(3) << p2i(0,0)+1, p2i(0,1)+1;

        T a = std::max(T(0), p2.x()-T(0.5)-p2i(0,0));
        T b = std::max(T(0), p2.y()-T(0.5)-p2i(0,1));
        Vec4 weights( (1-a)*(1-b), a*(1-b), (1-a)*b, a*b );

        T d2 = (inverse_depth ? 1/p2.z() : p2.z());

        for( int i = 0; i < 4; ++i )
        if( p2i(i,0) >= 0 && p2i(i,0) < image_x_size && p2i(i,1) >= 0 && p2i(i,1) < image_y_size )
        {
          T old_d2 = DEPTH2(p2i(i,0),p2i(i,1));
          if( (inverse_depth && pass_near) || (!inverse_depth && !pass_near) )
          {
            if( d2 >= depth_threshold*old_d2 )
            {
              SUM(p2i(i,0),p2i(i,1)) += weights(i);
              for( int z = 0; z < image_z_size; ++z )
                IMAGE2(p2i(i,0),p2i(i,1),z) += weights(i)*IMAGE(x,y,z);
            }
          }
          else
          {
            if( depth_threshold*d2 <= old_d2 )
            {
              SUM(p2i(i,0),p2i(i,1)) += weights(i);
              for( int z = 0; z < image_z_size; ++z )
                IMAGE2(p2i(i,0),p2i(i,1),z) += weights(i)*IMAGE(x,y,z);
            }
          }
        }
      }

    }

    for( int w = 0; w < image_w_size; ++w )
    {
      T* image2_ptr = image2 + w*image_xyz_size;
      T* sum_ptr = sum + w*xy_size;
#define IMAGE2(x,y,z) image2_ptr[(z)*xy_size+(y)*image_x_size+(x)]
#define SUM(x,y) sum_ptr[(y)*image_x_size+(x)]
      for( int y = 0; y < image_y_size; ++y )
      for( int x = 0; x < image_x_size; ++x )
      {
        T s = SUM(x,y);
        if( s == 0 )
        {
          for( int z = 0; z < image_z_size; ++z )
            IMAGE2(x,y,z) = NAN;
        }
        else
        {
          for( int z = 0; z < image_z_size; ++z )
            IMAGE2(x,y,z) /= s;
        }
      }
    }


#undef IMAGE
#undef IMAGE2
#undef DEPTH
#undef DEPTH2
#undef SUM
  }


private:
  RotationFormat rotation_format;
  bool inverse_depth;
  float depth_visible_threshold;
  bool pass_near;

};


#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("TransferKeyFrame2")                                                 \
    .Device(DEVICE_CPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    TransferKeyFrame2Op<type>);                                                
REG_KB(float)
//REG_KB(double)
#undef REG_KB




