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
#define EIGEN_USE_GPU
#include "config.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "helper.h"
#include "cuda_helper.h"
#include "rotation_format.h"
#include "Eigen/Core"
#include <cuda_runtime.h>

using namespace tensorflow;


namespace transferkeyframe2_internal
{
  __device__ __forceinline__ float atomicMin(float* address, float val)
  {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_int,
                      assumed, 
                      __float_as_int(fminf(__int_as_float(assumed),val)));
    } while (assumed != old);
    return __int_as_float(old);
  }

  __device__ __forceinline__ float atomicMax(float* address, float val)
  {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_int,
                      assumed, 
                      __float_as_int(fmaxf(__int_as_float(assumed),val)));
    } while (assumed != old);
    return __int_as_float(old);
  }


  template <class T, class VEC2T, class VEC3T, class MAT3T>
  __device__ inline void compute_p2( 
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





  /*template <class T, bool INVERSE_DEPTH>*/
  /*__global__ void transferdepth_kernel1(*/
      /*T* out, const T* depth,*/
      /*const T* intrinsics,*/
      /*const T* rotation,*/
      /*const T* translation,*/
      /*int depth_x_size, int depth_y_size, int depth_z_size, int depth_xy_size*/
      /*)*/
  /*{*/
    /*typedef Eigen::Matrix<T,2,1> Vec2;*/
    /*typedef Eigen::Matrix<T,3,1> Vec3;*/
    /*typedef Eigen::Matrix<T,3,3> Mat3;*/
    /*int z = blockIdx.z*blockDim.z + threadIdx.z;*/
    /*int y = blockIdx.y*blockDim.y + threadIdx.y;*/
    /*int x = blockIdx.x*blockDim.x + threadIdx.x;*/
    /*if( x >= depth_x_size || y >= depth_y_size || z >= depth_z_size )*/
      /*return;*/

    /*[>printf("%f %f %f %f\n", intrinsics[4*z+0],intrinsics[4*z+1],intrinsics[4*z+2], intrinsics[4*z+3]);<]*/
    
    /*Vec2 f, c;*/
    /*f.x() = intrinsics[4*z+0]*depth_x_size;*/
    /*f.y() = intrinsics[4*z+1]*depth_y_size;*/
    /*c.x() = intrinsics[4*z+2]*depth_x_size;*/
    /*c.y() = intrinsics[4*z+3]*depth_y_size;*/
    /*Vec2 inv_f(1/f.x(), 1/f.y());*/

    /*Eigen::Map<const Vec3> t(translation+3*z);*/
    /*Eigen::Map<const Mat3> R(rotation+9*z);*/
    /*printf("t=%f %f %f \n", t.x(), t.y(), t.z());*/
    /*printf("R=%f %f %f %f %f %f %f %f %f\n", R(0,0), R(0,1), R(0,2), R(1,0), R(1,1), R(1,2), R(2,0), R(2,1), R(2,2));*/

    /*const T* depthmap = depth+z*depth_xy_size;*/
    /*T* out_depth = out+z*depth_xy_size;*/
/*#define DEPTH(x,y) depthmap[(y)*depth_x_size+(x)]*/
/*#define OUT(x,y) out_depth[(y)*depth_x_size+(x)]*/
    /*{*/
      /*Vec3 p2;*/

      /*T d = DEPTH(x,y);*/
      /*if( INVERSE_DEPTH )*/
        /*d = 1/d;*/
      /*if( d > 0 && isfinite(d) )*/
      /*{*/
        /*Vec2 p1(x+T(0.5),y+T(0.5));*/
        /*compute_p2(p2, p1, d, f, inv_f, c, R, t);*/
        /*int x2 = std::floor(p2.x());*/
        /*int y2 = std::floor(p2.y());*/
        /*if( x2 >= 0 && x2 < depth_x_size && y2 >= 0 && y2 < depth_y_size )*/
        /*{*/
          /*if( INVERSE_DEPTH )*/
            /*atomicMax(&OUT(x2,y2),1/p2.z());*/
          /*else*/
            /*atomicMin(&OUT(x2,y2),p2.z());*/
        /*}*/
      /*}*/

    /*}*/
/*#undef DEPTH*/
/*#undef OUT*/
  /*}*/



  template <class T, bool INVERSE_DEPTH, bool PASS_NEAR>
  __global__ void transferdepth_kernel(
      T* out, const T* depth,
      const T* intrinsics,
      const T* rotation,
      const T* translation,
      int depth_x_size, int depth_y_size, int depth_z_size, int depth_xy_size
      )
  {
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,3,3> Mat3;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= depth_x_size || y >= depth_y_size || z >= depth_z_size )
      return;

    Vec2 f, c;
    f.x() = intrinsics[4*z+0]*depth_x_size;
    f.y() = intrinsics[4*z+1]*depth_y_size;
    c.x() = intrinsics[4*z+2]*depth_x_size;
    c.y() = intrinsics[4*z+3]*depth_y_size;
    Vec2 inv_f(1/f.x(), 1/f.y());

    Eigen::Map<const Vec3> t(translation+3*z);
    Eigen::Map<const Mat3> R(rotation+9*z);

    const T* depthmap = depth+z*depth_xy_size;
    T* out_depth = out+z*depth_xy_size;
#define DEPTH(x,y) depthmap[(y)*depth_x_size+(x)]
#define OUT(x,y) out_depth[(y)*depth_x_size+(x)]
    {
      Vec3 p2;

      T d = DEPTH(x,y);
      if( INVERSE_DEPTH )
        d = 1/d;
      if( d <= 0 || !isfinite(d) )
        return;

      Vec2 p1(x+T(0.5),y+T(0.5));
      compute_p2(p2, p1, d, f, inv_f, c, R, t);
      if( !isfinite(p2.x()) || !isfinite(p2.y()) || !isfinite(p2.z()) )
        return;

      Eigen::Matrix<int,4,2> p2i;
      p2i.row(0) << p2.x()-T(0.5), p2.y()-T(0.5);
      p2i.row(1) << p2i(0,0)+1, p2i(0,1);
      p2i.row(2) << p2i(0,0), p2i(0,1)+1;
      p2i.row(3) << p2i(0,0)+1, p2i(0,1)+1;

      /*T a = p2.x()-T(0.5)-p2i(0,0);*/
      /*T b = p2.y()-T(0.5)-p2i(0,1);*/
      /*Vec4 weights( (1-a)*(1-b), a*(1-b), (1-a)*b, a*b );*/

      T d2 = (INVERSE_DEPTH ? 1/p2.z() : p2.z());

      for( int i = 0; i < 4; ++i )
      if( p2i(i,0) >= 0 && p2i(i,0) < depth_x_size && p2i(i,1) >= 0 && p2i(i,1) < depth_y_size )
      {
        if( (INVERSE_DEPTH && PASS_NEAR) || (!INVERSE_DEPTH && !PASS_NEAR) )
        {
          atomicMax(&OUT(p2i(i,0),p2i(i,1)), d2);
        }
        else
        {
          atomicMin(&OUT(p2i(i,0),p2i(i,1)), d2);
        }
      }
    }
#undef DEPTH
#undef OUT
  }


  template <class T>
  void transferdepth_gpu( 
        const cudaStream_t& stream,
        T* depth2, 
        const T* depth, 
        const T* intrinsics,
        const T* rotation,
        const T* translation,
        int x_size, int y_size, int w_size,
        bool inverse_depth,
        bool pass_near )
  {
    dim3 block(32,4,1);
    dim3 grid;
    grid.x = divup(x_size,block.x);
    grid.y = divup(y_size,block.y);
    grid.z = divup(w_size,block.z);

    if( inverse_depth && pass_near )
    {
      transferdepth_kernel<T,true,true><<<grid,block,0,stream>>>(
          depth2, depth,
          intrinsics,
          rotation,
          translation,
          x_size, y_size, w_size, x_size*y_size);
    }
    else if( !inverse_depth && pass_near )
    {
      transferdepth_kernel<T,false,true><<<grid,block,0,stream>>>(
          depth2, depth,
          intrinsics,
          rotation,
          translation,
          x_size, y_size, w_size, x_size*y_size);
    }
    else if( inverse_depth && !pass_near )
    {
      transferdepth_kernel<T,true,false><<<grid,block,0,stream>>>(
          depth2, depth,
          intrinsics,
          rotation,
          translation,
          x_size, y_size, w_size, x_size*y_size);
    }
    else// if( !inverse_depth && !pass_near )
    {
      transferdepth_kernel<T,false,false><<<grid,block,0,stream>>>(
          depth2, depth,
          intrinsics,
          rotation,
          translation,
          x_size, y_size, w_size, x_size*y_size);
    }
    CHECK_CUDA_ERROR
  }

  template <class T>
  __global__ void setval_kernel( T* ptr, const T val, int x_size, int y_size, int z_size )
  {
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= x_size || y >= y_size || z >= z_size )
      return;
    ptr[z*y_size*x_size+y*x_size+x] = val;
  }

  template <class T>
  void setval_gpu( 
        const cudaStream_t& stream,
        T* out, T val,
        int x_size, int y_size, int z_size )
  {
    dim3 block(32,4,1);
    dim3 grid;
    grid.x = divup(x_size,block.x);
    grid.y = divup(y_size,block.y);
    grid.z = divup(z_size,block.z);

    setval_kernel<T><<<grid,block,0,stream>>>(
        out, val, x_size, y_size, z_size );
    CHECK_CUDA_ERROR
  }
  template void setval_gpu<float>(const cudaStream_t&, float*, float, int, int, int);


  template <class T>
  __global__ void normalize_image_kernel( T* image, const T* sum,
      int image_x_size, int image_y_size, int image_z_size, int image_w_size )
  {
    int w = blockIdx.z*blockDim.z + threadIdx.z;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= image_x_size || y >= image_y_size || w >= image_w_size )
      return;

    const int xy_size = image_x_size*image_y_size;
    const int xyz_size = xy_size*image_z_size;

    T s = sum[w*xy_size+y*image_x_size+x];

    if( s == 0 )
    {
      for( int z = 0; z < image_z_size; ++z )
        image[w*xyz_size+z*xy_size+y*image_x_size+x] = NAN;
    }
    else
    {
      for( int z = 0; z < image_z_size; ++z )
        image[w*xyz_size+z*xy_size+y*image_x_size+x] /= s;
    }
  }

  template <class T>
  void normalize_image_gpu( 
        const cudaStream_t& stream,
        T* image, T* sum,
        int x_size, int y_size, int z_size, int w_size )
  {
    dim3 block(32,4,1);
    dim3 grid;
    grid.x = divup(x_size,block.x);
    grid.y = divup(y_size,block.y);
    grid.z = divup(w_size,block.z);

    normalize_image_kernel<T><<<grid,block,0,stream>>>(
        image, sum, x_size, y_size, z_size, w_size );
    CHECK_CUDA_ERROR
  }
  template void normalize_image_gpu<float>(const cudaStream_t&, float*, float*, int, int, int, int);


  template <class T, bool INVERSE_DEPTH, bool PASS_NEAR>
  __global__ void transferimage_kernel(
      T* image2, T* sum, const T* depth2, const T* image, const T* depth,
      const T* intrinsics,
      const T* rotation,
      const T* translation,
      int image_x_size, int image_y_size, int image_z_size, int image_w_size,
      const T depth_threshold
      )
  {
    typedef Eigen::Matrix<T,2,1> Vec2;
    typedef Eigen::Matrix<T,3,1> Vec3;
    typedef Eigen::Matrix<T,4,1> Vec4;
    typedef Eigen::Matrix<T,3,3> Mat3;
    int w = blockIdx.z*blockDim.z + threadIdx.z;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if( x >= image_x_size || y >= image_y_size || w >= image_w_size )
      return;

    int xy_size = image_x_size*image_y_size;
    int image_xyz_size = image_z_size*xy_size;

    Vec2 f, c;
    f.x() = intrinsics[4*w+0]*image_x_size;
    f.y() = intrinsics[4*w+1]*image_y_size;
    c.x() = intrinsics[4*w+2]*image_x_size;
    c.y() = intrinsics[4*w+3]*image_y_size;
    Vec2 inv_f(1/f.x(), 1/f.y());

    Eigen::Map<const Vec3> t(translation+3*w);
    Eigen::Map<const Mat3> R(rotation+9*w);

    const T* image_ptr = image + w*image_xyz_size;
    const T* depth_ptr = depth + w*xy_size;
    T* image2_ptr = image2 + w*image_xyz_size;
    const T* depth2_ptr = depth2 + w*xy_size;
    T* sum_ptr = sum + w*xy_size;
#define IMAGE(x,y,z) image_ptr[(z)*xy_size+(y)*image_x_size+(x)]
#define IMAGE2(x,y,z) image2_ptr[(z)*xy_size+(y)*image_x_size+(x)]
#define DEPTH(x,y) depth_ptr[(y)*image_x_size+(x)]
#define DEPTH2(x,y) depth2_ptr[(y)*image_x_size+(x)]
#define SUM(x,y) sum_ptr[(y)*image_x_size+(x)]
    {
      Vec3 p2;

      T d = DEPTH(x,y);
      if( INVERSE_DEPTH )
        d = 1/d;
      if( d <= 0 || !isfinite(d) )
        return;

      Vec2 p1(x+T(0.5),y+T(0.5));
      compute_p2(p2, p1, d, f, inv_f, c, R, t);
      if( !isfinite(p2.x()) || !isfinite(p2.y()) || !isfinite(p2.z()) )
        return;

      Eigen::Matrix<int,4,2> p2i;
      p2i.row(0) << p2.x()-T(0.5), p2.y()-T(0.5);
      p2i.row(1) << p2i(0,0)+1, p2i(0,1);
      p2i.row(2) << p2i(0,0), p2i(0,1)+1;
      p2i.row(3) << p2i(0,0)+1, p2i(0,1)+1;

      T a = fmaxf(T(0), p2.x()-T(0.5)-p2i(0,0));
      T b = fmaxf(T(0), p2.y()-T(0.5)-p2i(0,1));
      Vec4 weights( (1-a)*(1-b), a*(1-b), (1-a)*b, a*b );

      T d2 = (INVERSE_DEPTH ? 1/p2.z() : p2.z());

      for( int i = 0; i < 4; ++i )
      if( p2i(i,0) >= 0 && p2i(i,0) < image_x_size && p2i(i,1) >= 0 && p2i(i,1) < image_y_size )
      {
        T old_d2 = DEPTH2(p2i(i,0),p2i(i,1));
        if( (INVERSE_DEPTH && PASS_NEAR) || (!INVERSE_DEPTH && !PASS_NEAR) )
        {
          if( d2 >= depth_threshold*old_d2 )
          {
            atomicAdd(&SUM(p2i(i,0),p2i(i,1)),weights(i));
            for( int z = 0; z < image_z_size; ++z )
              atomicAdd(&IMAGE2(p2i(i,0),p2i(i,1),z),weights(i)*IMAGE(x,y,z));
          }
        }
        else
        {
          if( depth_threshold*d2 <= old_d2 )
          {
            atomicAdd(&SUM(p2i(i,0),p2i(i,1)),weights(i));
            for( int z = 0; z < image_z_size; ++z )
              atomicAdd(&IMAGE2(p2i(i,0),p2i(i,1),z),weights(i)*IMAGE(x,y,z));
          }
        }
      }

    }
#undef IMAGE
#undef IMAGE2
#undef DEPTH
#undef DEPTH2
  }


  template <class T>
  void transferimage_gpu( 
        const cudaStream_t& stream,
        T* image2, 
        T* sum,
        const T* depth2, 
        const T* image, 
        const T* depth, 
        const T* intrinsics,
        const T* rotation,
        const T* translation,
        int image_x_size, int image_y_size, int image_z_size, int image_w_size,
        bool inverse_depth,
        float depth_visible_threshold,
        bool pass_near )
  {
    dim3 block(32,4,1);
    dim3 grid;
    grid.x = divup(image_x_size,block.x);
    grid.y = divup(image_y_size,block.y);
    grid.z = divup(image_w_size,block.z);

    if( inverse_depth && pass_near )
    {
      transferimage_kernel<T,true,true><<<grid,block,0,stream>>>(
          image2, sum, depth2, image, depth,
          intrinsics,
          rotation,
          translation,
          image_x_size, image_y_size, image_z_size, image_w_size,
          depth_visible_threshold);
    }
    else if( !inverse_depth && pass_near )
    {
      transferimage_kernel<T,false,true><<<grid,block,0,stream>>>(
          image2, sum, depth2, image, depth,
          intrinsics,
          rotation,
          translation,
          image_x_size, image_y_size, image_z_size, image_w_size,
          depth_visible_threshold);
    }
    else if( inverse_depth && !pass_near )
    {
      transferimage_kernel<T,true,false><<<grid,block,0,stream>>>(
          image2, sum, depth2, image, depth,
          intrinsics,
          rotation,
          translation,
          image_x_size, image_y_size, image_z_size, image_w_size,
          depth_visible_threshold);
    }
    else// if( !inverse_depth && !pass_near )
    {
      transferimage_kernel<T,false,false><<<grid,block,0,stream>>>(
          image2, sum, depth2, image, depth,
          intrinsics,
          rotation,
          translation,
          image_x_size, image_y_size, image_z_size, image_w_size,
          depth_visible_threshold);
    }
    CHECK_CUDA_ERROR
  }





}
using namespace transferkeyframe2_internal;






template <class T>
class TransferKeyFrame2Op_GPU : public OpKernel 
{
public:
  explicit TransferKeyFrame2Op_GPU(OpKernelConstruction* construction)
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
    /*auto rotation = rotation_tensor.flat<T>();*/
    const Tensor& translation_tensor = context->input(4);
    auto translation = translation_tensor.flat<T>();

    TensorShape image2_shape(image_shape);
    TensorShape depth2_shape(depth_shape);
    int64_t w_size = 1;
    for( int i = 0; i < depth_rank-2; ++i )
      w_size *= depth_shape.dim_size(i);

    Tensor* image2_tensor = 0; 
    OP_REQUIRES_OK(context, context->allocate_output(0, image2_shape, &image2_tensor));
    auto image2 = image2_tensor->flat<T>();

    Tensor* depth2_tensor = 0; 
    OP_REQUIRES_OK(context, context->allocate_output(1, depth2_shape, &depth2_tensor));
    auto depth2 = depth2_tensor->flat<T>();

    TensorShape sum_shape;
    sum_shape.AddDim(w_size);
    sum_shape.AddDim(image_shape.dim_size(image_rank-2));
    sum_shape.AddDim(image_shape.dim_size(image_rank-1));

    Tensor sum_tensor;
    OP_REQUIRES_OK(context, 
        context->allocate_temp( DataTypeToEnum<T>::v(), 
          sum_shape,
          &sum_tensor));
    auto sum = sum_tensor.flat<T>();

    auto device = context->eigen_gpu_device();
    
    cudaMemsetAsync(sum.data(), 0, sizeof(T)*sum_shape.num_elements(), device.stream());
    cudaMemsetAsync(image2.data(), 0, sizeof(T)*image2_shape.num_elements(), device.stream());
    if( (inverse_depth && pass_near) || (!inverse_depth && !pass_near) )
      cudaMemsetAsync(depth2.data(), 0, sizeof(T)*depth2_shape.num_elements(), device.stream());
    else
      setval_gpu(device.stream(), depth2.data(), 
            std::numeric_limits<T>::infinity(),
            depth_shape.dim_size(depth_rank-1),
            depth_shape.dim_size(depth_rank-2),
            w_size );


    if( rotation_format == MATRIX )
    {
      transferdepth_gpu( 
            device.stream(),
            depth2.data(),
            depth.data(),
            intrinsics.data(),
            rotation_tensor.flat<T>().data(),
            translation.data(),
            depth_shape.dim_size(depth_rank-1),
            depth_shape.dim_size(depth_rank-2),
            w_size,
            inverse_depth,
            pass_near );
      transferimage_gpu( 
            device.stream(),
            image2.data(),
            sum.data(),
            depth2.data(),
            image.data(),
            depth.data(),
            intrinsics.data(),
            rotation_tensor.flat<T>().data(),
            translation.data(),
            image_shape.dim_size(image_rank-1),
            image_shape.dim_size(image_rank-2),
            image_shape.dim_size(image_rank-3),
            w_size,
            inverse_depth,
            depth_visible_threshold,
            pass_near );
    }
    else if( rotation_format == ANGLEAXIS3 )
    {
      TensorShape rotmatrix_shape(rotation_tensor.shape());
      rotmatrix_shape.set_dim(rotmatrix_shape.dims()-1, 9);

      Tensor rotmatrix_tensor_gpu;
      OP_REQUIRES_OK(context, 
          context->allocate_temp( DataTypeToEnum<T>::v(), 
            rotmatrix_shape, 
            &rotmatrix_tensor_gpu));
      
      T *out_gpu = rotmatrix_tensor_gpu.flat<T>().data();
      const T *in_gpu = rotation_tensor.flat<T>().data();
      angleaxis_to_rotmatrix_gpu(device.stream(), out_gpu, in_gpu, w_size);

      transferdepth_gpu( 
            device.stream(),
            depth2.data(),
            depth.data(),
            intrinsics.data(),
            rotmatrix_tensor_gpu.flat<T>().data(),
            translation.data(),
            depth_shape.dim_size(depth_rank-1),
            depth_shape.dim_size(depth_rank-2),
            w_size,
            inverse_depth,
            pass_near );
      transferimage_gpu( 
            device.stream(),
            image2.data(),
            sum.data(),
            depth2.data(),
            image.data(),
            depth.data(),
            intrinsics.data(),
            rotmatrix_tensor_gpu.flat<T>().data(),
            translation.data(),
            image_shape.dim_size(image_rank-1),
            image_shape.dim_size(image_rank-2),
            image_shape.dim_size(image_rank-3),
            w_size,
            inverse_depth,
            depth_visible_threshold,
            pass_near );
    }
    else
    {
      // convert to rotation matrix on the cpu
      AllocatorAttributes attr;
      attr.set_on_host(true);
      attr.set_gpu_compatible(true);
      
      Tensor rotation_tensor_cpu;
      OP_REQUIRES_OK(context, 
          context->allocate_temp( DataTypeToEnum<T>::v(), 
            rotation_tensor.shape(), 
            &rotation_tensor_cpu,
            attr));

      TensorShape rotmatrix_shape(rotation_tensor.shape());
      rotmatrix_shape.set_dim(rotmatrix_shape.dims()-1, 9);
      Tensor rotmatrix_tensor_cpu;
      OP_REQUIRES_OK(context, 
          context->allocate_temp( DataTypeToEnum<T>::v(), 
            rotmatrix_shape, 
            &rotmatrix_tensor_cpu,
            attr));

      Tensor rotmatrix_tensor_gpu;
      OP_REQUIRES_OK(context, 
          context->allocate_temp( DataTypeToEnum<T>::v(), 
            rotmatrix_shape, 
            &rotmatrix_tensor_gpu));

      {
        typedef Eigen::Matrix<T,3,3> Mat3;
        const int step = rotation_format_size(rotation_format);
        const T *in_gpu = rotation_tensor.flat<T>().data();
        T *in_cpu = rotation_tensor_cpu.flat<T>().data();
        T *out_cpu = rotmatrix_tensor_cpu.flat<T>().data();
        T *out_gpu = rotmatrix_tensor_gpu.flat<T>().data();
        //device.memcpyDeviceToHost(in_cpu, in_gpu, sizeof(T)*w_size*step); // Is this async?
        cudaMemcpyAsync(in_cpu, in_gpu, sizeof(T)*w_size*step, cudaMemcpyDeviceToHost, device.stream() );
        cudaStreamSynchronize(device.stream());
        for( int i = 0; i < w_size; ++i )
        {
          Mat3 R = convert_to_rotation_matrix(in_cpu+step*i, rotation_format);
          Eigen::Map<Mat3> tmp(out_cpu+9*i);
          tmp = R;
        }
        //device.memcpyHostToDevice(out_gpu, out_cpu, sizeof(T)*w_size*9);
        cudaMemcpyAsync(out_gpu, out_cpu, sizeof(T)*w_size*9, cudaMemcpyHostToDevice, device.stream());

      }
      transferdepth_gpu( 
            device.stream(),
            depth2.data(),
            depth.data(),
            intrinsics.data(),
            rotmatrix_tensor_gpu.flat<T>().data(),
            translation.data(),
            depth_shape.dim_size(depth_rank-1),
            depth_shape.dim_size(depth_rank-2),
            w_size,
            inverse_depth,
            pass_near );
      transferimage_gpu( 
            device.stream(),
            image2.data(),
            sum.data(),
            depth2.data(),
            image.data(),
            depth.data(),
            intrinsics.data(),
            rotmatrix_tensor_gpu.flat<T>().data(),
            translation.data(),
            image_shape.dim_size(image_rank-1),
            image_shape.dim_size(image_rank-2),
            image_shape.dim_size(image_rank-3),
            w_size,
            inverse_depth,
            depth_visible_threshold,
            pass_near );
    }

    normalize_image_gpu(
        device.stream(),
        image2.data(),
        sum.data(),
        image_shape.dim_size(image_rank-1),
        image_shape.dim_size(image_rank-2),
        image_shape.dim_size(image_rank-3),
        w_size );

    
  }


private:
  RotationFormat rotation_format;
  bool inverse_depth;
  float depth_visible_threshold;
  bool pass_near;

};

#define REG_KB(type)                                                          \
REGISTER_KERNEL_BUILDER(                                                      \
    Name("TransferKeyFrame2")                                                  \
    .Device(DEVICE_GPU)                                                       \
    .TypeConstraint<type>("T"),                                               \
    TransferKeyFrame2Op_GPU<type>);                                            
REG_KB(float)
//REG_KB(double)
#undef REG_KB

