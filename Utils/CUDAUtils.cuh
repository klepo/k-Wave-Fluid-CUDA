/**
 * @file        CUDAUtils.cuhh
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file with CUDA utils functions. This routines are to
 *              be inlined
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        22 March    2016, 15:25 (created) \n
 *              18 April    2016, 14:45 (revised)
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2014 Jiri Jaros, Beau Johnston
 * and Bradley Treeby
 *
 * This file is part of the k-Wave. k-Wave is free software: you can
 * redistribute it and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation, either version
 * 3 of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with k-Wave. If not, see http://www.gnu.org/licenses/.
 */


#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

/**/
#include <Parameters/CUDADeviceConstants.cuh>

/**
 * @variable CUDADeviceConstants
 * @brief    This variable holds basic simulation constants for GPU.
 * @details  This variable holds necessary simulation constants in the CUDA GPU
 *           memory.
 *           The variable is defined in TCUDADeviceConstants.cu
 */
extern __constant__ TCUDADeviceConstants CUDADeviceConstants;


//--------------------Index routines -----------------------------------------//

/** Get x coordinate for 1D CUDA block
 * @return X coordinate for 1D CUDA block
 */
inline __device__ unsigned int GetIndex()
{
  return threadIdx.x + blockIdx.x * blockDim.x;
}// end of GetX
//------------------------------------------------------------------------------

/**
 * Get X stride for 3D CUDA block
 * @return X stride for 3D CUDA block
 */
inline __device__ unsigned int GetStride()
{
  return blockDim.x * gridDim.x;
}// end of GetX_Stride
//------------------------------------------------------------------------------

/**
 * Get a 3D coordinates for a real matrix form a 1D index
 * @param  [in] i - index
 * @return  3d coordinates
 */
inline __device__ dim3 GetReal3DCoords(const unsigned int i)
{
  return dim3( i % CUDADeviceConstants.Nx,
              (i / CUDADeviceConstants.Nx) % CUDADeviceConstants.Ny,
               i / ( CUDADeviceConstants.Nx * CUDADeviceConstants.Ny));
}// end of GetReal3DCoords
//------------------------------------------------------------------------------


/**
 * Get a 3D coordinates for a complex matrix form a 1D index
 * @param  [in] i - index
 * @return  3d coordinates
 */
inline __device__ dim3 GetComplex3DCoords(const unsigned int i)
{
  return dim3( i % CUDADeviceConstants.Complex_Nx,
              (i / CUDADeviceConstants.Complex_Nx) % CUDADeviceConstants.Complex_Ny,
               i / ( CUDADeviceConstants.Complex_Nx * CUDADeviceConstants.Complex_Ny));
}// end of GetReal3DCoords
//------------------------------------------------------------------------------



//----------------------- Multiplication operators for float2 ----------------//

/**
 * Operator * for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a.x * b.x, a.y * b.y
 */
inline  __device__ float2 operator*(const float2 a,
                                    const float2 b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}// end of operator*
//------------------------------------------------------------------------------

/**
 * Operator * for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a.x * b, a.y * b
 */
inline  __device__ float2 operator*(const float2 a,
                                    const float b)
{
  return make_float2(a.x * b, a.y * b);
}// end of operator*
//------------------------------------------------------------------------------

/**
 * Operator * for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a * b.x, a * b.y
 */
inline  __device__ float2 operator*(const float b,
                                    const float2 a)
{
  return make_float2(b * a.x, b * a.y);
}// end of operator*
//------------------------------------------------------------------------------


/**
 * Operator *= for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a.x *= b.x, a.y *= b.y
 */
inline  __device__ void operator*=(float2&      a,
                                   const float2 b)
{
  a.x *= b.x;
  a.y *= b.y;
}// end of operator*=
//------------------------------------------------------------------------------

/**
 * Operator *= for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a.x =* b, a.y =* b
 */
inline  __device__ void operator*=(float2&     a,
                                   const float b)
{
  a.x *= b;
  a.y *= b;
}// end of operator*=
//------------------------------------------------------------------------------

//----------------------- Addition operators for float2 ----------------------//

/**
 * Operator + for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a.x + b.x, a.y + b.y
 */
inline  __device__ float2 operator+(const float2 a,
                                    const float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}// end of operator+
//------------------------------------------------------------------------------

/**
 * Operator + for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a.x + b, a.y + b
 */
inline  __device__ float2 operator+(const float2 a,
                                    const float b)
{
  return make_float2(a.x + b, a.y + b);
}// end of operator+
//------------------------------------------------------------------------------

/**
 * Operator + for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a + b.x, a + b.y
 */
inline  __device__ float2 operator+(const float b,
                                    const float2 a)
{
  return make_float2(b + a.x, b + a.y);
}// end of operator*
//------------------------------------------------------------------------------


/**
 * Operator += for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a.x +=b.x, a.y += b.y
 */
inline  __device__ void operator+=(float2&      a,
                                   const float2 b)
{
  a.x += b.x;
  a.y += b.y;
}// end of operator+=
//------------------------------------------------------------------------------

/**
 * Operator += for float2 datatype (per element multiplication)
 * @param [in] a
 * @param [in] b
 * @return  a.x += b, a.y += b
 */
inline  __device__ void operator+=(float2&     a,
                                   const float b)
{
  a.x += b;
  a.y += b;
}// end of operator+=
//------------------------------------------------------------------------------


#endif /* CUDA_UTILS_CUH */

