/**
 * @file      CudaUtils.cuh
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file with CUDA utility functions. These routines are to be inlined.
 *
 * @version   kspaceFirstOrder3D 3.5
 *
 * @date      22 March     2016, 15:25 (created) \n
 *            16 August    2017, 15:45 (revised)
 *
 * @copyright Copyright (C) 2017 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the k-Wave Toolbox [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software  *Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */


#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <Parameters/CudaDeviceConstants.cuh>

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Variables ------------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
/**
 * This variable holds necessary simulation constants in the CUDA GPU memory. The variable is
 * defined in CUDADeviceConstants.cu.
 */
extern __constant__ CudaDeviceConstants cudaDeviceConstants;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Index routines ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief   Get global 1D coordinate for 1D CUDA block.
 * @details Get global 1D coordinate for 1D CUDA block.
 *
 * @return  x-coordinate for 1D CUDA block.
 */
inline __device__ unsigned int getIndex()
{
  return threadIdx.x + blockIdx.x * blockDim.x;
}// end of getIndex()
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief   Get x-stride for 3D CUDA block (for processing multiple grid points by a single thread).
 * @details Get x-stride for 3D CUDA block (for processing multiple grid points by a single thread).
 *
 * @return x stride for 3D CUDA block.
 */
inline __device__ unsigned int getStride()
{
  return blockDim.x * gridDim.x;
}// end of getStride
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief   Get 3D coordinates for a real matrix form a 1D index.
 * @details Get 3D coordinates for a real matrix form a 1D index.
 *
 * @param [in] i - index.
 * @return 3D coordinates.
 */
inline __device__ dim3 getReal3DCoords(const unsigned int i)
{
  return dim3( i % cudaDeviceConstants.nx,
              (i / cudaDeviceConstants.nx) % cudaDeviceConstants.ny,
               i / (cudaDeviceConstants.nx * cudaDeviceConstants.ny));
}// end of getReal3DCoords
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief   Get 3D coordinates for a complex matrix form a 1D index.
 * @details Get 3D coordinates for a complex matrix form a 1D index.
 *
 * @param [in] i - index.
 * @return 3D coordinates.
 */
inline __device__ dim3 getComplex3DCoords(const unsigned int i)
{
  return dim3( i % cudaDeviceConstants.nxComplex,
              (i / cudaDeviceConstants.nxComplex) % cudaDeviceConstants.nyComplex,
               i / (cudaDeviceConstants.nxComplex * cudaDeviceConstants.nyComplex));
}// end of getComplex3DCoords
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------- Multiplication operators for float2 ----------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief   Operator * for float2 datatype (per element multiplication).
 * @details Operator * for float2 datatype (per element multiplication).
 *
 * @param [in] a
 * @param [in] b
 * @return  a.x * b.x, a.y * b.y
 */
inline __device__ float2 operator*(const float2 a,
                                   const float2 b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}// end of operator*
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Operator * for float2 datatype (per element multiplication).
 * @details Operator * for float2 datatype (per element multiplication).
 *
 * @param [in] a
 * @param [in] b
 * @return  a.x * b, a.y * b
 */
inline __device__ float2 operator*(const float2 a,
                                   const float  b)
{
  return make_float2(a.x * b, a.y * b);
}// end of operator*
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief   Operator * for float2 datatype (per element multiplication).
 * @details Operator * for float2 datatype (per element multiplication).
 *
 * @param [in] a
 * @param [in] b
 * @return  a * b.x, a * b.y
 */
inline __device__ float2 operator*(const float  b,
                                   const float2 a)
{
  return make_float2(b * a.x, b * a.y);
}// end of operator*
//----------------------------------------------------------------------------------------------------------------------


/**
 * @brief   Operator *= for float2 datatype (per element multiplication).
 * @details Operator *= for float2 datatype (per element multiplication).
 *
 * @param [in,out] a
 * @param [in]     b
 * @return  a.x *= b.x, a.y *= b.y
 */
inline __device__ void operator*=(float2&      a,
                                  const float2 b)
{
  a.x *= b.x;
  a.y *= b.y;
}// end of operator*=
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief   Operator *= for float2 datatype (per element multiplication).
 * @details Operator *= for float2 datatype (per element multiplication).
 *
 * @param [in,out] a
 * @param [in] b
 * @return  a.x =* b, a.y =* b
 */
inline __device__ void operator*=(float2&     a,
                                  const float b)
{
  a.x *= b;
  a.y *= b;
}// end of operator*=
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------ Addition operators for float2 -------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief   Operator + for float2 datatype (per element multiplication).
 * @details Operator + for float2 datatype (per element multiplication).
 * @param [in] a
 * @param [in] b
 * @return  a.x + b.x, a.y + b.y
 */
inline __device__ float2 operator+(const float2 a,
                                   const float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}// end of operator+
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Operator + for float2 datatype (per element multiplication)
 * @details Operator + for float2 datatype (per element multiplication).
 *
 * @param [in] a
 * @param [in] b
 * @return  a.x + b, a.y + b
 */
inline __device__ float2 operator+(const float2 a,
                                   const float  b)
{
  return make_float2(a.x + b, a.y + b);
}// end of operator+
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief   Operator + for float2 datatype (per element multiplication).
 * @details Operator + for float2 datatype (per element multiplication).
 *
 * @param [in] a
 * @param [in] b
 * @return  a + b.x, a + b.y
 */
inline __device__ float2 operator+(const float  b,
                                   const float2 a)
{
  return make_float2(b + a.x, b + a.y);
}// end of operator+
//----------------------------------------------------------------------------------------------------------------------


/**
 * @brief   Operator += for float2 datatype (per element multiplication).
 * @details Operator += for float2 datatype (per element multiplication)
 *
 * @param [in,out] a
 * @param [in]     b
 * @return  a.x += b.x, a.y += b.y
 */
inline __device__ void operator+=(float2&      a,
                                  const float2 b)
{
  a.x += b.x;
  a.y += b.y;
}// end of operator+=
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief   Operator += for float2 datatype (per element multiplication).
 * @details Operator += for float2 datatype (per element multiplication).
 * @param [in,out] a
 * @param [in]     b
 * @return  a.x += b, a.y += b
 */
inline __device__ void operator+=(float2&     a,
                                  const float b)
{
  a.x += b;
  a.y += b;
}// end of operator+=
//----------------------------------------------------------------------------------------------------------------------

#endif /* CUDA_UTILS_H */
