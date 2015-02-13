/**
 * @file        OutputStreamsCUDAKernels.cu
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 *
 * @brief       The implementation file a list of cuda kernels used for data
 *              sampling (output streams)
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        27 January   2015, 17:21 (created) \n
 *              09 February  2015, 20:39 (revised)
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


#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

#include <OutputHDF5Streams/OutputStreamsCUDAKernels.h>

using namespace std;

//----------------------------------------------------------------------------//
//                                Constants                                   //
//----------------------------------------------------------------------------//

constexpr int CUDANumOfThreads = 256;
constexpr int CUDANumOfBlocks  = 64;

//----------------------------------------------------------------------------//
//--------------------------------- Macros -----------------------------------//
//----------------------------------------------------------------------------//

/**
 * Check errors of the CUDA routines and print error.
 * @param [in] code  - error code of last routine
 * @param [in] file  - The name of the file, where the error was raised
 * @param [in] line  - What is the line
 * @param [in] Abort - Shall the code abort?
 * @todo - check this routine and do it differently!
 */
inline void gpuAssert(cudaError_t code,
                      string file,
                      int line)
{
  if (code != cudaSuccess)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage,"GPUassert: %s %s %d\n",cudaGetErrorString(code),file.c_str(),line);

    // Throw exception
     throw std::runtime_error(ErrorMessage);
  }
}// end of gpuAssert
//------------------------------------------------------------------------------

/// Define to get the usage easier
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


//----------------------------------------------------------------------------//
//----------------------------- Global routines ------------------------------//
//----------------------------------------------------------------------------//

/**
 * Get X coordinate for 3D CUDA block
 * @return X coordinate for 3D CUDA block
 */
inline __device__ size_t GetX()
{
  return threadIdx.x + blockIdx.x * blockDim.x;
}// end of GetX
//------------------------------------------------------------------------------

/**
 * Get X stride for 3D CUDA block
 * @return X stride for 3D CUDA block
 */
inline __device__ size_t GetX_Stride()
{
  return blockDim.x * gridDim.x;
}// end of GetX_Stride
//------------------------------------------------------------------------------




//----------------------------------------------------------------------------//
//----------------------------- Exported routines ----------------------------//
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//--------------------------- Index based sampling ---------------------------//
//----------------------------------------------------------------------------//



/**
 * CUDA kernel to sample raw data based on index sensor mask
 * @param [out] SamplingBuffer  - buffer to sample data in
 * @param [in] SourceData       - source matrix
 * @param [in] SensorData       - sensor mask
 * @param [in] NumberOfSamples  - number of sampled points
 */
__global__ void CUDASampleRawIndex(      float  * SamplingBuffer,
                                   const float  * SourceData,
                                   const size_t * SensorData,
                                   const size_t   NumberOfSamples)
{
  for (size_t i = GetX(); i < NumberOfSamples; i += GetX_Stride())
  {
    SamplingBuffer[i] = SourceData[SensorData[i]];
  }
}// end of CUDASampleRawIndex
//------------------------------------------------------------------------------


/**
 * Sample the source matrix using the index sensor mask and store data in buffer
 * @param [out] SamplingBuffer  - buffer to sample data in
 * @param [in] SourceData       - source matrix
 * @param [in] SensorData       - sensor mask
 * @param [in] NumberOfSamples  - number of sampled points
 */
void OutputStreamsCUDAKernels::SampleRawIndex(      float  * SamplingBuffer,
                                              const float  * SourceData,
                                              const size_t * SensorData,
                                              const size_t   NumberOfSamples)
{
  CUDASampleRawIndex<<<CUDANumOfBlocks,CUDANumOfThreads>>>
                   (SamplingBuffer,
                    SourceData,
                    SensorData,
                    NumberOfSamples);
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of SampleRawIndex
//------------------------------------------------------------------------------


/**
 * CUDA kernel to sample and aggregate the source matrix using the index sensor
 * mask and apply max operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] SensorData         - sensor mask
 * @param [in] NumberOfSamples    - number of sampled points
 */
__global__ void CUDASampleMaxIndex(      float  * SamplingBuffer,
                                   const float  * SourceData,
                                   const size_t * SensorData,
                                   const size_t   NumberOfSamples)
{
  for (size_t i = GetX(); i < NumberOfSamples; i += GetX_Stride())
  {
    SamplingBuffer[i] = max(SamplingBuffer[i], SourceData[SensorData[i]]);
  }
}// end of CUDASampleMaxIndex
//------------------------------------------------------------------------------


/**
 * Sample and aggregate the source matrix using the index sensor mask and apply max operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] SensorData         - sensor mask
 * @param [in] NumberOfSamples    - number of sampled points
 */
void OutputStreamsCUDAKernels::SampleMaxIndex(      float  * SamplingBuffer,
                                              const float  * SourceData,
                                              const size_t * SensorData,
                                              const size_t   NumberOfSamples)
{
 CUDASampleMaxIndex<<<CUDANumOfBlocks,CUDANumOfThreads>>>
                   (SamplingBuffer,
                    SourceData,
                    SensorData,
                    NumberOfSamples);
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of SampleMaxIndex
//------------------------------------------------------------------------------


/**
 * CUDA kernel to sample and aggregate the source matrix using the index sensor
 * mask and apply min operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] SensorData         - sensor mask
 * @param [in] NumberOfSamples    - number of sampled points
 */
__global__ void CUDASampleMinIndex(      float  * SamplingBuffer,
                                   const float  * SourceData,
                                   const size_t * SensorData,
                                   const size_t   NumberOfSamples)
{
  for (size_t i = GetX(); i < NumberOfSamples; i += GetX_Stride())
  {
     SamplingBuffer[i] = min(SamplingBuffer[i], SourceData[SensorData[i]]);
  }
}// end of CUDASampleMinIndex
//------------------------------------------------------------------------------



/**
 * Sample and aggregate the source matrix using the index sensor mask and apply
 * min operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] SensorData         - sensor mask
 * @param [in] NumberOfSamples    - number of sampled points
 */
void OutputStreamsCUDAKernels::SampleMinIndex(      float  * SamplingBuffer,
                      const float  * SourceData,
                      const size_t * SensorData,
                      const size_t   NumberOfSamples)
{
  CUDASampleMinIndex<<<CUDANumOfBlocks,CUDANumOfThreads>>>
                   (SamplingBuffer,
                    SourceData,
                    SensorData,
                    NumberOfSamples);
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of SampleMinIndex
//------------------------------------------------------------------------------



/**
 * CUDA kernel to sample and aggregate the source matrix using the index sensor
 * mask and apply rms (sum) operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] SensorData         - sensor mask
 * @param [in] NumberOfSamples    - number of sampled points
 */
__global__ void CUDASampleRMSIndex(      float  * SamplingBuffer,
                                   const float  * SourceData,
                                   const size_t * SensorData,
                                   const size_t   NumberOfSamples)
{
  for (size_t i = GetX(); i < NumberOfSamples; i += GetX_Stride())
  {
    SamplingBuffer[i] += (SourceData[SensorData[i]] * SourceData[SensorData[i]]);
  }
}// end of CUDASampleRMSIndex
//------------------------------------------------------------------------------



/**
 * Sample and aggregate the source matrix using the index sensor mask and apply
 * rms operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] SensorData         - sensor mask
 * @param [in] NumberOfSamples    - number of sampled points
 */
void OutputStreamsCUDAKernels::SampleRMSIndex(      float  * SamplingBuffer,
                                              const float  * SourceData,
                                              const size_t * SensorData,
                                              const size_t   NumberOfSamples)
{
  CUDASampleRMSIndex<<<CUDANumOfBlocks,CUDANumOfThreads>>>
                   (SamplingBuffer,
                    SourceData,
                    SensorData,
                    NumberOfSamples);
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of SampleRMSIndex
//------------------------------------------------------------------------------



//----------------------------------------------------------------------------//
//---------------------- Whole domain based sampling -------------------------//
//----------------------------------------------------------------------------//



/**
 * CUDA kernel to sample and aggregate the source matrix on the whole domain
 * and apply max operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] NumberOfSamples    - number of sampled points
 */
__global__ void CUDASampleMaxAll(      float  * SamplingBuffer,
                                   const float  * SourceData,
                                   const size_t   NumberOfSamples)
{
  for (size_t i = GetX(); i < NumberOfSamples; i += GetX_Stride())
  {
    SamplingBuffer[i] = max(SamplingBuffer[i], SourceData[i]);
  }
}// end of CUDASampleMaxAll
//------------------------------------------------------------------------------


/**
 * Sample and aggregate the source matrix on the whole domain and apply
 * max operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] NumberOfSamples    - number of sampled points
 */
void OutputStreamsCUDAKernels::SampleMaxAll(      float  * SamplingBuffer,
                                            const float  * SourceData,
                                            const size_t   NumberOfSamples)
{
  CUDASampleMaxAll<<<CUDANumOfBlocks,CUDANumOfThreads>>>
                 (SamplingBuffer,
                  SourceData,
                  NumberOfSamples);
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of SampleMaxAll
//------------------------------------------------------------------------------

/**
 * CUDA kernel to ample and aggregate the source matrix on the whole domain
 * and apply min operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] NumberOfSamples    - number of sampled points
 */
__global__ void CUDASampleMinAll(        float  * SamplingBuffer,
                                   const float  * SourceData,
                                   const size_t   NumberOfSamples)
{
  for (size_t i = GetX(); i < NumberOfSamples; i += GetX_Stride())
  {
    SamplingBuffer[i] = min(SamplingBuffer[i], SourceData[i]);
  }
}// end of CUDASampleMinAll
//------------------------------------------------------------------------------


/**
 * Sample and aggregate the source matrix on the whole domain and apply
 * min operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] NumberOfSamples    - number of sampled points
 */
void OutputStreamsCUDAKernels::SampleMinAll(      float  * SamplingBuffer,
                                            const float  * SourceData,
                                            const size_t   NumberOfSamples)
{
  CUDASampleMinAll<<<CUDANumOfBlocks,CUDANumOfThreads>>>
                 (SamplingBuffer,
                  SourceData,
                  NumberOfSamples);
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of SampleMinAll
//------------------------------------------------------------------------------


/**
 * CUDA kernel to sample and aggregate the source matrix on the whole domain
 * and apply rms  operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] NumberOfSamples    - number of sampled points
 */
__global__ void CUDASampleRMSAll(      float  * SamplingBuffer,
                                   const float  * SourceData,
                                   const size_t   NumberOfSamples)
{
  for (size_t i = GetX(); i < NumberOfSamples; i += GetX_Stride())
  {
    SamplingBuffer[i] += (SourceData[i] * SourceData[i]);
  }
}// end of CUDASampleRMSAll
//------------------------------------------------------------------------------

/**
 * Sample and aggregate the source matrix on the whole domain and apply
 * rms operator
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] NumberOfSamples    - number of sampled points
 */
void OutputStreamsCUDAKernels::SampleRMSAll(      float  * SamplingBuffer,
                                            const float  * SourceData,
                                            const size_t   NumberOfSamples)
{
  CUDASampleRMSAll<<<CUDANumOfBlocks,CUDANumOfThreads>>>
                 (SamplingBuffer,
                  SourceData,
                  NumberOfSamples);
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of SampleRMSAll
//------------------------------------------------------------------------------



//----------------------------------------------------------------------------//
//------------------------------ Post-processing -----------------------------//
//----------------------------------------------------------------------------//


/**
 * CUDA kernel to apply post-processing for RMS
 * @param [in, out] SamplingBuffer  - buffer to apply post-processing on
 * @param [in]      NumberOfSamples - number of ellements
 */
__global__ void CUDAPostProcessingRMS(      float * SamplingBuffer,
                                      const float   ScalingCoeff,
                                      const size_t  NumberOfSamples)
{
  for (size_t i = GetX(); i < NumberOfSamples; i += GetX_Stride())
  {
    SamplingBuffer[i] = sqrt(SamplingBuffer[i] * ScalingCoeff);
  }
}// end of CUDAPostProcessingRMS
//------------------------------------------------------------------------------


/**
 * Calculate post-processing for RMS
 * @param [in, out] SamplingBuffer  - buffer to apply post-processing on
 * @param [in]      ScalingCoeff    - Scaling coefficent
 * @param [in]      NumberOfSamples - number of elements
 */
void OutputStreamsCUDAKernels::PostProcessingRMS(      float  * SamplingBuffer,
                                                 const float    ScalingCoeff,
                                                 const size_t   NumberOfSamples)
{
  CUDAPostProcessingRMS<<<CUDANumOfBlocks,CUDANumOfThreads>>>
                       (SamplingBuffer,
                        ScalingCoeff,
                        NumberOfSamples);

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of PostProcessingRMS
//------------------------------------------------------------------------------