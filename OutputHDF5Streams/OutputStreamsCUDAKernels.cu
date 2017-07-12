/**
 * @file        OutputStreamsCUDAKernels.cu
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file of cuda kernels used for data sampling (output streams).
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        27 January   2015, 17:21 (created) \n
 *              12 July      2017, 13:48 (revised)
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2016 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see http://www.gnu.org/licenses/.
 */


#include <cuda.h>
#include <cuda_runtime.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>
#include <OutputHDF5Streams/OutputStreamsCUDAKernels.cuh>

#include <Parameters/Parameters.h>
#include <Logger/Logger.h>
#include <Utils/CudaUtils.cuh>




//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//-------------------------------------- Global routines -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Get Sampler CUDA Block size.
 *
 * @return CUDA block size
 */
int GetSamplerBlockSize()
{
  return TParameters::GetInstance().GetCudaParameters().getSamplerBlockSize1D();
}// end of GetSamplerBlockSize
//--------------------------------------------------------------------------------------------------


/**
 * Get sampler CUDA grid size.
 *
 * @return CUDA grid size
 */
int GetSamplerGridSize()
{
  return TParameters::GetInstance().GetCudaParameters().getSamplerGridSize1D();
}// end of GetSamplerGridSize
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//------------------------------------ Index mask sampling ---------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * CUDA kernel to sample data based on index sensor mask. The operator is given by
 * the template parameter.
 *
 * @param [out] samplingBuffer  - Buffer to sample data in
 * @param [in]  sourceData      - Source matrix
 * @param [in]  sensorData      - Sensor mask
 * @param [in]  nSamples        - Number of sampled points
 */
template <TBaseOutputHDF5Stream::TReduceOperator reduceOp>
__global__ void CUDASampleIndex(float*        samplingBuffer,
                                const float*  sourceData,
                                const size_t* sensorData,
                                const size_t  nSamples)
{
  for (auto i = getIndex(); i < nSamples; i += getStride())
  {
    switch (reduceOp)
    {
      case TBaseOutputHDF5Stream::TReduceOperator::NONE:
      {
        samplingBuffer[i] = sourceData[sensorData[i]];
        break;
      }
      case TBaseOutputHDF5Stream::TReduceOperator::RMS:
      {
        samplingBuffer[i] += (sourceData[sensorData[i]] * sourceData[sensorData[i]]);
        break;
      }
      case TBaseOutputHDF5Stream::TReduceOperator::MAX:
      {
        samplingBuffer[i] = max(samplingBuffer[i], sourceData[sensorData[i]]);
        break;
      }
      case TBaseOutputHDF5Stream::TReduceOperator::MIN:
      {
        samplingBuffer[i] = min(samplingBuffer[i], sourceData[sensorData[i]]);
        break;
      }
    }// switch
  } // for
}// end of CUDASampleIndex
//--------------------------------------------------------------------------------------------------

/**
 * Sample the source matrix using the index sensor mask and store data in buffer.
 *
 * @param [out] samplingBuffer   - Buffer to sample data in
 * @param [in] sourceData        - Source matrix
 * @param [in] sensorData        - Sensor mask
 * @param [in] nSamples          - Number of sampled points
 */
template<TBaseOutputHDF5Stream::TReduceOperator reduceOp>
void OutputStreamsCUDAKernels::SampleIndex(float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples)
{
  CUDASampleIndex<reduceOp>
                 <<<GetSamplerGridSize(),GetSamplerBlockSize()>>>
                 (samplingBuffer,
                  sourceData,
                  sensorData,
                  nSamples);

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of SampleIndex
//--------------------------------------------------------------------------------------------------

//------------------------------ Explicit instances of SampleIndex -------------------------------//
template
void OutputStreamsCUDAKernels::SampleIndex<TBaseOutputHDF5Stream::TReduceOperator::NONE>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);

template
void OutputStreamsCUDAKernels::SampleIndex<TBaseOutputHDF5Stream::TReduceOperator::RMS>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);

template
void OutputStreamsCUDAKernels::SampleIndex<TBaseOutputHDF5Stream::TReduceOperator::MAX>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);

template
void OutputStreamsCUDAKernels::SampleIndex<TBaseOutputHDF5Stream::TReduceOperator::MIN>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);


//------------------------------------------------------------------------------------------------//
//----------------------------------- Cuboid mask sampling ---------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Transform 3D coordinates within the cuboid into 1D coordinates within the matrix being sampled.
 *
 * @param [in] cuboidIdx         - Cuboid index
 * @param [in] topLeftCorner     - Top left corner
 * @param [in] bottomRightCorner - Bottom right corner
 * @param [in] matrixSize        - Size of the matrix being sampled
 * @return 1D index into the matrix being sampled
 */
inline __device__ size_t TransformCoordinates(const size_t cuboidIdx,
                                              const dim3&  topLeftCorner,
                                              const dim3&  bottomRightCorner,
                                              const dim3&  matrixSize)
{
  dim3 localPosition;
  // calculate the cuboid size
  dim3 cuboidSize(bottomRightCorner.x - topLeftCorner.x + 1,
                  bottomRightCorner.y - topLeftCorner.y + 1,
                  bottomRightCorner.z - topLeftCorner.z + 1);

  // find coordinates within the cuboid
  size_t slabSize = cuboidSize.x * cuboidSize.y;
  localPosition.z =  cuboidIdx / slabSize;
  localPosition.y = (cuboidIdx % slabSize) / cuboidSize.x;
  localPosition.x = (cuboidIdx % slabSize) % cuboidSize.x;

  // transform the coordinates to the global dimensions
  dim3 globalPosition(localPosition);
  globalPosition.z += topLeftCorner.z;
  globalPosition.y += topLeftCorner.y;
  globalPosition.x += topLeftCorner.x;

  // calculate 1D index
  return (globalPosition.z * matrixSize.x * matrixSize.y +
          globalPosition.y * matrixSize.x +
          globalPosition.x);
}// end of TransformCoordinates
//--------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to sample data inside one cuboid, operation is selected by a template parameter.

 * @param [out] samplingBuffer    - Buffer to sample data in
 * @param [in]  sourceData        - Source matrix
 * @param [in]  topLeftCorner     - Top left corner of the cuboid
 * @param [in]  bottomRightCorner - Bottom right corner of the cuboid
 * @param [in]  matrixSize        - Dimension sizes of the matrix being sampled
 * @param [in]  nSamples          - Number of grid points inside the cuboid
 */
template <TBaseOutputHDF5Stream::TReduceOperator reduceOp>
__global__ void CUDASampleCuboid(float*       samplingBuffer,
                                 const float* sourceData,
                                 const dim3   topLeftCorner,
                                 const dim3   bottomRightCorner,
                                 const dim3   matrixSize,
                                 const size_t nSamples)
{
  // iterate over all grid points
  for (auto i = getIndex(); i < nSamples; i += getStride())
  {
    auto Position = TransformCoordinates(i, topLeftCorner, bottomRightCorner, matrixSize);
    switch (reduceOp)
    {
      case TBaseOutputHDF5Stream::TReduceOperator::NONE:
      {
        samplingBuffer[i] = sourceData[Position];
        break;
      }
      case TBaseOutputHDF5Stream::TReduceOperator::RMS:
      {
        samplingBuffer[i] += (sourceData[Position] * sourceData[Position]);
        break;
      }
      case TBaseOutputHDF5Stream::TReduceOperator::MAX:
      {
        samplingBuffer[i] = max(samplingBuffer[i], sourceData[Position]);
        break;
      }
      case TBaseOutputHDF5Stream::TReduceOperator::MIN:
      {
        samplingBuffer[i] = min(samplingBuffer[i], sourceData[Position]);
        break;
      }
    }// switch
  } // for
}// end of CUDASampleCuboid
//-------------------------------------------------------------------------------------------------


/**
 * Sample data inside one cuboid and store it to buffer. The operation is given in the template
 * parameter.
 *
 * @param [out] samplingBuffer    - Buffer to sample data in
 * @param [in]  sourceData        - Source matrix
 * @param [in]  topLeftCorner     - Top left corner of the cuboid
 * @param [in]  bottomRightCorner - Bottom right corner of the cuboid
 * @param [in]  matrixSize        - Size of the matrix being sampled
 * @param [in]  nSamples          - Number of grid points inside the cuboid
 */
template<TBaseOutputHDF5Stream::TReduceOperator reduceOp>
void OutputStreamsCUDAKernels::SampleCuboid(float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples)
{
  CUDASampleCuboid<reduceOp>
                  <<<GetSamplerGridSize(),GetSamplerBlockSize()>>>
                  (samplingBuffer,
                   sourceData,
                   topLeftCorner,
                   bottomRightCorner,
                   matrixSize,
                   nSamples);
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of SampleCuboid
//--------------------------------------------------------------------------------------------------


//----------------------------- Explicit instances of SampleCuboid -------------------------------//
template
void OutputStreamsCUDAKernels::SampleCuboid<TBaseOutputHDF5Stream::TReduceOperator::NONE>
                                           (float*       samplingBuffer,
                                            const float*  sourceData,
                                            const dim3    topLeftCorner,
                                            const dim3    bottomRightCorner,
                                            const dim3    matrixSize,
                                            const size_t  nSamples);
template
void OutputStreamsCUDAKernels::SampleCuboid<TBaseOutputHDF5Stream::TReduceOperator::RMS>
                                           (float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples);
template
void OutputStreamsCUDAKernels::SampleCuboid<TBaseOutputHDF5Stream::TReduceOperator::MAX>
                                           (float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples);
template
void OutputStreamsCUDAKernels::SampleCuboid<TBaseOutputHDF5Stream::TReduceOperator::MIN>
                                           (float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples);

//------------------------------------------------------------------------------------------------//
//--------------------------------- Whole domain based sampling ----------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * CUDA kernel to sample and aggregate the source matrix on the whole domain and apply a reduce
 * operator.
 *
 * @param [in,out] samplingBuffer - Buffer to sample data in
 * @param [in]     sourceData     - Source matrix
 * @param [in]     nSamples       - Number of sampled points
 */
template <TBaseOutputHDF5Stream::TReduceOperator reduceOp>
__global__ void CUDASampleAll(float*       samplingBuffer,
                              const float* sourceData,
                              const size_t nSamples)
{
  for (size_t i = getIndex(); i < nSamples; i += getStride())
  {
    switch (reduceOp)
    {
      case TBaseOutputHDF5Stream::TReduceOperator::RMS:
      {
        samplingBuffer[i] += (sourceData[i] * sourceData[i]);
        break;
      }
      case TBaseOutputHDF5Stream::TReduceOperator::MAX:
      {
        samplingBuffer[i] = max(samplingBuffer[i], sourceData[i]);
        break;
      }
      case TBaseOutputHDF5Stream::TReduceOperator::MIN:
      {
        samplingBuffer[i] = min(samplingBuffer[i], sourceData[i]);
        break;
      }
    }
  }
}// end of CUDASampleAll
//--------------------------------------------------------------------------------------------------


/**
 * Sample and the whole domain and apply a defined operator.
 *
 * @param [in,out] samplingBuffer - Buffer to sample data in
 * @param [in]     sourceData     - Source matrix
 * @param [in]     nSamples       - Number of sampled points
 */
template<TBaseOutputHDF5Stream::TReduceOperator reduceOp>
void OutputStreamsCUDAKernels::SampleAll(float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples)
{
  CUDASampleAll<reduceOp>
               <<<GetSamplerGridSize(),GetSamplerBlockSize()>>>
               (samplingBuffer,
                sourceData,
                nSamples);
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of SampleMaxAll
//--------------------------------------------------------------------------------------------------


//------------------------------ Explicit instances of SampleAll ---------------------------------//
template
void OutputStreamsCUDAKernels::SampleAll<TBaseOutputHDF5Stream::TReduceOperator::RMS>
                                        (float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples);
template
void OutputStreamsCUDAKernels::SampleAll<TBaseOutputHDF5Stream::TReduceOperator::MAX>
                                        (float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples);
template
void OutputStreamsCUDAKernels::SampleAll<TBaseOutputHDF5Stream::TReduceOperator::MIN>
                                        (float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples);


//----------------------------------------------------------------------------//
//------------------------------ Post-processing -----------------------------//
//----------------------------------------------------------------------------//


/**
 * CUDA kernel to apply post-processing for RMS
 * @param [in, out] samplingBuffer - Buffer to apply post-processing on
 * @param [in]      scalingCoeff   - Scaling coeficinet for RMS
 * @param [in]      nSamples       - Number of elements
 */
__global__ void CUDAPostProcessingRMS(float*       samplingBuffer,
                                      const float  scalingCoeff,
                                      const size_t nSamples)
{
  for (size_t i = getIndex(); i < nSamples; i += getStride())
  {
    samplingBuffer[i] = sqrt(samplingBuffer[i] * scalingCoeff);
  }
}// end of CUDAPostProcessingRMS
//--------------------------------------------------------------------------------------------------


/**
 * Calculate post-processing for RMS.
 *
 * @param [in, out] samplingBuffer - Buffer to apply post-processing on
 * @param [in]      scalingCoeff   - Scaling coefficent
 * @param [in]      nSamples       - Number of elements
 */
void OutputStreamsCUDAKernels::PostProcessingRMS(float*       samplingBuffer,
                                                 const float  scalingCoeff,
                                                 const size_t nSamples)
{
  CUDAPostProcessingRMS<<<GetSamplerGridSize(),GetSamplerBlockSize()>>>
                       (samplingBuffer, scalingCoeff, nSamples);

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of PostProcessingRMS
//--------------------------------------------------------------------------------------------------