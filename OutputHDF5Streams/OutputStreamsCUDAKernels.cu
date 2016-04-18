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
 *              12 April     2016, 15:03 (revised)
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
#include <cuda.h>
#include <cuda_runtime.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>
#include <OutputHDF5Streams/OutputStreamsCUDAKernels.cuh>

#include <Parameters/Parameters.h>
#include <Utils/ErrorMessages.h>
#include <Utils/CUDAUtils.cuh>

using namespace std;

//----------------------------------------------------------------------------//
//                                Constants                                   //
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//----------------------------- Global routines ------------------------------//
//----------------------------------------------------------------------------//

/**
 * Get Sampler CUDA Block size
 * @return CUDA block size
 */
int GetSamplerBlockSize()
{
  return TParameters::GetInstance()->CUDAParameters.GetSamplerBlockSize1D();
}// end of GetSamplerBlockSize
//------------------------------------------------------------------------------


/**
 * Get sampler CUDA grid size
 * @return CUDA grid size
 */
int GetSamplerGridSize()
{
  return TParameters::GetInstance()->CUDAParameters.GetSamplerGridSize1D();
}// end of GetSamplerGridSize
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//--------------------------- Index based sampling ---------------------------//
//----------------------------------------------------------------------------//


/**
 * CUDA kernel to sample data based on index sensor mask. The operator is given by
 * the template parameter
 * @param [out] SamplingBuffer  - buffer to sample data in
 * @param [in] SourceData       - source matrix
 * @param [in] SensorData       - sensor mask
 * @param [in] NumberOfSamples  - number of sampled points
 */
template <TBaseOutputHDF5Stream::TReductionOperator ReductionOp>
__global__ void CUDASampleIndex(float*        SamplingBuffer,
                                const float*  SourceData,
                                const size_t* SensorData,
                                const size_t  NumberOfSamples)
{
  for (auto i = GetIndex(); i < NumberOfSamples; i += GetStride())
  {
    switch (ReductionOp)
    {
      case TBaseOutputHDF5Stream::TReductionOperator::roNONE:
      {
        SamplingBuffer[i] = SourceData[SensorData[i]];
        break;
      }
      case TBaseOutputHDF5Stream::TReductionOperator::roRMS:
      {
        SamplingBuffer[i] += (SourceData[SensorData[i]] * SourceData[SensorData[i]]);
        break;
      }
      case TBaseOutputHDF5Stream::TReductionOperator::roMAX:
      {
        SamplingBuffer[i] = max(SamplingBuffer[i], SourceData[SensorData[i]]);
        break;
      }
      case TBaseOutputHDF5Stream::TReductionOperator::roMIN:
      {
        SamplingBuffer[i] = min(SamplingBuffer[i], SourceData[SensorData[i]]);
        break;
      }
    }// switch
  } // for
}// end of CUDASampleRawIndex
//------------------------------------------------------------------------------

/**
 * Sample the source matrix using the index sensor mask and store data in buffer
 * @param [out] SamplingBuffer   - buffer to sample data in
 * @param [in] SourceData        - source matrix
 * @param [in] SensorData        - sensor mask
 * @param [in] NumberOfSamples   - number of sampled points
 * @param [in] ReductionOperator - number of sampled points
 */
template<TBaseOutputHDF5Stream::TReductionOperator ReductionOp>
void OutputStreamsCUDAKernels::SampleIndex(float*        SamplingBuffer,
                                           const float * SourceData,
                                           const size_t* SensorData,
                                           const size_t  NumberOfSamples)
{
  CUDASampleIndex<ReductionOp>
                 <<<GetSamplerGridSize(),GetSamplerBlockSize()>>>
                 (SamplingBuffer,
                  SourceData,
                  SensorData,
                  NumberOfSamples);

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of SampleRawIndex
//------------------------------------------------------------------------------

//------------------------ Explicit instances of SampleIndex -----------------//
template
void OutputStreamsCUDAKernels::SampleIndex<TBaseOutputHDF5Stream::TReductionOperator::roNONE>
                                          (float*        SamplingBuffer,
                                           const float*  SourceData,
                                           const size_t* SensorData,
                                           const size_t  NumberOfSamples);

template
void OutputStreamsCUDAKernels::SampleIndex<TBaseOutputHDF5Stream::TReductionOperator::roRMS>
                                          (float*        SamplingBuffer,
                                           const float*  SourceData,
                                           const size_t* SensorData,
                                           const size_t  NumberOfSamples);

template
void OutputStreamsCUDAKernels::SampleIndex<TBaseOutputHDF5Stream::TReductionOperator::roMAX>
                                          (float*        SamplingBuffer,
                                           const float*  SourceData,
                                           const size_t* SensorData,
                                           const size_t  NumberOfSamples);

template
void OutputStreamsCUDAKernels::SampleIndex<TBaseOutputHDF5Stream::TReductionOperator::roMIN>
                                          (float*        SamplingBuffer,
                                           const float*  SourceData,
                                           const size_t* SensorData,
                                           const size_t  NumberOfSamples);


//----------------------------------------------------------------------------//
//-------------------------- Cuboid based sampling ---------------------------//
//----------------------------------------------------------------------------//

/**
 * Get 3D coordinates in the cuboid
 * @param [in] Index             - 1D index
 * @param [in] TopLeftCorner     - Top left corner
 * @param [in] BottomRightCorner - Bottom right corner
 * @param [in] DimensionSizes    - DimensionSizes of the matrix being sampled
 * @return 3D coordinates within the cuboid
 */
inline __device__ size_t TransformCoordinates(const size_t Index,
                                              const dim3 & TopLeftCorner,
                                              const dim3 & BottomRightCorner,
                                              const dim3 & DimensionSizes)
{
  dim3 LocalPosition;
  // calculate the cuboid size
  dim3 CuboidSize(BottomRightCorner.x - TopLeftCorner.x + 1,
                  BottomRightCorner.y - TopLeftCorner.y + 1,
                  BottomRightCorner.z - TopLeftCorner.z + 1);

  // find coordinates within the cuboid
  size_t XY_Size = CuboidSize.x * CuboidSize.y;
  LocalPosition.z =  Index / XY_Size;
  LocalPosition.y = (Index % XY_Size) / CuboidSize.x;
  LocalPosition.x = (Index % XY_Size) % CuboidSize.x;

  // transform the coordinates to the global dimensions
  dim3 GlobalPosition(LocalPosition);
  GlobalPosition.z += TopLeftCorner.z;
  GlobalPosition.y += TopLeftCorner.y;
  GlobalPosition.x += TopLeftCorner.x;

  // calculate 1D index
  return (GlobalPosition.z * DimensionSizes.x * DimensionSizes.y +
          GlobalPosition.y * DimensionSizes.x +
          GlobalPosition.x);
}// end of TransformCoordinates
//------------------------------------------------------------------------------




/**
 * CUDA kernel to sample data inside one cuboid, operation is selected by
 * a template parameter.
 * @param [out] SamplingBuffer    - buffer to sample data in
 * @param [in]  SourceData        - source matrix
 * @param [in]  TopLeftCorner     - top left corner of the cuboid
 * @param [in]  BottomRightCorner - bottom right corner of the cuboid
 * @param [in]  DimensionSizes    - dimension sizes of the matrix being sampled
 * @param [in]  NumberOfSamples   - number of grid points inside the cuboid
 */
template <TBaseOutputHDF5Stream::TReductionOperator ReductionOp>
__global__ void CUDASampleCuboid(float*       SamplingBuffer,
                                 const float* SourceData,
                                 const dim3   TopLeftCorner,
                                 const dim3   BottomRightCorner,
                                 const dim3   DimensionSizes,
                                 const size_t NumberOfSamples)
{
  // iterate over all grid points
  for (auto i = GetIndex(); i < NumberOfSamples; i += GetStride())
  {
    auto Position = TransformCoordinates(i,
                                         TopLeftCorner,
                                         BottomRightCorner,
                                         DimensionSizes);
    switch (ReductionOp)
    {
      case TBaseOutputHDF5Stream::TReductionOperator::roNONE:
      {
        SamplingBuffer[i] = SourceData[Position];
        break;
      }
      case TBaseOutputHDF5Stream::TReductionOperator::roRMS:
      {
        SamplingBuffer[i] += (SourceData[Position] * SourceData[Position]);
        break;
      }
      case TBaseOutputHDF5Stream::TReductionOperator::roMAX:
      {
        SamplingBuffer[i] = max(SamplingBuffer[i], SourceData[Position]);
        break;
      }
      case TBaseOutputHDF5Stream::TReductionOperator::roMIN:
      {
        SamplingBuffer[i] = min(SamplingBuffer[i], SourceData[Position]);
        break;
      }
    }// switch
  } // for
}// end of CUDASampleRawCuboid
//------------------------------------------------------------------------------


/**
 * CUDA kernel to sample data inside one cuboid and store it to buffer. The operation
 * is given in the template paramerter
 *
 * @param [out] SamplingBuffer    - buffer to sample data in
 * @param [in]  SourceData        - source matrix
 * @param [in]  TopLeftCorner     - top left corner of the cuboid
 * @param [in]  BottomRightCorner - bottom right corner of the cuboid
 * @param [in]  NumberOfSamples   - number of grid points inside the cuboid
 */
template<TBaseOutputHDF5Stream::TReductionOperator ReductionOp>
void OutputStreamsCUDAKernels::SampleCuboid(float*       SamplingBuffer,
                                            const float* SourceData,
                                            const dim3   TopLeftCorner,
                                            const dim3   BottomRightCorner,
                                            const dim3   DimensionSizes,
                                            const size_t NumberOfSamples)
{
  CUDASampleCuboid<ReductionOp>
                  <<<GetSamplerGridSize(),GetSamplerBlockSize()>>>
                  (SamplingBuffer,
                   SourceData,
                   TopLeftCorner,
                   BottomRightCorner,
                   DimensionSizes,
                   NumberOfSamples);
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of SampleRawCuboid
//------------------------------------------------------------------------------


//------------------------ Explicit instances of SampleCuboid ----------------//
template
void OutputStreamsCUDAKernels::SampleCuboid<TBaseOutputHDF5Stream::TReductionOperator::roNONE>
                                           (float  * SamplingBuffer,
                                            const float  * SourceData,
                                            const dim3     TopLeftCorner,
                                            const dim3     BottomRightCorner,
                                            const dim3     DimensionSizes,
                                            const size_t   NumberOfSamples);

template
void OutputStreamsCUDAKernels::SampleCuboid<TBaseOutputHDF5Stream::TReductionOperator::roRMS>
                                           (float  * SamplingBuffer,
                                            const float  * SourceData,
                                            const dim3     TopLeftCorner,
                                            const dim3     BottomRightCorner,
                                            const dim3     DimensionSizes,
                                            const size_t   NumberOfSamples);
template
void OutputStreamsCUDAKernels::SampleCuboid<TBaseOutputHDF5Stream::TReductionOperator::roMAX>
                                           (float  * SamplingBuffer,
                                            const float  * SourceData,
                                            const dim3     TopLeftCorner,
                                            const dim3     BottomRightCorner,
                                            const dim3     DimensionSizes,
                                            const size_t   NumberOfSamples);
template
void OutputStreamsCUDAKernels::SampleCuboid<TBaseOutputHDF5Stream::TReductionOperator::roMIN>
                                           (float  * SamplingBuffer,
                                            const float  * SourceData,
                                            const dim3     TopLeftCorner,
                                            const dim3     BottomRightCorner,
                                            const dim3     DimensionSizes,
                                            const size_t   NumberOfSamples);

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
template <TBaseOutputHDF5Stream::TReductionOperator ReductionOp>
__global__ void CUDASampleAll(      float  * SamplingBuffer,
                                 const float  * SourceData,
                                 const size_t   NumberOfSamples)
{
  for (size_t i = GetIndex(); i < NumberOfSamples; i += GetStride())
  {
    switch (ReductionOp)
    {
      case TBaseOutputHDF5Stream::TReductionOperator::roRMS:
      {
        SamplingBuffer[i] += (SourceData[i] * SourceData[i]);
        break;
      }
      case TBaseOutputHDF5Stream::TReductionOperator::roMAX:
      {
        SamplingBuffer[i] = max(SamplingBuffer[i], SourceData[i]);
        break;
      }
      case TBaseOutputHDF5Stream::TReductionOperator::roMIN:
      {
        SamplingBuffer[i] = min(SamplingBuffer[i], SourceData[i]);
        break;
      }
    }
  }
}// end of CUDASampleMaxAll
//------------------------------------------------------------------------------


/**
 * Sample and the whole domain and apply a defined operator
 *
 * @param [in,out] SamplingBuffer - buffer to sample data in
 * @param [in] SourceData         - source matrix
 * @param [in] NumberOfSamples    - number of sampled points
 */
template<TBaseOutputHDF5Stream::TReductionOperator ReductionOp>
void OutputStreamsCUDAKernels::SampleAll(float*       SamplingBuffer,
                                         const float* SourceData,
                                         const size_t NumberOfSamples)
{
  CUDASampleAll<ReductionOp>
               <<<GetSamplerGridSize(),GetSamplerBlockSize()>>>
               (SamplingBuffer,
                SourceData,
                NumberOfSamples);
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of SampleMaxAll
//------------------------------------------------------------------------------


//------------------------ Explicit instances of SampleAll -------------------//
template
void OutputStreamsCUDAKernels::SampleAll<TBaseOutputHDF5Stream::TReductionOperator::roRMS>
                                        (float*       SamplingBuffer,
                                         const float* SourceData,
                                         const size_t NumberOfSamples);

template
void OutputStreamsCUDAKernels::SampleAll<TBaseOutputHDF5Stream::TReductionOperator::roMAX>
                                        (float*       SamplingBuffer,
                                         const float* SourceData,
                                         const size_t NumberOfSamples);

template
void OutputStreamsCUDAKernels::SampleAll<TBaseOutputHDF5Stream::TReductionOperator::roMIN>
                                        (float*       SamplingBuffer,
                                         const float* SourceData,
                                         const size_t NumberOfSamples);


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
  for (size_t i = GetIndex(); i < NumberOfSamples; i += GetStride())
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
void OutputStreamsCUDAKernels::PostProcessingRMS(float*       SamplingBuffer,
                                                 const float  ScalingCoeff,
                                                 const size_t NumberOfSamples)
{
  CUDAPostProcessingRMS<<<GetSamplerGridSize(),GetSamplerBlockSize()>>>
                       (SamplingBuffer,
                        ScalingCoeff,
                        NumberOfSamples);

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of PostProcessingRMS
//------------------------------------------------------------------------------