/**
 * @file        OutputStreamsCUDAKernels.cuh
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file of cuda kernels used for data sampling (output streams).
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        27 January   2015, 16:25 (created) \n
 *              29 July      2016, 14:19 (revised)
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


#ifndef OUTPUT_STREAMS_CUDA_KERNELS_CUH
#define	OUTPUT_STREAMS_CUDA_KERNELS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>

/**
 * @namespace   OutputStreamsCUDAKernels
 * @brief       List of cuda kernels used for sampling data.
 * @details     List of cuda kernels used for sampling data.
 *
 */
namespace OutputStreamsCUDAKernels
{
  /// Kernel to sample quantities using an index sensor mask
  template<TBaseOutputHDF5Stream::TReduceOperator reduceOp>
  void SampleIndex(float*        samplingBuffer,
                   const float*  sourceData,
                   const size_t* sensorData,
                   const size_t  nSamples);

  /// Kernel to sample quantities inside one cuboid
  template<TBaseOutputHDF5Stream::TReduceOperator reduceOp>
  void SampleCuboid(float*       samplingBuffer,
                    const float* sourceData,
                    const dim3   topLeftCorner,
                    const dim3   bottomRightCorner,
                    const dim3   matrixSize,
                    const size_t nSamples);

  /// Kernel to sample of the quantity on the whole domain
  template<TBaseOutputHDF5Stream::TReduceOperator reduceOp>
  void SampleAll(float*       samplingBuffer,
                 const float* sourceData,
                 const size_t nSamples);

  /// Kernel to calculate post-processing for RMS
  void PostProcessingRMS(float*       samplingBuffer,
                         const float  scalingCoeff,
                         const size_t nSamples);
}// end of OutputStreamsCUDAKernels
//--------------------------------------------------------------------------------------------------

#endif	/* OUTPUT_STREAMS_CUDA_KERNELS_H */

