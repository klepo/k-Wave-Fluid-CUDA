/**
 * @file        OutputStreamsCUDAKernles.cuh
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 *
 * @brief       The header file a list of cuda kernels used for data sampling
 *              (output streams)
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        27 January   2015, 16:25 (created) \n
 *              22 March     2016, 17:07 (revised)
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
 *O
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with k-Wave. If not, see http://www.gnu.org/licenses/.
 */


#ifndef OUTPUT_STREAMS_CUDA_KERNELS_CUH
#define	OUTPUT_STREAMS_CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>

/**
 * @namespace   OutputStreamsCUDAKernels
 * @brief       List of cuda kernels used for sampling data
 * @details     List of cuda kernels used for sampling data
 *
 */
namespace OutputStreamsCUDAKernels
{
  /// Kernel to sample quantities using an index sensor mask
  template<TBaseOutputHDF5Stream::TReductionOperator ReductionOp>
  void SampleIndex(float*        SamplingBuffer,
                   const float*  SourceData,
                   const size_t* SensorData,
                   const size_t  NumberOfSamples);



  /// Kernel to sample quantities inside one cuboid
  template<TBaseOutputHDF5Stream::TReductionOperator ReductionOp>
  void SampleCuboid(float*       SamplingBuffer,
                    const float* SourceData,
                    const dim3   TopLeftCorner,
                    const dim3   BottomRightCorner,
                    const dim3   DimensionSizes,
                    const size_t NumberOfSamples);


  /// Kernel to sample of the quantity on the whole domain
  template<TBaseOutputHDF5Stream::TReductionOperator ReductionOp>
  void SampleAll(float*       SamplingBuffer,
                 const float* SourceData,
                 const size_t NumberOfSamples);


  /// Kernel to calculate post-processing for RMS
  void PostProcessingRMS(float*       SamplingBuffer,
                         const float  ScalingCoeff,
                         const size_t NumberOfSamples);

}

#endif	/* OUTPUTSTREAMSCUDAKERNLES_H */

