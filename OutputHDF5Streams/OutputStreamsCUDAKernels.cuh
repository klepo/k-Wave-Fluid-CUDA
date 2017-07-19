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
 *              19 July      2017, 15:22 (revised)
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


#ifndef OutputStreamsCudaKernelsH
#define OutputStreamsCudaKernelsH

#include <cuda.h>
#include <cuda_runtime.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>

/**
 * @namespace   OutputStreamsCudaKernels
 * @brief       List of cuda kernels used for sampling data.
 * @details     List of cuda kernels used for sampling data.
 *
 */
namespace OutputStreamsCudaKernels
{
  /**
   * @brief  Sample the source matrix using the index sensor mask and store data in buffer.
   *
   * @tparam      reduceOp         - Reduction operator
   * @param [out] samplingBuffer   - Buffer to sample data in
   * @param [in]  sourceData       - Source matrix
   * @param [in]  sensorData       - Sensor mask
   * @param [in]  nSamples         - Number of sampled points
   */
  template<BaseOutputStream::ReduceOperator reduceOp>
  void sampleIndex(float*        samplingBuffer,
                   const float*  sourceData,
                   const size_t* sensorData,
                   const size_t  nSamples);

  /**
   * @brief  Sample data inside one cuboid and store it to buffer. The operation is given in the template parameter.
   *
   * @tparam      reduceOp         - Reduction operator
   * @param [out] samplingBuffer    - Buffer to sample data in
   * @param [in]  sourceData        - Source matrix
   * @param [in]  topLeftCorner     - Top left corner of the cuboid
   * @param [in]  bottomRightCorner - Bottom right corner of the cuboid
   * @param [in]  matrixSize        - Size of the matrix being sampled
   * @param [in]  nSamples          - Number of grid points inside the cuboid
   */
  template<BaseOutputStream::ReduceOperator reduceOp>
  void sampleCuboid(float*       samplingBuffer,
                    const float* sourceData,
                    const dim3   topLeftCorner,
                    const dim3   bottomRightCorner,
                    const dim3   matrixSize,
                    const size_t nSamples);

  /**
   * @brief Sample and the whole domain and apply a defined operator.
   *
   * @tparam      reduceOp         - Reduction operator
   * @param [in,out] samplingBuffer - Buffer to sample data in
   * @param [in]     sourceData     - Source matrix
   * @param [in]     nSamples       - Number of sampled points
   */
  template<BaseOutputStream::ReduceOperator reduceOp>
  void sampleAll(float*       samplingBuffer,
                 const float* sourceData,
                 const size_t nSamples);

  /**
   * @brief Calculate post-processing for RMS.
   *
   * @param [in, out] samplingBuffer - Buffer to apply post-processing on
   * @param [in]      scalingCoeff   - Scaling coefficent
   * @param [in]      nSamples       - Number of elements
   */
  void postProcessingRms(float*       samplingBuffer,
                         const float  scalingCoeff,
                         const size_t nSamples);
}// end of OutputStreamsCudaKernels
//----------------------------------------------------------------------------------------------------------------------

#endif	/* OutputStreamsCudaKernelsH */

