/**
 * @file        BaseFloatMatrix.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the base class for
 *              single precisions floating point numbers (floats).
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        11 July      2011, 12:13 (created) \n
 *              11 July      2016, 15:15 (revised)
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


#include <string.h>
#include <immintrin.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <MatrixClasses/BaseFloatMatrix.h>

#include <Utils/DimensionSizes.h>
#include <Logger/ErrorMessages.h>


using std::string;

//----------------------------------------------------------------------------//
//                              Constants                                     //
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//                              Implementation                                //
//                              public methods                                //
//----------------------------------------------------------------------------//

/**
 * Copy data from another matrix with same size.
 *
 * @param [in] src - source matrix
 *
 */
void TBaseFloatMatrix::CopyData(const TBaseFloatMatrix & src)
{
  memcpy(pMatrixData,
         src.pMatrixData,
         sizeof(float) * pTotalAllocatedElementCount);
}//end of CopyDataSameSize
//------------------------------------------------------------------------------


/**
 * Zero all allocated elements in parallel. \n
 * Also work as the first touch strategy on NUMA machines.
 *
 */
void TBaseFloatMatrix::ZeroMatrix()
{
  #pragma omp parallel for schedule (static)
  for (size_t i=0; i < pTotalAllocatedElementCount; i++)
  {
    pMatrixData[i] = 0.0f;
  }
}// end of ZeroMatrix
//------------------------------------------------------------------------------

/**
 * Divide a scalar by the elements of matrix.
 * @param [in] scalar - scalar to be divided
 *
 */
void TBaseFloatMatrix::ScalarDividedBy(const float  scalar)
{
  #pragma omp parallel for schedule (static)
  for (size_t i=0; i < pTotalAllocatedElementCount; i++)
  {
    pMatrixData[i] = scalar / pMatrixData[i];
  }
}// end of ScalarDividedBy
//------------------------------------------------------------------------------



/**
 * Copy data from CPU (pmatrixData) to GPU (pdMatrixData).
 * The transfer is synchronous (there is nothing to overlap with in the code)
 */
void TBaseFloatMatrix::CopyToDevice()
{
  checkCudaErrors(cudaMemcpy(pdMatrixData,
                             pMatrixData,
                             pTotalAllocatedElementCount * sizeof(float),
                             cudaMemcpyHostToDevice));
}// end of CopyToDevice
//------------------------------------------------------------------------------

/**
 * Copy data from GPU (pdMatrixData) to CPU (pmatrixData).
 * The transfer is synchronous (there is nothing to overlap with in the code)
 */
void TBaseFloatMatrix::CopyFromDevice()
{
  checkCudaErrors(cudaMemcpy(pMatrixData,
                             pdMatrixData,
                             pTotalAllocatedElementCount*sizeof(float),
                             cudaMemcpyDeviceToHost));
}// end of CopyFromDevice
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                              Implementation                                //
//                             protected methods                              //
//----------------------------------------------------------------------------//

/**
 * Memory allocation based on the total number of elements. \n
 *
 * CPU memory is aligned by the DATA_ALIGNMENT and then registered as pinned and
 * zeroed.
 *
 */
void TBaseFloatMatrix::AllocateMemory()
{
  assert(pMatrixData == NULL);

  //Size of memory to allocate
  size_t SizeInBytes = pTotalAllocatedElementCount * sizeof(float);

  // Allocate CPU memory
  pMatrixData = static_cast<float *> (_mm_malloc(SizeInBytes, DATA_ALIGNMENT));
  if (!pMatrixData)
  {
    throw bad_alloc();
  }

  // Register Host memory (pin in memory)
  checkCudaErrors(cudaHostRegister(pMatrixData,
                                   SizeInBytes,
                                   cudaHostRegisterPortable));

  // Allocate memory on the GPU
  assert(pdMatrixData == NULL);

  checkCudaErrors(cudaMalloc<float>(&pdMatrixData, SizeInBytes));

  if (!pdMatrixData)
  {
    throw bad_alloc();
  }
  // This has to be done for simulations based on input sources
  ZeroMatrix();
}//end of AllocateMemory
//----------------------------------------------------------------------------

/**
 * Free memory.
 * Both on the CPU and GPU side
 */
void TBaseFloatMatrix::FreeMemory()
{
  // Free CPU memory
  if (pMatrixData)
  {
    cudaHostUnregister(pMatrixData);
    _mm_free(pMatrixData);
  }
  pMatrixData = NULL;

  // Free GPU memory
  if (pdMatrixData)
  {
    checkCudaErrors(cudaFree(pdMatrixData));
  }
  pdMatrixData = NULL;
}//end of FreeMemory
//----------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//                            Implementation                                //
//                            private methods                               //
//--------------------------------------------------------------------------//

