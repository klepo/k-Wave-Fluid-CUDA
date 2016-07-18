/**
 * @file        BaseIndexMatrix.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the base class for
 *              64b-wide integers (long for Linux/ size_t for Windows).
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        26 July     2011, 14:17 (created) \n
 *              14 July     2016, 13:36 (revised)
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

#include <MatrixClasses/BaseIndexMatrix.h>
#include <Utils/DimensionSizes.h>
#include <Logger/ErrorMessages.h>


using std::string;


//---------------------------------------------------------------------------//
//                             Constants                                     //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//                             Definitions                                   //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//                             Implementation                                //
//                             public methods                                //
//---------------------------------------------------------------------------//

/**
 *  Zero all allocated elements.
 *
 */
void TBaseIndexMatrix::ZeroMatrix()
{
  ///@todo: This breaks the first touch policy! - however we don't know the distribution
  memset(pMatrixData, 0, pTotalAllocatedElementCount * sizeof(size_t));
}// end of ZeroMatrix
//------------------------------------------------------------------------------

/**
 * Copy data from CPU (Host) to GPU (Device).
 */
void TBaseIndexMatrix::CopyToDevice()
{
  checkCudaErrors(cudaMemcpy(pdMatrixData,
                             pMatrixData,
                             pTotalAllocatedElementCount * sizeof(size_t),
                             cudaMemcpyHostToDevice));

}// end of CopyToDevice
//------------------------------------------------------------------------------

/**
 * Copy data from GPU (Device) to CPU (Host).
 */
void TBaseIndexMatrix::CopyFromDevice()
{
  checkCudaErrors(cudaMemcpy(pMatrixData,
                             pdMatrixData,
                             pTotalAllocatedElementCount * sizeof(size_t),
                             cudaMemcpyDeviceToHost));
}// end of CopyFromDevice
//------------------------------------------------------------------------------

//---------------------------------------------------------------------------//
//                             Implementation                                //
//                            protected methods                              //
//---------------------------------------------------------------------------//

/**
 * Memory allocation based on the total number of elements. \n
 *
 * CPU memory is aligned by the DATA_ALIGNMENT and then registered as pinned and
 * zeroed. The GPU memory is allocated on GPU but not zeroed (no reason)
 *
 */
void TBaseIndexMatrix::AllocateMemory()
{
  /* No memory allocated before this function*/
  assert(pMatrixData == NULL);

  //size of memory to allocate
  size_t SizeInBytes = pTotalAllocatedElementCount * sizeof(size_t);

  pMatrixData = static_cast<size_t*>(_mm_malloc(SizeInBytes, DATA_ALIGNMENT));

  if (!pMatrixData)
  {
    throw bad_alloc();
  }

  // Register Host memory (pin in memory)
  checkCudaErrors(cudaHostRegister(pMatrixData,
                                   SizeInBytes,
                                   cudaHostRegisterPortable));


  assert(pdMatrixData == NULL);

  checkCudaErrors(cudaMalloc<size_t>(&pdMatrixData, SizeInBytes));

  if (!pdMatrixData)
  {
    throw bad_alloc();
  }
}// end of AllocateMemory
//-----------------------------------------------------------------------------

/*
 * Free memory
 */
void TBaseIndexMatrix::FreeMemory()
{
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
}// end of MemoryDeallocation
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
//                             Implementation                                //
//                             private methods                               //
//---------------------------------------------------------------------------//
