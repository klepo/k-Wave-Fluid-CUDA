/**
 * @file        BaseLongMatrix.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the base class for
 *              64b-wide integers (long for Linux/ size_t for Windows).
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        26 July     2011, 14:17 (created) \n
 *              04 November 2014, 17:11 (revised)
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
#include <assert.h>

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#if VANILLA_CPP_VERSION

#endif
#if CUDA_VERSION
#include <cuda_runtime.h>
#include <iostream> //fprintf
#include <stdlib.h> //exit

using std::string;

inline void gpuAssert(cudaError_t code,
                      string file,
                      int line,
                      bool Abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,
                "GPUassert: %s %s %d\n",
                cudaGetErrorString(code),file.c_str(),line);
        if (Abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif
#if OPENCL_VERSION

#include "../OpenCL/ClSupremo.h"

#endif

#include "../MatrixClasses/BaseLongMatrix.h"
#include "../Utils/DimensionSizes.h"
#include "../Utils/ErrorMessages.h"
#include "../Utils/MemoryMeasure.h"

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

void TBaseLongMatrix::CopyData(TBaseLongMatrix & src)
{
    memcpy(pMatrixData,
           src.pMatrixData,
           pTotalAllocatedElementCount*sizeof(size_t));
}

#if CUDA_VERSION
//Host   -> Device
void TBaseLongMatrix::CopyIn(const size_t * HostSource)
{
    gpuErrchk(cudaMemcpy(pdMatrixData,
                         HostSource,
                         pTotalAllocatedElementCount*sizeof(size_t),
                         cudaMemcpyHostToDevice));
}

//Device -> Host
void TBaseLongMatrix::CopyOut(size_t* HostDestination)
{
    gpuErrchk(cudaMemcpy(HostDestination,
                         pdMatrixData,
                         pTotalAllocatedElementCount*sizeof(size_t),
                         cudaMemcpyDeviceToHost));
}

//Device -> Device
void TBaseLongMatrix::CopyForm(const size_t* DeviceSource)
{
    gpuErrchk(cudaMemcpy(pdMatrixData,
                         DeviceSource,
                         pTotalAllocatedElementCount*sizeof(size_t),
                         cudaMemcpyDeviceToDevice));
}
#endif
#if OPENCL_VERSION
//Host   -> Device
void TBaseLongMatrix::CopyIn(const size_t* HostSource)
{
    ClSupremo* handle = ClSupremo::GetInstance();
    handle->PassLongMemoryToClMemory(pdMatrixData,
                                     const_cast<size_t*>(HostSource),
                                     pTotalAllocatedElementCount);
}

//Device -> Host
void TBaseLongMatrix::CopyOut(size_t* HostDestination)
{
    ClSupremo* handle = ClSupremo::GetInstance();
    handle->PassClMemoryToLongMemory(pdMatrixData,
                                     HostDestination,
                                     pTotalAllocatedElementCount);
}

//Device -> Device
void TBaseLongMatrix::CopyForm(const cl_mem DeviceSource)
{
    ClSupremo* handle = ClSupremo::GetInstance();
    pdMatrixData = handle->CopyLongClMemory(DeviceSource,
                                            pTotalAllocatedElementCount);
}
#endif

//---------------------------------------------------------------------------//
//                             Implementation                                //
//                            protected methods                              //
//---------------------------------------------------------------------------//

/**
 * Memory allocation based on the total number of elements. \n
 */
void TBaseLongMatrix::AllocateMemory()
{

    /* No memory allocated before this function*/
    assert(pMatrixData == NULL);

    //if testing, print the current memory size
    LogPreAlloc();

    //size of memory to allocate
    size_t size_in_bytes = pTotalAllocatedElementCount*sizeof(size_t);

    //if testing, print how much memory is requested
    LogAlloc(size_in_bytes);

    pMatrixData = static_cast<size_t*>(malloc(size_in_bytes));

    if (!pMatrixData) {
        fprintf(stderr,Matrix_ERR_FMT_Not_Enough_Memory, "TBaseLongMatrix");
        throw bad_alloc();
    }

#if CUDA_VERSION
    assert(pdMatrixData == NULL);

    gpuErrchk(cudaMalloc((void **)&pdMatrixData,
                         size_in_bytes));

    if (!pdMatrixData) {
        fprintf(stderr,Matrix_ERR_FMT_Not_Enough_Memory, "TBaseLongMatrix");
        throw bad_alloc();
    }
#endif

#if OPENCL_VERSION
    assert(pdMatrixData == NULL);

    pdMatrixData =
        ClSupremo::GetInstance()->CreateLongClMemory(pTotalAllocatedElementCount);

    if (!pdMatrixData) {
        fprintf(stderr,Matrix_ERR_FMT_Not_Enough_Memory, "TBaseLongMatrix");
        throw bad_alloc();
    }
#endif

    //if testing, print the current memory size
    LogPostAlloc();

}// end of AllocateMemory
//-----------------------------------------------------------------------------

/*
 * Free memory
 */
void TBaseLongMatrix::FreeMemory()
{
    if (pMatrixData) free(pMatrixData);
    pMatrixData = NULL;

#if CUDA_VERSION
    if (pdMatrixData) gpuErrchk(cudaFree(pdMatrixData));
    pdMatrixData = NULL;
#endif
#if OPENCL_VERSION
    if (pdMatrixData) clReleaseMemObject(pdMatrixData);
    pdMatrixData = NULL;
#endif

}// end of MemoryDeallocation
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
//                             Implementation                                //
//                             private methods                               //
//---------------------------------------------------------------------------//
