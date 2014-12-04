/**
 * @file        OutputHDF5Stream.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file of the class saving RealMatrix data
 *              into the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        11 July      2012, 10:30 (created) \n
 *              04 November  2014, 17:01 (revised)
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

#include "./BaseOutputHDF5Stream.h"
#include "../../Utils/ErrorMessages.h"
#include "../../Parameters/Parameters.h"

#include <cmath>
#include <limits>

using namespace std;

//--------------------------------------------------------------------------//
//                              Constants                                   //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                              Definitions                                 //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                              Implementation                              //
//                              public methods                              //
//--------------------------------------------------------------------------//

/**
 * Apply post-processing on the buffer. It supposes the elements are independent
 *
 */
void TBaseOutputHDF5Stream::PostProcess()
{
    switch (ReductionOp) {
        case roNONE:
        {
            // do nothing
            break;
        }

        case roRMS:
        {
            const float ScalingCoeff = 1.0f / (TParameters::GetInstance()->Get_Nt()
                    - TParameters::GetInstance()->GetStartTimeIndex());

#pragma omp parallel for if (BufferSize > MinGridpointsToSampleInParallel)
            for (size_t i = 0; i < BufferSize; i++) {
                StoreBuffer[i] = sqrt(StoreBuffer[i] * ScalingCoeff);
            }
            break;
        }

        case roMAX:
        {
            // do nothing
            break;
        }

        case roMIN:
        {
            // do nothing
            break;
        }
}// switch

}// end of ApplyPostProcessing
//------------------------------------------------------------------------------


/**
 * Allocate memory using a proper memory alignment.
 * @warning - This can routine is not used in the base class (should be used in
 *            derived ones
 */
void TBaseOutputHDF5Stream::AllocateMemory()
{
    StoreBuffer = new float [BufferSize];

    if (!StoreBuffer) {
        fprintf(stderr, Matrix_ERR_FMT_Not_Enough_Memory, "TBaseOutputHDF5Stream");
        throw bad_alloc();
    }

    // we need different initialization for different reduction ops
    switch (ReductionOp) {
        case roNONE :
            {
                // zero the matrix
#pragma omp parallel for if (BufferSize > MinGridpointsToSampleInParallel)
                for (size_t i = 0; i < BufferSize; i++) {
                    StoreBuffer[i] = 0.0f;
                }
                break;
            }

        case roRMS  :
            {
                // zero the matrix
#pragma omp parallel for if (BufferSize > MinGridpointsToSampleInParallel)
                for (size_t i = 0; i < BufferSize; i++) {
                    StoreBuffer[i] = 0.0f;
                }
                break;
            }

        case roMAX  :
            {
                // set the values to the highest negative float value
#pragma omp parallel for if (BufferSize > MinGridpointsToSampleInParallel)
                for (size_t i = 0; i < BufferSize; i++) {
                    StoreBuffer[i] = -1 * std::numeric_limits<float>::max();
                }
                break;
            }

        case roMIN  :
            {
                // set the values to the highest float value
#pragma omp parallel for if (BufferSize > MinGridpointsToSampleInParallel)
                for (size_t i = 0; i < BufferSize; i++)
                {
                    StoreBuffer[i] = std::numeric_limits<float>::max();
                }
                break;
            }
    }// switch

}// end of AllocateMemory
//------------------------------------------------------------------------------

/**
 * Free memory.
 * @warning - This can routine is not used in the base class (should be used in
 *            derived ones
 */
void TBaseOutputHDF5Stream::FreeMemory()
{
    if (StoreBuffer) {
        delete[] StoreBuffer;
        StoreBuffer = NULL;
  }
}// end of FreeMemory
//------------------------------------------------------------------------------


//--------------------------------------------------------------------------//
//                              Implementation                              //
//                             protected methods                            //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                              Implementation                              //
//                              private methods                             //
//--------------------------------------------------------------------------//

