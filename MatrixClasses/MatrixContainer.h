/**
 * @file        MatrixContainer.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the matrix container.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        14 September 2012, 14:33 (created) \n
 *              04 November  2014, 17:17 (revised)
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

#ifndef MATRIXCONTAINER_H
#define	MATRIXCONTAINER_H

#include <string.h>
#include <map>

#include "BaseMatrix.h"
#include "BaseFloatMatrix.h"
#include "RealMatrix.h"
#include "ComplexMatrix.h"
#include "LongMatrix.h"
#include "MatrixRecord.h"
#include "../Utils/MatrixNames.h"
#include "../Utils/DimensionSizes.h"

#if VANILLA_CPP_VERSION
#include "../VanillaC++/MatrixClasses/FFTWComplexMatrix.h"
#endif
#if CUDA_VERSION
#include "../CUDA/MatrixClasses/CUFFTComplexMatrix.h"
#endif
#if OPENCL_VERSION
#include "../OpenCL/MatrixClasses/ClFFTComplexMatrix.h"
#endif

/**
 * @enum TMatrixID
 * @brief Matrix identifers of all matrices in the k-space code
 */
enum TMatrixID
{
    kappa, c2, p,

    ux_sgx,uy_sgy, uz_sgz,
    ux_shifted, uy_shifted, uz_shifted,
    duxdx, duydy, duzdz,
    dxudxn    , dyudyn    , dzudzn,
    dxudxn_sgx, dyudyn_sgy, dzudzn_sgz,

    rhox, rhoy, rhoz, rho0,
    dt_rho0_sgx, dt_rho0_sgy, dt_rho0_sgz,

    p0_source_input, sensor_mask_index, sensor_mask_corners,
    ddx_k_shift_pos, ddy_k_shift_pos, ddz_k_shift_pos,
    ddx_k_shift_neg, ddy_k_shift_neg, ddz_k_shift_neg,
    x_shift_neg_r, y_shift_neg_r, z_shift_neg_r,
    pml_x_sgx, pml_y_sgy, pml_z_sgz,
    pml_x    , pml_y    , pml_z,

    absorb_tau, absorb_eta, absorb_nabla1, absorb_nabla2, BonA,

    ux_source_input, uy_source_input, uz_source_input,
    p_source_input,

    u_source_index, p_source_index, transducer_source_input,
    delay_mask,

    //------------------- redundant ----------------//
    Ix_sensor_avg, Iy_sensor_avg, Iz_sensor_avg,
    Ix_sensor_max, Iy_sensor_max, Iz_sensor_max,

    //---------------- output matrices -------------//
    p_sensor_raw,  p_sensor_rms, p_sensor_max, p_sensor_min,
    p_sensor_max_all, p_sensor_min_all,
    ux_sensor_raw, uy_sensor_raw, uz_sensor_raw,

    ux_shifted_sensor_raw, uy_shifted_sensor_raw, uz_shifted_sensor_raw, //non_staggered
    ux_sensor_rms, uy_sensor_rms, uz_sensor_rms,
    ux_sensor_max, uy_sensor_max, uz_sensor_max,
    ux_sensor_min, uy_sensor_min, uz_sensor_min,
    ux_sensor_max_all, uy_sensor_max_all, uz_sensor_max_all,
    ux_sensor_min_all, uy_sensor_min_all, uz_sensor_min_all,

    //--------------Temporary matrices -------------//
    Temp_1_RS3D, Temp_2_RS3D, Temp_3_RS3D,
#if VANILLA_CPP_VERSION & CUDA_VERSION
    FFT_X_temp, FFT_Y_temp, FFT_Z_temp, FFT_shift_temp,
    CUFFT_X_temp, CUFFT_Y_temp, CUFFT_Z_temp, CUFFT_shift_temp
#elif VANILLA_CPP_VERSION
    FFT_X_temp, FFT_Y_temp, FFT_Z_temp, FFT_shift_temp
#elif CUDA_VERSION
    CUFFT_X_temp, CUFFT_Y_temp, CUFFT_Z_temp, CUFFT_shift_temp
#elif OPENCL_VERSION

#endif
};


/**
 * @enum TMatrixDataType
 * @brief All possible types of the matrix
 */

/**
 * @typedef TMatrixRecordContainer
 * @brief map associating the matrix name enum and matric record
 */
typedef map<TMatrixID, TMatrixRecord> TMatrixRecordContainer;

/**
 * @class TMatrixContainer
 * @brief Class implementing the matrix container
 */
class TMatrixContainer {
    public:

        /// Constructor
        TMatrixContainer();
        /// Destructor
        virtual ~TMatrixContainer();

        /**
         * Get number of matrices in the container
         * @return number of matrices in the container
         */
        size_t size()
        {
            return MatrixContainer.size();
        };
        /**
         * Is the container empty?
         * @return true if the container is empty
         */
        bool empty()
        {
            return MatrixContainer.empty();
        };

        /// Create instances of all objects in the container
        void CreateAllObjects();
        /// Load all matrices from the HDF5 file
        void LoadDataFromInputHDF5File(THDF5_File & HDF5_File);
        /// Load all matrices from the HDF5 file
        void LoadDataFromCheckpointHDF5File(THDF5_File & HDF5_File);
        /// Store selected matrices into the checkpoint file
        void StoreDataIntoCheckpointHDF5File(THDF5_File & HDF5_File);
        /// Free all matrices - destroy them
        void FreeAllMatrices();

        /// Set all matrices recored - populate the container
        void AddMatricesIntoContainer();

#if CUDA_VERSION || OPENCL_VERSION
        /// Copy matrix host memory to GPU Device memory
        void CopyAllMatricesToGPU();
        /// Copy GPU Device memory to matrix host memory
        void CopyAllMatricesFromGPU();
#endif
#if COMPARE_CPU_TO_GPU
        void CompareAllMatrices();
#endif
        /**
         * Get matrix record
         * @param [in] MatrixID - Matrix identifier
         * @return the matrix record
         */
        inline TMatrixRecord& GetMatrixRecord(const TMatrixID MatrixID)
        {
            return MatrixContainer[MatrixID];
        };

        /**
         * operator []
         * @param [in]  MatrixID - Matrix identifier
         * @return the matrix record
         */
        inline TMatrixRecord& operator [] (const TMatrixID MatrixID)
        {
            return MatrixContainer[MatrixID];
        };

        /**
         * Get BaseMatrix from the container
         * @param [in] MatrixID - Matrix identifier
         * @return Base Matrix
         */
        inline TBaseMatrix& GetBaseMatrix(const TMatrixID MatrixID)
        {
            return static_cast<TBaseMatrix&>(*(MatrixContainer[MatrixID].MatrixPtr));
        };

        /**
         * Get BaseFloatMatrix from the container
         * @param [in] MatrixID - Matrix identifier
         * @return BaseFloatMatrix
         */
        inline TBaseFloatMatrix& GetBaseFloatMatrix(const TMatrixID MatrixID)
        {
            return static_cast<TBaseFloatMatrix&>(*(MatrixContainer[MatrixID].MatrixPtr));
        };

        /**
         * Get RealMatrix from the container
         * @param [in] MatrixID - Matrix identifier
         * @return RealMatrix
         */
        inline TRealMatrix& GetRealMatrix(const TMatrixID MatrixID)
        {
            return static_cast<TRealMatrix&>(*(MatrixContainer[MatrixID].MatrixPtr));
        };

        /**
         * Get Uxyz_sgzMatrix from the container
         * @param [in] MatrixID - Matrix identifier
         * @return  Uxyz_sgzMatrix
         */
        inline TRealMatrix& GetUxyz_sgxyzMatrix(const TMatrixID MatrixID)
        {
            return static_cast<TRealMatrix&>(*(MatrixContainer[MatrixID].MatrixPtr));
        };

        /**
         * Get ComplexMatrix from the container
         * @param [in] MatrixID - Matrix identifier
         * @return ComplexMatrix
         */
        inline TComplexMatrix& GetComplexMatrix(const TMatrixID MatrixID)
        {
            return static_cast<TComplexMatrix&>(*(MatrixContainer[MatrixID].MatrixPtr));
        };

        /**
         * GetFFTWComplexMatrix from the container
         * @param [in] MatrixID - Matrix identifier
         * @return FFTWComplexMatrix
         */
#if VANILLA_CPP_VERSION
        inline TFFTWComplexMatrix& GetFFTWComplexMatrix(
                const TMatrixID MatrixID){
            return static_cast<TFFTWComplexMatrix &>  (*(MatrixContainer[MatrixID].MatrixPtr));
        };
#endif
#if CUDA_VERSION
        inline TCUFFTComplexMatrix& GetCUFFTComplexMatrix(
                const TMatrixID MatrixID){
            return static_cast<TCUFFTComplexMatrix&>(*(MatrixContainer[MatrixID].MatrixPtr));
        };
#endif
#if OPENCL_VERSION

#endif

        /**
         * Get LongMatrix matrix from the container
         * @param [in] MatrixID - Matrix identifier
         * @return LongMatrix
         */
        inline TLongMatrix& GetLongMatrix (const TMatrixID MatrixID)
        {
            return static_cast<TLongMatrix&>(*(MatrixContainer[MatrixID].MatrixPtr));
        };

        size_t GetSpeculatedMemoryFootprintInMegabytes();

    protected:

    private:

        // map holding the container
        TMatrixRecordContainer MatrixContainer;

        // Copy constructor is not allowed for public
        TMatrixContainer(const TMatrixContainer& orig);

        // Print error and throw an exception
        void PrintErrorAndThrowException(const char* FMT,
                                         const string HDF5MatrixName,
                                         const char* File,
                                         const int Line);

};// end of TMatrixContainer

#endif	/* MATRIXCONTAINER_H */

