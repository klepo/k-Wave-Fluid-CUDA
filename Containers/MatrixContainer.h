/**
 * @file        MatrixContainer.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the matrix and container and related
 *              matrix record.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        02 December  2014, 16:17 (created) \n
 *              02 December  2014, 16:17 (revised)
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

#ifndef MATRIX_CONTAINER_H
#define	MATRIX_CONTAINER_H

#include <string.h>
#include <map>

#include <MatrixClasses/BaseMatrix.h>
#include <MatrixClasses/BaseFloatMatrix.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CUFFTComplexMatrix.h>

#include <Utils/MatrixNames.h>
#include <Utils/DimensionSizes.h>

#include <Containers/MatrixRecord.h>


/**
 * @enum TMatrixID
 * @brief Matrix identifers of all matrices in the k-space code
 */
enum TMatrixID
{
  kappa, c2, p,

  ux_sgx    , uy_sgy    , uz_sgz,
  ux_shifted, uy_shifted, uz_shifted,
  duxdx     , duydy     , duzdz,
  dxudxn    , dyudyn    , dzudzn,
  dxudxn_sgx, dyudyn_sgy, dzudzn_sgz,

  rhox, rhoy , rhoz, rho0,
  dt_rho0_sgx, dt_rho0_sgy, dt_rho0_sgz,

  p0_source_input, sensor_mask_index, sensor_mask_corners,
  ddx_k_shift_pos, ddy_k_shift_pos, ddz_k_shift_pos,
  ddx_k_shift_neg, ddy_k_shift_neg, ddz_k_shift_neg,
  x_shift_neg_r  , y_shift_neg_r  , z_shift_neg_r,
  pml_x_sgx      , pml_y_sgy      , pml_z_sgz,
  pml_x          , pml_y          , pml_z,

  absorb_tau, absorb_eta, absorb_nabla1, absorb_nabla2, BonA,

  ux_source_input, uy_source_input, uz_source_input,
  p_source_input,

  u_source_index, p_source_index, transducer_source_input,
  delay_mask,

  //--------------Temporary matrices -------------//
  Temp_1_RS3D , Temp_2_RS3D , Temp_3_RS3D,
  CUFFT_X_temp, CUFFT_Y_temp, CUFFT_Z_temp, CUFFT_shift_temp
};// end of TMatrixID
//------------------------------------------------------------------------------


/**
 * @class TMatrixContainer
 * @brief Class implementing the matrix container.
 * @details This container is responsible to maintain all the matrices in the
 *          code except the output streams. The matrices are allocated, freed, loaded
 *          stored and checkpointed from here.
 */
class TMatrixContainer
{
  public:

    /// Constructor
    TMatrixContainer() {};
    /// Destructor - no need for virtual destructor (no polymorphism).
    ~TMatrixContainer();

    /**
     * @brief Get number of matrices in the container.
     * @details Get number of matrices in the container.
     * @return number of matrices in the container.
     */
    size_t size()
    {
      return MatrixContainer.size();
    };

    /**
     * @brief Is the container empty?
     * @details Is the container empty?
     * @return true if the container is empty.
     */
    bool empty() const
    {
      return MatrixContainer.empty();
    };

    /// Create instances of all objects in the container.
    void CreateAllObjects();
    /// Set all matrices recored - populate the container.
    void AddMatricesIntoContainer();
    /// Free all matrices - destroy them.
    void FreeAllMatrices();

    /// Load all matrices from the HDF5 file.
    void LoadDataFromInputHDF5File(THDF5_File & HDF5_File);
    /// Load all matrices from the HDF5 file.
    void LoadDataFromCheckpointHDF5File(THDF5_File & HDF5_File);
    /// Store selected matrices into the checkpoint file.
    void StoreDataIntoCheckpointHDF5File(THDF5_File & HDF5_File);

    /// Copy host (CPU) matrices to GPU Device memory.
    void CopyAllMatricesToGPU();
    /// Copy GPU Device memory matrices to host (CPU) memory.
    void CopyAllMatricesFromGPU();

    /**
     * Get matrix record
     * @param [in] MatrixID - Matrix identifier
     * @return the matrix record.
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
     * @brief Get the matrix with a specific type from the container.
     * @details This template routine returns the reference to the matrix recasted to
     * the specific class.
     * @param [in] MatrixID - Matrix identifier
     * @return Base Matrix
     */
    template <typename T>
    inline T& GetMatrix(const TMatrixID MatrixID)
    {
      return static_cast<T &> (*(MatrixContainer[MatrixID].MatrixPtr));
    };

    /// Try to guess how much memory is necessary to run the simulation.
    size_t GetSpeculatedMemoryFootprintInMegabytes();

  protected:

  private:

    /// Datatype for map associating the matrix ID enum and matrix record.
    typedef map<TMatrixID, TMatrixRecord> TMatrixRecordContainer;

    // map holding the container
    TMatrixRecordContainer MatrixContainer;

    // Copy constructor is not allowed for public
    TMatrixContainer(const TMatrixContainer& orig);

    /// Operator = is not allowed for public.
    TMatrixContainer & operator = (const TMatrixContainer& src);

    /// Print error and throw an exception.
    void PrintErrorAndThrowException(const char* FMT,
                                     const string HDF5MatrixName,
                                     const char* File,
                                     const int Line);

};// end of TMatrixContainer
//------------------------------------------------------------------------------
#endif	/* MATRIX_CONTAINER_H */

