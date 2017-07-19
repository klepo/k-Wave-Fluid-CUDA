/**
 * @file        MatrixContainer.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the matrix container and the related
 *              matrix record class.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        02 December  2014, 16:17 (created) \n
 *              07 July      2017, 13:56 (revised)
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

#ifndef MATRIX_CONTAINER_H
#define	MATRIX_CONTAINER_H

#include <map>

#include <MatrixClasses/BaseMatrix.h>
#include <MatrixClasses/BaseFloatMatrix.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CufftComplexMatrix.h>

#include <Utils/MatrixNames.h>
#include <Utils/DimensionSizes.h>

#include <Containers/MatrixRecord.h>


/**
 * @class   TMatrixContainer
 * @brief   Class implementing the matrix container.
 * @details This container is responsible to maintain all the matrices in the code except the output
 *          streams. The matrices are allocated, freed, loaded stored and check-pointed from here.
 */
class TMatrixContainer
{
  public:

    /**
     * @enum TMatrixIdx
     * @brief Matrix identifers of all matrices in the k-space code, names based on the Matlab notation.
     */
    enum class TMatrixIdx
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
      temp_1_real_3D, temp_2_real_3D, temp_3_real_3D,
      cufft_x_temp, cufft_y_temp, cufft_z_temp, cufft_shift_temp
    };// end of TMatrixID
    //----------------------------------------------------------------------------------------------


    /// Constructor.
    TMatrixContainer();
    /// Copy constructor is not allowed.
    TMatrixContainer(const TMatrixContainer&) = delete;
    /// Destructor.
    ~TMatrixContainer();

    /// Operator = is not allowed.
    TMatrixContainer& operator=(const TMatrixContainer&) = delete;

    /**
     * @brief   Get the number of matrices in the container.
     * @details Get the number of matrices in the container.
     * @return  The number of matrices in the container.
     */
    inline size_t Size() const
    {
      return matrixContainer.size();
    };

    /**
     * @brief   Is the container empty?
     * @details Is the container empty?
     * @return  true if the container is empty.
     */
    inline bool IsEmpty() const
    {
      return matrixContainer.empty();
    };

    /**
     * @brief   operator[]
     * @details operator[]
     * @param [in]  matrixIdx - Matrix identifier
     * @return Matrix record
     */
    inline TMatrixRecord& operator[] (const TMatrixIdx matrixIdx)
    {
      return matrixContainer[matrixIdx];
    };

    /**
     * @brief   Get the matrix with a specific type from the container.
     * @details This template routine returns the reference to the matrix re-casted to the specific
     *          class type.
     * @param [in] matrixIdx - Matrix identifier
     * @return     Reference to the Matrix
     */
    template <typename T>
    inline T& GetMatrix(const TMatrixIdx matrixIdx)
    {
      return static_cast<T &> (*(matrixContainer[matrixIdx].matrixPtr));
    };

    /// Create all matrices in the container.
    void CreateMatrices();
    /// Populate the container based on the simulation type.
    void AddMatrices();
    /// Destroy and free all matrices.
    void FreeMatrices();

    /// Load all matrices from the input HDF5 file.
    void LoadDataFromInputFile(THDF5_File& inputFile);
    /// Load all matrices from the output HDF5 file.
    void LoadDataFromCheckpointFile(THDF5_File& checkpointFile);
    /// Store selected matrices into the checkpoint file.
    void StoreDataIntoCheckpointFile(THDF5_File& checkpointFile);

    /// Copy all matrices from host to device (CPU -> GPU).
    void CopyMatricesToDevice();
    /// Copy all matrices from device to host (GPU -> CPU).
    void CopyMatricesFromDevice();

  protected:

  private:

    /// Datatype for the map associating the matrix ID enum and matrix record.
    using TMatrixRecordContainer = std::map<TMatrixIdx, TMatrixRecord>;

    /// map holding the container
    TMatrixRecordContainer matrixContainer;

};// end of TMatrixContainer
//--------------------------------------------------------------------------------------------------
#endif	/* MATRIX_CONTAINER_H */

