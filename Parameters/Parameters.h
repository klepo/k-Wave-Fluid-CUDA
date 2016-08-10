/**
 * @file        Parameters.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the parameters of the simulation.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        08 December 2011, 16:34 (created)      \n
 *              25 July     2016, 15:08 (revised)
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

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <string>

#include <Parameters/CUDAParameters.h>
#include <Parameters/CommandLineParameters.h>
#include <Utils/DimensionSizes.h>
#include <HDF5/HDF5_File.h>

/**
 * @class TParameters
 * @brief Class storing all parameters of the simulation.
 * @brief Class storing all parameters of the simulation.
 * @warning This is a singleton class.
 *
 */
class TParameters
{
  public:

    /**
     * @enum TSensorMaskType
     * @brief   Sensor mask type (linear indices or cuboid corners).
     * @details Sensor mask type (linear indices or cuboid corners).
     */
    enum TSensorMaskType {INDEX = 0, CORNERS = 1};

    /// Get instance of the singleton class
    static TParameters& GetInstance();

    /// Destructor.
    virtual ~TParameters();

    /// Parse command line and read scalar values to init the class.
    void Init(int argc, char** argv);

    /// Select cuda device for execution.
    void SelectDevice();

    /// Print the simulation setup (all parameters)
    void PrintSimulatoinSetup();

    /// Get class with CUDA Parameters (runtime setup), cosnt version
    const TCUDAParameters& GetCudaParameters() const {return cudaParameters;};
    /// Get class with CUDA Parameters (runtime setup)
    TCUDAParameters&       GetCudaParameters()       {return cudaParameters;};

    /// Get input file handle
    THDF5_File& GetInputFile()        {return inputFile;};
    /// Get output file handle
    THDF5_File& GetOutputFile()       {return outputFile;};
    /// Gest checkpoint file handle
    THDF5_File& GetCheckpointFile()   {return checkpointFile;};
    /// Get File header handle
    THDF5_FileHeader& GetFileHeader() {return fileHeader;};

    /// Read scalar values from the input HDF5 file.
    void ReadScalarsFromInputFile(THDF5_File& inputFile);
    /// Save scalar values into the output HDF5 file
    void SaveScalarsToFile(THDF5_File& outputFile);

    /// Full dimension sizes of the simulation (real classes).
    TDimensionSizes GetFullDimensionSizes()    const {return fullDimensionSizes; };
    /// Reduced dimension sizes of the simulation (complex classes).
    TDimensionSizes GetReducedDimensionSizes() const {return reducedDimensionSizes;};

    /// Get Nt value.
    size_t  Get_nt()               const {return nt;};
    /// Get simulation time step.
    size_t  Get_t_index()          const {return t_index;};
    /// Set simulation time step - should be used only when recovering from checkpoint
    void  Set_t_index(const size_t new_t_index) {t_index = new_t_index;};
    /// Increment simulation time step.
    void  Increment_t_index()            {t_index++;};

    /// Get dt value.
    float Get_dt()               const {return dt;};
    /// Get dx value.
    float Get_dx()               const {return dx;};
    /// Get dy value.
    float Get_dy()               const {return dy;};
    /// Get dz value.
    float Get_dz()               const {return dz;};

    /// Get c_ref value.
    float Get_c_ref()            const {return c_ref;};
    /// Get alpha_power value.
    float Get_alpha_power()      const {return alpha_power; };

    /// Get pml_x_size value.
    size_t  Get_pml_x_size()     const {return pml_x_size;};
    /// Get pml_y_size value.
    size_t  Get_pml_y_size()     const {return pml_y_size;};
    /// Get pml_z_size value.
    size_t  Get_pml_z_size()     const {return pml_z_size;};

    /// Get pml_x_alpha_size value.
    float Get_pml_x_alpha_size() const {return pml_x_alpha;};
    /// Get pml_y_alpha_size value.
    float Get_pml_y_alpha_size() const {return pml_y_alpha;};
    /// Get pml_z_alpha_size value.
    float Get_pml_z_alpha_size() const {return pml_z_alpha;};

    /// Get ux_source_flag value.
    size_t  Get_ux_source_flag() const {return ux_source_flag;};
    /// Get uy_source_flag value.
    size_t  Get_uy_source_flag() const {return uy_source_flag;};
    /// Get uz_source_flag value.
    size_t  Get_uz_source_flag() const {return uz_source_flag;};
    /// Get u_source_many value.
    size_t  Get_u_source_many()  const {return u_source_many;};
    /// Get ux_source_mode value.
    size_t  Get_u_source_mode()  const {return u_source_mode;};

    /// Get p_source_flag value.
    size_t  Get_p_source_flag()  const {return p_source_flag; };
    /// Get p0_source_flag value.
    size_t  Get_p0_source_flag() const {return p0_source_flag;};
    /// Get p_source_many value.
    size_t  Get_p_source_many()  const {return p_source_many;};
    /// Get p_source_mode value.
    size_t  Get_p_source_mode()  const {return p_source_mode;};

    /// Get nonuniform_grid_flag value.
    size_t  Get_nonuniform_grid_flag()         const { return nonuniform_grid_flag;};
    /// Get absorbing_flag value.
    size_t  Get_absorbing_flag()               const { return absorbing_flag; };
    /// Get nonlinear_flag value.
    size_t  Get_nonlinear_flag()               const { return nonlinear_flag; };
    /// Get transducer_source_flag value.
    size_t  Get_transducer_source_flag()       const {return transducer_source_flag;};

    /// Get sensor mask type (linear or corners).
    TSensorMaskType Get_sensor_mask_type()     const {return sensor_mask_type;};
    /// Get sensor_mask_index_size value.
    size_t  Get_sensor_mask_index_size()       const {return sensor_mask_ind_size;}
    /// Get number of cubes in the mask.
    size_t  Get_sensor_mask_corners_size()     const {return sensor_mask_corners_size;};

    /// Get u_source_index_size value.
    size_t  Get_u_source_index_size()          const { return u_source_index_size;}
    /// Get p_source_index_size value.
    size_t  Get_p_source_index_size()          const { return p_source_index_size;}
    /// Get transducer_source_input_size value.
    size_t  Get_transducer_source_input_size() const { return transducer_source_input_size;}

    /// Get alpha_coeff_scalar_flag value.
    bool   Get_alpha_coeff_scalar_flag()   const { return alpha_coeff_scalar_flag;};
    /// Get alpha_coeff_scalar value. Note: cannot be const because of other optimizations.
    float& Get_alpha_coeff_scalar()              { return alpha_coeff_scalar; }

    /// Get c0_scalar_flag value.
    bool   Get_c0_scalar_flag()             const {return c0_scalar_flag; };
    /// Get c0_scalar value.
    float& Get_c0_scalar()                        {return c0_scalar; };

    /// Get absorb_eta_scalar value.
    float & Get_absorb_eta_scalar()               {return absorb_eta_scalar; };
    /// Get absorb_tau_scalar value.
    float & Get_absorb_tau_scalar()               {return absorb_tau_scalar; };

    /// Get BonA_scalar_flag value.
    bool   Get_BonA_scalar_flag()         const  { return BonA_scalar_flag; };
    /// Get BonA_scalar value.
    float& Get_BonA_scalar()                     { return BonA_scalar; };

    /// Get rho0_scalar_flag value.
    bool    Get_rho0_scalar_flag()        const  {return rho0_scalar_flag; };
    /// Get rho0_scalar value.
    float&  Get_rho0_scalar()                    {return rho0_scalar; };
    /// Get rho0_sgx_scalar value.
    float&  Get_rho0_sgx_scalar()                {return rho0_sgx_scalar; };
    /// Get rho0_sgy_scalar value.
    float&  Get_rho0_sgy_scalar()                {return rho0_sgy_scalar; };
    /// Get rho0_sgz_scalar value.
    float&  Get_rho0_sgz_scalar()                {return rho0_sgz_scalar; };

    /// Get input file name.
    std::string GetInputFileName()      const {return commandLineParameters.GetInputFileName();};
    /// Get output file name.
    std::string GetOutputFileName()     const {return commandLineParameters.GetOutputFileName();};
    /// Get checkpoint filename.
    std::string GetCheckpointFileName() const {return commandLineParameters.GetCheckpointFileName();};

    /// Get compression level.
    size_t GetCompressionLevel()   const {return commandLineParameters.GetCompressionLevel();};
    /// Get number of threads.
    size_t GetNumberOfThreads()    const {return commandLineParameters.GetNumberOfThreads();};
    /// Get progress print interval.
    size_t GetProgressPrintInterval() const {return commandLineParameters.GetProgressPrintInterval();};

    /// Get start time index for sensor recording.
    size_t GetStartTimeIndex()     const {return commandLineParameters.GetStartTimeIndex();};

    /// Is checkpoint enabled?
    bool   IsCheckpointEnabled()   const {return commandLineParameters.IsCheckpointEnabled();};
    /// Get checkpoint interval.
    size_t GetCheckpointInterval() const {return commandLineParameters.GetCheckpointInterval();};

    /// Is --version specified at the command line?
    bool IsVersion()         const {return commandLineParameters.IsVersion();};
    /// Is  -p or --p_raw specified at the command line?
    bool IsStore_p_raw()     const {return commandLineParameters.IsStore_p_raw();};
    /// Is --p_rms specified at the command line?
    bool IsStore_p_rms()     const {return commandLineParameters.IsStore_p_rms();};
    /// Is --p_max specified at the command line?
    bool IsStore_p_max()     const {return commandLineParameters.IsStore_p_max();};
    /// Is --p_min specified at the command line?
    bool IsStore_p_min()     const {return commandLineParameters.IsStore_p_min();};
    /// Is --p_max_all specified at the command line?
    bool IsStore_p_max_all() const {return commandLineParameters.IsStore_p_max_all();};
    /// Is --p_min_all specified at the command line?
    bool IsStore_p_min_all() const {return commandLineParameters.IsStore_p_min_all();};
    /// Is  --p_final specified at the command line?
    bool IsStore_p_final()   const {return commandLineParameters.IsStore_p_final();};

    /// Is -u or --u_raw specified at the command line?
    bool IsStore_u_raw()               const {return commandLineParameters.IsStore_u_raw();};
    /// Is --u_non_staggered_raw set?
    bool IsStore_u_non_staggered_raw() const {return commandLineParameters.IsStore_u_non_staggered_raw();};
    /// Is --u_raw specified at the command line?
    bool IsStore_u_rms()               const {return commandLineParameters.IsStore_u_rms();};
    /// Is --u_max specified at the command line?
    bool IsStore_u_max()               const {return commandLineParameters.IsStore_u_max();};
    /// Is --u_min specified at the command line?
    bool IsStore_u_min()               const {return commandLineParameters.IsStore_u_min();};
    /// Is --u_max_all specified at the command line?
    bool IsStore_u_max_all()           const {return commandLineParameters.IsStore_u_max_all();};
    /// Is --u_min_all specified at the command line?
    bool IsStore_u_min_all()           const {return commandLineParameters.IsStore_u_min_all();};
    /// Is --u_final specified at the command line
    bool IsStore_u_final()             const {return commandLineParameters.IsStore_u_final();};

    /// is --copy_mask set
    bool IsCopySensorMask()            const {return commandLineParameters.IsCopySensorMask();};

    /// Get Git hash of the code
    std::string GetGitHash()  const;

  protected:
    /// Constructor.
    TParameters();
    /// Copy constructor,
    TParameters(const TParameters& orig);

  private:
   /// Class with CUDA Parameters (runtime setup)
    TCUDAParameters cudaParameters;
    /// Class with command line parameters
    TCommandLineParameters commandLineParameters;

    /// Handle to the input HDF5 file.
    THDF5_File inputFile;
    /// Handle to the output HDF5 file.
    THDF5_File outputFile;
    /// Handle to the checkpoint HDF5 file.
    THDF5_File checkpointFile;

    /// Handle to file header.
    THDF5_FileHeader fileHeader;

    /// Nt value.
    size_t  nt;
    /// actual time index (time step of the simulation).
    size_t  t_index;

    /// dt value.
    float dt;
    /// dx value.
    float dx;
    /// dy value.
    float dy;
    /// dz value.
    float dz;

    /// c_ref value.
    float c_ref;
    /// alpha_power value.
    float alpha_power;

    /// Full 3D dimension sizes.
    TDimensionSizes fullDimensionSizes;
    /// Reduced 3D dimension sizes.
    TDimensionSizes reducedDimensionSizes;

    /// sensor mask type (0 = index, 1 = corners).
    TSensorMaskType sensor_mask_type;
    /// sensor_mask_ind_size value.
    size_t sensor_mask_ind_size;
    /// sensor_mask_corners_size - how many cuboids are in the mask.
    size_t sensor_mask_corners_size;

    /// u_source_index_size value.
    size_t u_source_index_size;
    /// p_source_index_size value.
    size_t p_source_index_size;
    /// transducer_source_input_size value.
    size_t transducer_source_input_size;

    /// ux_source_flag value.
    size_t ux_source_flag;
    /// uy_source_flag value.
    size_t uy_source_flag;
    /// uz_source_flag value.
    size_t uz_source_flag;

    /// p_source_flag value.
    size_t p_source_flag;
    /// p0_source_flag value.
    size_t p0_source_flag;
    /// transducer_source_flag value.
    size_t transducer_source_flag;

    /// u_source_many value.
    size_t u_source_many;
    /// u_source_mode value.
    size_t u_source_mode;

    /// p_source_mode value.
    size_t p_source_mode;
    /// p_source_many value.
    size_t p_source_many;

    /// nonuniform_grid_flag value.
    size_t nonuniform_grid_flag;
    /// absorbing_flag value.
    size_t absorbing_flag;
    /// nonlinear_flag value.
    size_t nonlinear_flag;

    /// pml_x_size value.
    size_t pml_x_size;
    /// pml_y_size value.
    size_t pml_y_size;
    /// pml_z_size value.
    size_t pml_z_size;

    /// pml_x_alpha value.
    float pml_x_alpha;
    /// pml_y_alpha value.
    float pml_y_alpha;
    /// pml_z_alpha value.
    float pml_z_alpha;

    /// alpha_coeff_scalar_flag value.
    bool  alpha_coeff_scalar_flag;
    /// alpha_coeff_scalar value.
    float alpha_coeff_scalar;

    /// c0_scalar_flag value.
    bool  c0_scalar_flag;
    /// c0_scalar value.
    float c0_scalar;

    /// absorb_eta_scalar value.
    float absorb_eta_scalar;
    /// absorb_tau_scalar value.
    float absorb_tau_scalar;

    /// BonA_scalar_flag value.
    bool  BonA_scalar_flag;
    /// BonA_scalar value.
    float BonA_scalar;

    /// rho0_scalar_flag value.
    bool  rho0_scalar_flag;
    /// rho0_scalar value.
    float rho0_scalar;
    /// rho0_sgx_scalar value.
    float rho0_sgx_scalar;
    /// rho0_sgy_scalar value.
    float rho0_sgy_scalar;
    /// rho0_sgz_scalar value.
    float rho0_sgz_scalar;

    /// singleton flag
    static bool parametersInstanceFlag;
    /// singleton instance
    static TParameters* parametersSingleInstance;

  private:

};// end of TParameters
//--------------------------------------------------------------------------------------------------

#endif	/* PARAMETERS_H */
