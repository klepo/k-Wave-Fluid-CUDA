/**
 * @file        HDF5_File.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the HDF5 related classes.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        27 July     2012, 14:14 (created) \n
 *              28 June     2017, 15:08 (revised)
 *
 *
 * @section HDF HDF5 File Structure
 *
 * The CUDA/C++ code has been designed as a standalone application which is not dependent  on MATLAB
 * libraries or a MEX interface. This is of particular importance when using servers and
 * supercomputers without MATLAB support. For this reason, simulation data must be transferred
 * between the CUDA/C++ code and MATLAB using external input and output files. These files are
 * stored using the Hierarchical Data Format HDF5 (http://www.hdfgroup.org/HDF5/). This is a data
 * model, library, and file format for storing and managing data. It supports a variety of
 * datatypes, and is designed for flexible and efficient I/O and for high volume and complex data.
 * The HDF5 technology suite includes tools and applications for managing, manipulating, viewing,
 * and analysing data in the HDF5 format.
 *
 *
 * Each HDF5 file is a container for storing a variety of scientific data and is composed of two
 * primary types of objects: groups and datasets. An HDF5 group is a structure  containing zero or
 * more HDF5 objects, together with supporting metadata. A HDF5 group can be seen as a disk folder.
 * An HDF5 dataset is a multidimensional array of data elements, together with supporting metadata.
 * A HDF5 dataset can be seen as a disk file. Any HDF5 group or dataset may also have an associated
 * attribute list. An HDF5 attribute is a user-defined HDF5 structure that provides extra
 * information about an HDF5 object. More information can be obtained from the HDF5 documentation
 * (http://www.hdfgroup.org/HDF5/doc/index.html).
 *
 *
 * kspaceFirstOrder3D-CUDA v1.1 introduces a new version of the HDF5 input and output file format.
 * The code is happy to work with both versions (1.0 and 1.1), however when working with an input
 * file of version 1.0, some features are not supported, namely the cuboid sensor mask, and
 * u_non_staggered_raw. When running from within the actual MATLAB K-Wave Toolbox, the files will
 * always be generated in version 1.1
 *
 * The HDF5 input file for the CUDA/C++ simulation code contains a file header with brief
 * description of the simulation stored in string attributes, and the root group `/' which stores
 * all the simulation properties in the form of 3D datasets (a complete list of input datasets is
 * given bellow).
 * The HDF5 checkpoint file contains the same file header as the input file and the root group `/'
 * with a few datasets keeping the actual simulation state.
 * The HDF5 output file contains a file header with the simulation description as well as
 * performance statistics, such as the simulation  time and memory consumption, stored in string
 * attributes.

 * The results of the simulation are stored in the root group `/' in the form of 3D datasets. If the
 * linear sensor mask is used, all output quantities are stored as datasets in the root group. If
 * the cuboid corners sensor mask is used, the sampled quantities form private groups containing
 * datasets on per cuboid basis.
 *
 *
 * \verbatim
==============================================================================================================
                                        Input File/Checkpoint File Header
=============================================================================================================
created_by                              Short description of the tool that created this file
creation_date                           Date when the file was created
file_description                        Short description of the content of the file (e.g. simulation name)
file_type                               Type of the file (input)
major_version                           Major version of the file definition (1)
minor_version                           Minor version of the file definition (1)
==============================================================================================================
 \endverbatim
 *
 * \verbatim
==============================================================================================================
                                        Output File Header
==============================================================================================================
created_by                              Short description of the tool that created this file
creation_date                           Date when the file was created
file_description                        Short description of the content of the file (e.g. simulation name)
file_type                               Type of the file (output)
major_version                           Major version of the file definition (1)
minor_version                           Minor version of the file definition (1)
-------------------------------------------------------------------------------------------------------------
host_names                              List of hosts (computer names) the simulation was executed on
number_of_cpu_cores                     Number of CPU cores used for the simulation
data_loading_phase_execution_time       Time taken to load data from the file
pre-processing_phase_execution_time     Time taken to pre-process data
simulation_phase_execution_time         Time taken to run the simulation
post-processing_phase_execution_time    Time taken to complete the post-processing phase
total_execution_time                    Total execution time
peak_core_memory_in_use                 Peak memory required per core during the simulation
total_memory_in_use Total               Peak memory in use
==============================================================================================================
 \endverbatim
 *
 *
 *
 * The input and checkpoint file stores all quantities as three dimensional datasets with dimension
 * sizes designed by <tt>(Nx, Ny, Nz)</tt>. In order to support scalars and 1D and 2D arrays, the
 * unused dimensions are set to 1. For example, scalar variables are stored with a dimension size of
 * <tt>(1,1,1)</tt>, 1D vectors oriented in y-direction are stored with a dimension  size of
 * <tt>(1, Ny, 1)</tt>,  and so on. If the dataset stores a complex variable, the real and imaginary
 * parts are stored in an interleaved layout and the lowest used dimension size is doubled (i.e.,
 * Nx for a 3D matrix, Ny for a 1D vector oriented in the y-direction). The datasets are physically
 * stored in row-major order (in contrast to column-major order used by MATLAB) using either the
 * <tt>`H5T_IEEE_F32LE'</tt> data type for floating point datasets or <tt>`H5T_STD_U64LE'</tt> for
 * integer based datasets. All the datasets are store under the root group.
 *
 *
 * The output file of version 1.0 could only store recorded quantities as 3D datasets  under the
 * root group. However, with version 1.1 and the new cuboid corner sensor mask, the sampled
 * quantities may be laid out as 4D quantities stored under specific groups. The dimensions are
 * always <tt>(Nx, Ny, Nz, Nt)</tt>, every sampled cuboid is stored as a distinct dataset and the
 * datasets are grouped under a group named by the quantity stored. This makes the file
 * clearly readable and easy to parse.
 *
 *
 * In order to enable compression and more efficient data processing, big datasets are not stored as
 * monolithic blocks but broken into chunks that may be compressed by the ZIP library  and stored
 * separately. The chunk size is defined as follows:
 *
 * \li <tt> (1M elements, 1, 1) </tt> in the case of 1D variables - index sensor mask (8MB blocks).
 * \li <tt> (Nx, Ny, 1)         </tt> in the case of 3D variables (one 2D slab).
 * \li <tt> (Nx, Ny, Nz, 1)     </tt> in the case of 4D variables (one time step).
 * \li <tt> (N_sensor_points, 1, 1) </tt> in the case of the output time series (one time step of
 *                                    the simulation).
 *
 *
 * All datasets have two attributes that specify the content of the dataset. The \c `data_type'
 * attribute specifies the data type of the dataset. The admissible values are either \c `float' or
 * \c `long'. The \c `domain_type' attribute specifies the domain of the dataset. The admissible
 * values are either \c `real' for the real domain or \c `complex' for the complex domain. The
 * C++ code reads these attributes and checks their values.
 *
 *
 *
 * \verbatim
==============================================================================================================
                                        Input File Datasets
==============================================================================================================
Name                            Size           Data type       Domain Type      Condition of Presence
==============================================================================================================
  1. Simulation Flags
--------------------------------------------------------------------------------------------------------------
  ux_source_flag                (1, 1, 1)       long           real
  uy_source_flag                (1, 1, 1)       long           real
  uz_source_flag                (1, 1, 1)       long           real
  p_source_flag                 (1, 1, 1)       long           real
  p0_source_flag                (1, 1, 1)       long           real
  transducer_source_flag        (1, 1, 1)       long           real
  nonuniform_grid_flag          (1, 1, 1)       long           real             must be set to 0
  nonlinear_flag                (1, 1, 1)       long           real
  absorbing_flag                (1, 1, 1)       long           real
--------------------------------------------------------------------------------------------------------------
  2. Grid Properties
--------------------------------------------------------------------------------------------------------------
  Nx                            (1, 1, 1)       long           real
  Ny                            (1, 1, 1)       long           real
  Nz                            (1, 1, 1)       long           real
  Nt                            (1, 1, 1)       long           real
  dt                            (1, 1, 1)       float          real
  dx                            (1, 1, 1)       float          real
  dy                            (1, 1, 1)       float          real
  dz                            (1, 1, 1)       float          real
--------------------------------------------------------------------------------------------------------------
  3 Medium Properties
--------------------------------------------------------------------------------------------------------------
  3.1 Regular Medium Properties
  rho0                          (Nx, Ny, Nz)    float          real             heterogenous
                                (1, 1, 1)       float          real             homogenous
  rho0_sgx                      (Nx, Ny, Nz)    float          real             heterogenous
                                (1, 1, 1)       float          real             homogenous
  rho0_sgy                      (Nx, Ny, Nz)    float          real             heterogenous
                                (1, 1, 1)       float          real             homogenous
  rho0_sgz                      (Nx, Ny, Nz)    float          real             heterogenous
                                (1, 1, 1)       float          real             homogenous
  c0                            (Nx, Ny, Nz)    float          real             heterogenous
                                (1, 1, 1)       float          real             homogenous
  c_ref                         (1, 1, 1)       float          real

  3.2 Nonlinear Medium Properties (defined if (nonlinear_flag == 1))
  BonA                          (Nx, Ny, Nz)    float          real             heterogenous
                                (1, 1, 1)       float          real             homogenous

  3.3 Absorbing Medium Properties (defined if (absorbing_flag == 1))
  alpha_coef                    (Nx, Ny, Nz)    float          real             heterogenous
                                (1, 1, 1)       float          real             homogenous
  alpha_power                   (1, 1, 1)       float          real
--------------------------------------------------------------------------------------------------------------
  4. Sensor Variables
--------------------------------------------------------------------------------------------------------------
  sensor_mask_type              (1, 1, 1)       long           real             file version 1.1 (0 = index, 1 = corners)
  sensor_mask_index             (Nsens, 1, 1)   long           real             file version 1.0 always, File version 1.1 if sensor_mask_type == 0
  sensor_mask_corners           (Ncubes, 6, 1)  long           real             file version 1.1, if sensor_mask_type == 1
--------------------------------------------------------------------------------------------------------------
  5 Source Properties
--------------------------------------------------------------------------------------------------------------
  5.1 Velocity Source Terms (defined if (ux_source_flag == 1 || uy_source_flag == 1 || uz_source_flag == 1))
  u_source_mode                 (1, 1, 1)          long        real
  u_source_many                 (1, 1, 1)          long        real
  u_source_index                (Nsrc, 1, 1)       long        real
  ux_source_input               (1, Nt_src, 1)     float       real             u_source_many == 0
                                (Nsrc, Nt_src, 1)  float       real             u_source_many == 1
  uy_source_input               (1, Nt_src,  1)    float       real             u_source_many == 0
                                (Nsrc, Nt_src, 1)  float       real             u_source_many == 1
  uz_source_input               (1, Nt_src, 1)     float       real             u_source_many == 0
                                (Nt_src, Nsrc, 1)  float       real             u_source_many == 1

  5.2 Pressure Source Terms (defined if (p_source_flag == 1))
  p_source_mode                 (1, 1, 1)          long        real
  p_source_many                 (1, 1, 1)          long        real
  p_source_index                (Nsrc, 1, 1)       long        real
  p_source_input                (Nsrc, Nt_src, 1)  float       real             p_source_many == 1
                                (1, Nt_src, 1)     float       real             p_source_many == 0

  5.3 Transducer Source Terms (defined if (transducer_source_flag == 1))
  u_source_index                (Nsrc, 1, 1)       long        real
  transducer_source_input       (Nt_src, 1, 1)     float       real
  delay_mask                    (Nsrc, 1, 1)       float       real

  5.4 IVP Source Terms (defined if ( p0_source_flag ==1))
  p0_source_input               (Nx, Ny, Nz)        float      real
--------------------------------------------------------------------------------------------------------------
  6. K-space and Shift Variables
--------------------------------------------------------------------------------------------------------------
  ddx_k_shift_pos_r             (Nx/2 + 1, 1, 1)  float        complex
  ddx_k_shift_neg_r             (Nx/2 + 1, 1, 1)  float        complex
  ddy_k_shift_pos               (1, Ny, 1)        float        complex
  ddy_k_shift_neg               (1, Ny, 1)        float        complex
  ddz_k_shift_pos               (1, 1, Nz)        float        complex
  ddz_k_shift_neg               (1, 1, Nz)        float        complex
  x_shift_neg_r                 (Nx/2 + 1, 1, 1)  float        complex          file version 1.1
  y_shift_neg_r                 (1, Ny/2 + 1, 1)  float        complex          file version 1.1
  z_shift_neg_r                 (1, 1, Nz/2)      float        complex          file version 1.1
--------------------------------------------------------------------------------------------------------------
  7. PML Variables
--------------------------------------------------------------------------------------------------------------
  pml_x_size                    (1, 1, 1)       long           real
  pml_y_size                    (1, 1, 1)       long           real
  pml_z_size                    (1, 1, 1)       long           real
  pml_x_alpha                   (1, 1, 1)       float          real
  pml_y_alpha                   (1, 1, 1)       float          real
  pml_z_alpha                   (1, 1, 1)       float          real

  pml_x                         (Nx, 1, 1)      float          real
  pml_x_sgx                     (Nx, 1, 1)      float          real
  pml_y                         (1, Ny, 1)      float          real
  pml_y_sgy                     (1, Ny, 1)      float          real
  pml_z                         (1, 1, Nz)      float          real
  pml_z_sgz                     (1, 1, Nz)      float          real
==============================================================================================================
 \endverbatim

\verbatim
==============================================================================================================
                                        Checkpoint File Datasets
==============================================================================================================
Name                            Size           Data type       Domain Type      Condition of Presence
==============================================================================================================
  1. Grid Properties
--------------------------------------------------------------------------------------------------------------
  Nx                            (1, 1, 1)       long           real
  Ny                            (1, 1, 1)       long           real
  Nz                            (1, 1, 1)       long           real
  Nt                            (1, 1, 1)       long           real
  t_index                       (1, 1, 1)       long           real
--------------------------------------------------------------------------------------------------------------
  2. Simulation State
--------------------------------------------------------------------------------------------------------------
  p                            (Nx, Ny, Nz)    float           real
  ux_sgx                       (Nx, Ny, Nz)    float           real
  uy_sgy                       (Nx, Ny, Nz)    float           real
  uz_sgz                       (Nx, Ny, Nz)    float           real
  rhox                         (Nx, Ny, Nz)    float           real
  rhoy                         (Nx, Ny, Nz)    float           real
  rhoz                         (Nx, Ny, Nz)    float           real
--------------------------------------------------------------------------------------------------------------
\endverbatim


 \verbatim
==============================================================================================================
                                        Output File Datasets
==============================================================================================================
Name                            Size           Data type       Domain Type      Condition of Presence
==============================================================================================================
  1. Simulation Flags
--------------------------------------------------------------------------------------------------------------
  ux_source_flag                (1, 1, 1)       long           real
  uy_source_flag                (1, 1, 1)       long           real
  uz_source_flag                (1, 1, 1)       long           real
  p_source_flag                 (1, 1, 1)       long           real
  p0_source_flag                (1, 1, 1)       long           real
  transducer_source_flag        (1, 1, 1)       long           real
  nonuniform_grid_flag          (1, 1, 1)       long           real
  nonlinear_flag                (1, 1, 1)       long           real
  absorbing_flag                (1, 1, 1)       long           real
  u_source_mode                 (1, 1, 1)       long           real             if u_source
  u_source_many                 (1, 1, 1)       long           real             if u_source
  p_source_mode                 (1, 1, 1)       long           real             if p_source
  p_source_many                 (1, 1, 1)       long           real             if p_source
--------------------------------------------------------------------------------------------------------------
  2. Grid Properties
--------------------------------------------------------------------------------------------------------------
  Nx                            (1, 1, 1)       long           real
  Ny                            (1, 1, 1)       long           real
  Nz                            (1, 1, 1)       long           real
  Nt                            (1, 1, 1)       long           real
  dt                            (1, 1, 1)       float          real
  dx                            (1, 1, 1)       float          real
  dy                            (1, 1, 1)       float          real
  dz                            (1, 1, 1)       float          real
-------------------------------------------------------------------------------------------------------------
  3. PML Variables
--------------------------------------------------------------------------------------------------------------
  pml_x_size                    (1, 1, 1)       long           real
  pml_y_size                    (1, 1, 1)       long           real
  pml_z_size                    (1, 1, 1)       long           real
  pml_x_alpha                   (1, 1, 1)       float          real
  pml_y_alpha                   (1, 1, 1)       float          real
  pml_z_alpha                   (1, 1, 1)       float          real

  pml_x                         (Nx, 1, 1)      float          real
  pml_x_sgx                     (Nx, 1, 1)      float          real
  pml_y                         (1, Ny, 1)      float          real
  pml_y_sgy                     (1, Ny, 1)      float          real
  pml_z                         (1, 1, Nz)      float          real
  pml_z_sgz                     (1, 1, Nz)      float          real
--------------------------------------------------------------------------------------------------------------
  4. Sensor Variables (present if --copy_sensor_mask)
--------------------------------------------------------------------------------------------------------------
  sensor_mask_type              (1, 1, 1)       long           real             file version 1.1 and --copy_sensor_mask
  sensor_mask_index             (Nsens, 1, 1)   long           real             file version 1.1 and if sensor_mask_type == 0
  sensor_mask_corners           (Ncubes, 6, 1)  long           real             file version 1.1 and if sensor_mask_type == 1
--------------------------------------------------------------------------------------------------------------
  5a. Simulation Results: if sensor_mask_type == 0 (index), or File version == 1.0
--------------------------------------------------------------------------------------------------------------
  p                             (Nsens, Nt - s, 1) float       real             -p or --p_raw
  p_rms                         (Nsens, 1, 1)      float       real             --p_rms
  p_max                         (Nsens, 1, 1)      float       real             --p_max
  p_min                         (Nsens, 1, 1)      float       real             --p_min
  p_max_all                     (Nx, Ny, Nz)       float       real             --p_max_all
  p_min_all                     (Nx, Ny, Nz)       float       real             --p_min_all
  p_final                       (Nx, Ny, Nz)       float       real             --p_final


  ux                            (Nsens, Nt - s, 1) float       real             -u or --u_raw
  uy                            (Nsens, Nt - s, 1) float       real             -u or --u_raw
  uz                            (Nsens, Nt - s, 1) float       real             -u or --u_raw

  ux_non_staggered              (Nsens, Nt - s, 1) float       real             --u_non_staggered_raw (File version ==1.1)
  uy_non_staggered              (Nsens, Nt - s, 1) float       real             --u_non_staggered_raw (File version ==1.1)
  uz_non_staggered              (Nsens, Nt - s, 1) float       real             --u_non_staggered_raw (File version ==1.1)

  ux_rms                        (Nsens, 1, 1)      float       real             --u_rms
  uy_rms                        (Nsens, 1, 1)      float       real             --u_rms
  uz_rms                        (Nsens, 1, 1)      float       real             --u_rms

  ux_max                        (Nsens, 1, 1)      float       real             --u_max
  uy_max                        (Nsens, 1, 1)      float       real             --u_max
  uz_max                        (Nsens, 1, 1)      float       real             --u_max

  ux_min                        (Nsens, 1, 1)      float       real             --u_min
  uy_min                        (Nsens, 1, 1)      float       real             --u_min
  uz_min                        (Nsens, 1, 1)      float       real             --u_min

  ux_max_all                    (Nx, Ny, Nz)       float       real             --u_max_all
  uy_max_all                    (Nx, Ny, Nz)       float       real             --u_max_all
  uz_max_all                    (Nx, Ny, Nz)       float       real             --u_max_all

  ux_min_all                    (Nx, Ny, Nz)       float       real             --u_min_all
  uy_min_all                    (Nx, Ny, Nz)       float       real             --u_min_all
  uz_min_all                    (Nx, Ny, Nz)       float       real             --u_min_all

  ux_final                      (Nx, Ny, Nz)       float       real             --u_final
  uy_final                      (Nx, Ny, Nz)       float       real             --u_final
  uz_final                      (Nx, Ny, Nz)       float       real             --u_final
--------------------------------------------------------------------------------------------------------------
  5b. Simulation Results: if sensor_mask_type == 1 (corners) and file version == 1.1
--------------------------------------------------------------------------------------------------------------
  /p                            group of datasets, one per cuboid               -p or --p_raw
  /p/1                          (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid
  /p/2                          (Cx, Cy, Cz, Nt-s) float       real               2nd sampled cuboid, etc.

  /p_rms                        group of datasets, one per cuboid               --p_rms
  /p_rms/1                      (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid

  /p_max                        group of datasets, one per cuboid               --p_max
  /p_max/1                      (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid

  /p_min                        group of datasets, one per cuboid               --p_min
  /p_min/1                      (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid

  p_max_all                     (Nx, Ny, Nz)       float       real             --p_max_all
  p_min_all                     (Nx, Ny, Nz)       float       real             --p_min_all
  p_final                       (Nx, Ny, Nz)       float       real             --p_final


  /ux                           group of datasets, one per cuboid               -u or --u_raw
  /ux/1                         (Cx, Cy, Cz, Nt-s) float       real                1st sampled cuboid
  /uy                           group of datasets, one per cuboid               -u or --u_raw
  /uy/1                         (Cx, Cy, Cz, Nt-s) float       real                1st sampled cuboid
  /uz                           group of datasets, one per cuboid               -u or --u_raw
  /uz/1                         (Cx, Cy, Cz, Nt-s) float       real                1st sampled cuboid

  /ux_non_staggered             group of datasets, one per cuboid               --u_non_staggered_raw
  /ux_non_staggered/1           (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid
  /uy_non_staggered             group of datasets, one per cuboid               --u_non_staggered_raw
  /uy_non_staggered/1           (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid
  /uz_non_staggered             group of datasets, one per cuboid               --u_non_staggered_raw
  /uz_non_staggered/1           (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid

  /ux_rms                       group of datasets, one per cuboid               --u_rms
  /ux_rms/1                     (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid
  /uy_rms                       group of datasets, one per cuboid               --u_rms
  /uy_rms/1                     (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid
  /uz_rms                       group of datasets, one per cuboid               --u_rms
  /uy_rms/1                     (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid

  /ux_max                       group of datasets, one per cuboid               --u_max
  /ux_max/1                     (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid
  /uy_max                       group of datasets, one per cuboid               --u_max
  /ux_max/1                     (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid
  /uz_max                       group of datasets, one per cuboid               --u_max
  /ux_max/1                     (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid

  /ux_min                       group of datasets, one per cuboid               --u_min
  /ux_min/1                     (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid
  /uy_min                       group of datasets, one per cuboid               --u_min
  /ux_min/1                     (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid
  /uz_min                       group of datasets, one per cuboid               --u_min
  /ux_min/1                     (Cx, Cy, Cz, Nt-s) float       real               1st sampled cuboid

  ux_max_all                    (Nx, Ny, Nz)       float       real             --u_max_all
  uy_max_all                    (Nx, Ny, Nz)       float       real             --u_max_all
  uz_max_all                    (Nx, Ny, Nz)       float       real             --u_max_all

  ux_min_all                    (Nx, Ny, Nz)       float       real             --u_min_all
  uy_min_all                    (Nx, Ny, Nz)       float       real             --u_min_all
  uz_min_all                    (Nx, Ny, Nz)       float       real             --u_min_all

  ux_final                      (Nx, Ny, Nz)       float       real             --u_final
  uy_final                      (Nx, Ny, Nz)       float       real             --u_final
  uz_final                      (Nx, Ny, Nz)       float       real             --u_final
==============================================================================================================
\endverbatim
 *
 *
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

#ifndef THDF5_FILE_H
#define	THDF5_FILE_H


#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstring>
#include <map>

// Linux build
#ifdef __linux__
  #include <unistd.h>
#endif
// Windows build
#ifdef _WIN64
  #include <io.h>
#endif

#include <Utils/DimensionSizes.h>
#include <Utils/MatrixNames.h>


// Class with File header
class THDF5_FileHeader;

/**
 * @class THDF5_File
 * @brief Class wrapping the HDF5 routines.
 * @details This class is responsible for working with HDF5 files. It offers routines to manage
 * files (create, open, close) as well as creating, reading and modifying the contents
 * (groups and datasets).
 */
class THDF5_File
{
  public:

    /**
     * @enum    TMatrixDataType
     * @brief   HDF5 matrix data type (float or uint64).
     * @details HDF5 matrix data type (float or uint64).
     */
    enum class TMatrixDataType
    {
      FLOAT = 0,
      LONG  = 1
    };

    /**
     * @enum    TMatrixDomainType
     * @brief   HDF5 Matrix domain type (real or complex).
     * @details HDF5 Matrix domain type (real or complex).
     */
    enum class TMatrixDomainType
    {
      REAL    = 0,
      COMPLEX = 1
    };


    /// Constructor of the class.
    THDF5_File();
    /// Destructor.
    virtual ~THDF5_File();

    //----------------------- Basic file operations --------------------------//
    /// Create a file.
    void Create(const std::string& fileName,
                unsigned int       flags = H5F_ACC_TRUNC);

    /// Open a file.
    void Open(const std::string& fileName,
              unsigned int       flags  = H5F_ACC_RDONLY);
    /**
     * @brief   Is the file opened?
     * @details Is the file opened?
     * @return  true if the file is opened.
     */
    bool IsOpen() const {return file != H5I_BADID;};

    /**
     * @brief      Does the file exist?
     * @details    Check if the file exist.
     * @param [in] fileName
     * @return true if the file exists.
     */
    static bool IsHDF5(const std::string& fileName);

    /// Close file.
    void Close();

    //----------------------------------- Group manipulators -------------------------------------//
    /// Create a HDF5 group at a specified place in the file tree.
    hid_t CreateGroup(const hid_t  parentGroup,
                      TMatrixName& groupName);
    /// Open a HDF5 group at a specified place in the file tree.
    hid_t OpenGroup(const hid_t  parentGroup,
                    TMatrixName& groupName);
    /// Close group.
    void CloseGroup(const hid_t group);

    /**
     * @brief   Get handle to the root group.
     * @details Get handle to the root group.
     * @return  handle to the root group
     */
    hid_t GetRootGroup() const {return file;};


    //---------------------------------- Dataset manipulators -------------------------------------//
    /// Open the HDF5 dataset at a specified place in the file tree.
    hid_t OpenDataset(const hid_t parentGroup,
                      TMatrixName& datasetName);

    /// Create a float HDF5 dataset at a specified place in the file tree (3D/4D).
    hid_t CreateFloatDataset(const hid_t            parentGroup,
                             TMatrixName&           datasetName,
                             const TDimensionSizes& dimensionSizes,
                             const TDimensionSizes& chunkSizes,
                             const size_t           compressionLevel);

    /// Create an index HDF5 dataset at a specified place in the file tree (3D only).
    hid_t CreateIndexDataset(const hid_t            parentGroup,
                             TMatrixName&           datasetName,
                             const TDimensionSizes& dimensionSizes,
                             const TDimensionSizes& chunkSizes,
                             const size_t           compressionLevel);

    /// Close the HDF5 dataset.
    void  CloseDataset (const hid_t dataset);


    //----------------------------- Dataset Read/Write operations --------------------------------//
    /// Write a hyper-slab into the dataset - float dataset.
    void WriteHyperSlab(const hid_t            dataset,
                        const TDimensionSizes& position,
                        const TDimensionSizes& size,
                        const float*           data);
    /// Write a hyper-slab into the dataset - long dataset.
    void WriteHyperSlab(const hid_t            dataset,
                        const TDimensionSizes& position,
                        const TDimensionSizes& size,
                        const size_t*          data);

    /// Write a cuboid selected within the matrixData into a hyperslab.
    void WriteCuboidToHyperSlab(const hid_t            dataset,
                                const TDimensionSizes& hyperslabPosition,
                                const TDimensionSizes& cuboidPosition,
                                const TDimensionSizes& cuboidSize,
                                const TDimensionSizes& matrixDimensions,
                                const float*           mMatrixData);

    /// Write sensor data selected by the sensor mask - Occasionally very slow, do not use!
    void WriteDataByMaskToHyperSlab(const hid_t            dataset,
                                    const TDimensionSizes& hyperslabPosition,
                                    const size_t           indexSensorSize,
                                    const size_t*          indexSensorData,
                                    const TDimensionSizes& matrixDimensions,
                                    const float*           matrixData);

    /// Write the scalar value under a specified group, float value.
    void WriteScalarValue(const hid_t  parentGroup,
                          TMatrixName& datasetName,
                          const float  value);
    /// Write the scalar value under a specified group, index value.
    void WriteScalarValue(const hid_t  parentGroup,
                          TMatrixName& datasetName,
                          const size_t value);

    /// Read the scalar value under a specified group, float value.
    void ReadScalarValue(const hid_t  parentGroup,
                         TMatrixName& datasetName,
                         float&       value);
    /// Read the scalar value under a specified group, index value.
    void ReadScalarValue(const hid_t  parentGroup,
                         TMatrixName& datasetName,
                         size_t&      value);

    /// Read data from the dataset under a specified group, float dataset.
    void ReadCompleteDataset(const hid_t            parentGroup,
                             TMatrixName&           datasetName,
                             const TDimensionSizes& dimensionSizes,
                             float*                 data);
    /// Read data from the dataset under a specified group, index dataset.
    void ReadCompleteDataset(const hid_t            parentGroup,
                             TMatrixName&           datasetName,
                             const TDimensionSizes& dimensionSizes,
                             size_t*                data);

    //---------------------------- Attributes Read/Write operations ------------------------------//

    /// Get dimension sizes of the dataset  under a specified group.
    TDimensionSizes GetDatasetDimensionSizes(const hid_t  parentGroup,
                                             TMatrixName& datasetName);

    /// Get number of dimensions of the dataset  under a specified group.
    size_t GetDatasetNumberOfDimensions(const hid_t  parentGroup,
                                        TMatrixName& datasetName);

    /// Get dataset element count under a specified group.
    size_t GetDatasetElementCount(const hid_t  parentGroup,
                                  TMatrixName& datasetName);


    /// Write matrix data type into the dataset under a specified group.
    void WriteMatrixDataType (const hid_t                parentGroup,
                              TMatrixName&               datasetName,
                              const TMatrixDataType& matrixDataType);
    /// Write matrix domain type into the dataset under the root group.
    void WriteMatrixDomainType(const hid_t                  parentGroup,
                               TMatrixName&                 datasetName,
                               const TMatrixDomainType& matrixDomainType);

    /// Read matrix data type from the dataset.
    THDF5_File::TMatrixDataType   ReadMatrixDataType(const hid_t  parentGroup,
                                                     TMatrixName& datasetName);
    /// Read matrix domain type from the dataset under a specified group.
    THDF5_File::TMatrixDomainType ReadMatrixDomainType(const hid_t  parentGroup,
                                                       TMatrixName& datasetName);


    /// Write string attribute into the dataset under the root group.
    void   WriteStringAttribute(const hid_t        parentGroup,
                                TMatrixName&       datasetName,
                                TMatrixName&       attributeName,
                                const std::string& value);
    /// Read string attribute from the dataset under the root group.
    std::string ReadStringAttribute(const hid_t  parentGroup,
                                    TMatrixName& datasetName,
                                    TMatrixName& attributeName);

   protected:

    /// Copy constructor is not allowed for public.
    THDF5_File(const THDF5_File& src);
    /// Operator = is not allowed for public.
    THDF5_File& operator= (const THDF5_File& src);

  private:
    /// String representation of the Domain type in the HDF5 file.
    static const std::string matrixDomainTypeName;
    /// String representation of the Data type in the HDF5 file.
    static const std::string matrixDataTypeName;

    /// String representation of different domain types.
    static const std::string matrixDomainTypeNames[];
    /// String representation of different data types.
    static const std::string matrixDataTypeNames[];

    /// HDF file handle.
    hid_t  file;
    /// File name.
    std::string fileName;
}; // THDF5_File
//--------------------------------------------------------------------------------------------------


/**
 * @class THDF5_FileHeader
 * @brief Class for HDF5 header.
 * @details This class manages all information that can be stored in the input output or checkpoint
 * file header.
 */
class THDF5_FileHeader
{
  public:

    /**
     * @enum  TFileHeaderItems
     * @brief List of all header items.
     * @details List of all header items.
     * @todo  In the future we should add number of GPUs, peak GPU memory.
     */
    enum class TFileHeaderItems
    {
      CREATED_BY                   =  0,
      CREATION_DATA                =  1,
      FILE_DESCRIPTION             =  2,
      MAJOR_VERSION                =  3,
      MINOR_VERSION                =  4,
      FILE_TYPE                    =  5,
      HOST_NAME                    =  6,
      TOTAL_MEMORY_CONSUMPTION     =  7,
      PEAK_CORE_MEMORY_CONSUMPTION =  8,
      TOTAL_EXECUTION_TIME         =  9,
      DATA_LOAD_TIME               = 10,
      PREPROCESSING_TIME           = 11,
      SIMULATION_TIME              = 12,
      POST_PROCESSING_TIME         = 13,
      NUMBER_OF_CORES              = 14
    };

    /**
     * @enum    TFileType
     * @brief   HDF5 file type.
     * @details HDF5 file type.
     */
    enum class TFileType
    {
      INPUT      = 0,
      OUTPUT     = 1,
      CHECKPOINT = 2,
      UNKNOWN    = 3
    };

    /**
     * @enum    TFileVersion
     * @brief   HDF5 file version.
     * @details HDF5 file version.
     */
    enum class TFileVersion
    {
      VERSION_10      = 0,
      VERSION_11      = 1,
      VERSION_UNKNOWN = 2
    };

    /// Constructor.
    THDF5_FileHeader();
    /// Copy constructor.
    THDF5_FileHeader(const THDF5_FileHeader& src);
    /// Destructor.
    ~THDF5_FileHeader();

    /// Read header from the input file.
    void ReadHeaderFromInputFile(THDF5_File& inputFile);
    /// Read Header from output file (necessary for checkpoint-restart).
    void ReadHeaderFromOutputFile(THDF5_File& outputFile);
    /// Read Header from checkpoint file (necessary for checkpoint-restart).
    void ReadHeaderFromCheckpointFile(THDF5_File& checkpointFile);

    /// Write header to the output file
    void WriteHeaderToOutputFile(THDF5_File& outputFile);
    /// Write header to the output file
    void WriteHeaderToCheckpointFile(THDF5_File& checkpointFile);

    /**
     * @brief Set code name.
     * @details Set code name to the header.
     * @param [in] codeName - code version
     */
    void SetCodeName(const std::string& codeName)
    {
      headerValues[TFileHeaderItems::CREATED_BY] = codeName;
    };

    /// Set creation time.
    void SetActualCreationTime();

    /**
     * @brief   Get string version of current Major version.
     * @details Get string version of current Major version.
     * @return  Current major version
     */
    static std::string GetCurrentHDF5_MajorVersion()
    {
      return hdf5_MajorFileVersionsNames[0];
    };

    /**
     * @brief   Get string version of current Minor version.
     * @details Get string version of current Minor version.
     * @return  Current minor version
     */
    static std::string GetCurrentHDF5_MinorVersion()
    {
      return hdf5_MinorFileVersionsNames[1];
    };

    /**
     * @brief  Set major file version.
     * @details Set major file version.
     */
    void SetMajorFileVersion()
    {
      headerValues[TFileHeaderItems::MAJOR_VERSION] = GetCurrentHDF5_MajorVersion();
    };

    /**
     * @brief   Set minor file version.
     * @details  Set minor file version.
     */
    void SetMinorFileVersion()
    {
      headerValues[TFileHeaderItems::MINOR_VERSION] = GetCurrentHDF5_MinorVersion();
    };

    /// Set major file version in a string.
    TFileVersion GetFileVersion();

    /**
     * @brief   Check major file version.
     * @details Check major file version.
     * @return true if ok
     */
    bool CheckMajorFileVersion()
    {
      return (headerValues[TFileHeaderItems::MAJOR_VERSION] == GetCurrentHDF5_MajorVersion());
    };

    /**
     * @brief   Check minor file version.
     * @details Check minor file version.
     * @return true if ok
     */
    bool CheckMinorFileVersion()
    {
      return (headerValues[TFileHeaderItems::MINOR_VERSION] <= GetCurrentHDF5_MinorVersion());
    };


    /// Get File type.
    THDF5_FileHeader::TFileType GetFileType();
    /// Set file type.
    void SetFileType(const THDF5_FileHeader::TFileType fileType);

    /// Set host name.
    void SetHostName();
    /// Set memory consumption.
    void SetMemoryConsumption(const size_t totalMemory);
    /// Set execution times.
    void SetExecutionTimes(const double totalTime,
                           const double loadTime,
                           const double preProcessingTime,
                           const double simulationTime,
                           const double postprocessingTime);

  /// Get execution times stored in the output file header.
  void GetExecutionTimes(double& totalTime,
                         double& loadTime,
                         double& preProcessingTime,
                         double& simulationTime,
                         double& postprocessingTime);
  /// Set number of cores.
  void SetNumberOfCores();

private:
  /// Populate the map with the header items.
  void PopulateHeaderFileMap();

  /// map for the header values.
  std::map<TFileHeaderItems, std::string> headerValues;
  /// map for the header names.
  std::map<TFileHeaderItems, std::string> headerNames;

  ///String representation of different file types.
  static const std::string hdf5_FileTypesNames[];
  /// String representations of Major file versions.
  static const std::string hdf5_MajorFileVersionsNames[];
  /// String representations of Major file versions.
  static const std::string hdf5_MinorFileVersionsNames[];

};// THDF5_FileHeader
//--------------------------------------------------------------------------------------------------

#endif	/* THDF5_FILE_H */