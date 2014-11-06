/**
 * @file        HDF5_File.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the HDF5 related classes.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        27 July     2012, 14:14 (created) \n
 *              04 November 2014, 14:14 (revised)
 *
 *
 *
 * @section HDF HDF5 File Structure
 *
 * The C++ code has been designed as a standalone application which is not dependent
 * on MATLAB libraries or a MEX interface. This is of particular importance when
 * using servers and supercomputers without MATLAB support. For this reason, simulation
 * data must be transferred between the C++ code and MATLAB using external input
 * and output files. These files are stored using the Hierarchical Data Format
 * HDF5 (http://www.hdfgroup.org/HDF5/). This is a data model, library, and file
 * format for storing and managing data. It supports a variety of datatypes, and
 * is designed for flexible and efficient I/O and for high volume and complex data.
 * The HDF5 technology suite includes tools and applications for managing,
 * manipulating, viewing, and analysing data in the HDF5 format.
 *
 *
 *
 * Each HDF5 file is a container for storing a variety of scientific data and is composed
 * of two primary types of objects: groups and datasets. A HDF5 group is a structure
 * containing zero or more HDF5 objects, together with supporting metadata. A HDF5
 * group can be seen as a disk folder. A HDF5 dataset is a multidimensional array of
 * data elements, together with supporting metadata. A HDF5 dataset can be seen as a
 * disk file. Any HDF5 group or dataset may also have an associated attribute list. A
 * HDF5 attribute is a user-defined HDF5 structure that provides extra information about a
 * HDF5 object. More information can be obtained from the HDF5 documentation (http://www.hdfgroup.org/HDF5/doc/index.html).
 *
 *
 * kspaceFirstOrder3D-OMP v1.1 introduces a new version of the HDF5 input and
 * output file format. The code is happy to work with both versions (1.0 and 1.1), however when
 * working with an input file of version 1.0, some features are not supported,
 * namely the cuboid sensor mask, and u_non_staggered_raw.
 * When running from within the actual K-Wave Toolbox, the files will always be generated
 * in version 1.1
 *
 * The HDF5 input file for the C++ simulation code contains a file header with
 * brief description of the simulation stored in string attributes,
 * and the root group `/' which stores all the simulation properties in the form
 * of 3D datasets (a complete list of input datasets is given bellow).
 * The HDF5 checkpoint file contains the same file header as the input file and
 * the root group `/' with a few datasets keeping the actual simulation state
 * The HDF5 output file contains a file header with the simulation description
 * as well as performance statistics, such as the simulation time and memory
 * consumption, stored in string attributes.
 * The results of the simulation are stored in the root group `/' in the form of 3D datasets.
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
 * The input and checkpoint file stores all quantities as three dimensional datasets with
 * dimension sizes designed by <tt>(Nx, Ny, Nz)</tt>. In order to support scalars
 * and 1D and 2D arrays, the unused dimensions are set to 1.
 * For example, scalar variables are stored with a dimension size of <tt>(1,1,1)</tt>,
 * 1D vectors oriented in y-direction are stored with a dimension  size of <tt>(1, Ny, 1)</tt>,
 * and so on. If the dataset stores a complex variable, the real and imaginary parts
 * are stored in an interleaved layout and the lowest used dimension size is
 * doubled (i.e., Nx for a 3D matrix, Ny for a 1D vector oriented in the y-direction). The
 * datasets are physically stored in row-major order (in contrast to column-major order used
 * by MATLAB) using either the <tt>`H5T_IEEE_F32LE'</tt> data type for floating point datasets or
 * <tt>`H5T_STD_U64LE'</tt> for integer based datasets.
 * All the datasets are store under the root group.
 *
 *
 * The output file of version 1.0 could only store recorded quantities as 3D datasets
 * under the root group. However, with version 1.1 and the new cuboid corner sensor
 * mask, the sampled quantities may be laid out as 4D quantities stored under specific
 * groups. The dimensions are always <tt>(Nx, Ny, Nz, Nt)</tt>, every sampled cuboid
 * is stored as a distinct dataset and the datasets are grouped under a group named
 * by the quantity stored. This makes the file clearly readable and easy to parse.
 *
 *
 * In order to enable compression and more efficient data processing, big datasets
 * are not stored as monolithic blocks
 * but broken into chunks that may be compressed by the ZIP library  and stored
 * separately. The chunk size is defined as follows:
 *
 * \li <tt> (1M elements, 1, 1) </tt> in the case of 1D variables - index sensor mask (8MB blocks).
 * \li <tt> (Nx, Ny, 1)         </tt> in the case of 3D variables (one 2D slab).
 * \li <tt> (Nx, Ny, Nz, 1)     </tt> in the case of 4D variables (one time step).
 * \li <tt> (N_sensor_points, 1, 1) </tt> in the case of the output time series (one time step of the simulation).
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
  x_shift_neg_r                 (Nx/2+1, 1, 1)  float          complex          File version 1.1
  y_shift_neg_r                 (1, Ny/2+1, 1)  float          complex          File version 1.1
  z_shift_neg_r                 (1, 1, Nz/2+1)  float          complex          File version 1.1
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
  sensor_mask_type              (1, 1, 1)       long           real             File version 1.1 (0 = index, 1 = corners)
  sensor_mask_index             (Nsens, 1, 1)   long           real             File version 1.0 always, File version 1.1 if sensor_mask_type == 0
  sensor_mask_corners           (Ncubes, 6, 1)  long           real             File version 1.1, if sensor_mask_type == 1
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

  5.2 Pressure Source Terms (defined if p_source_flag == 1))
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
  2. Simulation state
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
  sensor_mask_type              (1, 1, 1)       long           real             File version 1.1 and --copy_sensor_mask
  sensor_mask_index             (Nsens, 1, 1)   long           real             File version 1.1 and if sensor_mask_type == 0
  sensor_mask_corners           (Ncubes, 6, 1)  long           real             File version 1.1 and if sensor_mask_type == 1
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
  5b. Simulation Results: if sensor_mask_type == 1 (corners) and File version == 1.1
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

#ifndef THDF5_FILE_H
#define	THDF5_FILE_H


#include <hdf5.h>
#include <hdf5_hl.h>
#include <string>
#include <map>

// Linux build
#ifdef __linux__
  #include <unistd.h>
#endif
// Windows build
#ifdef _WIN64
  #include <io.h>
#endif

#include "../Utils/DimensionSizes.h"



    using namespace std;

    // Class with File header
    class THDF5_FileHeader;

    /**
     * @class THDF5_File
     * @brief Class wrapping the HDF5 routines
     */
    class THDF5_File
{
    public:

        /**
         * @enum THDF5_MatrixDataType
         * @brief HDF5 matrix data type
         */
        enum THDF5_MatrixDataType   {hdf5_mdt_float = 0, hdf5_mdt_long    = 1};

        /**
         * @enum  THDF5_MatrixDomainType
         * @brief HDF5 Matrix domain type
         */
        enum THDF5_MatrixDomainType {hdf5_mdt_real  = 0, hdf5_mdt_complex = 1};


        /// Constructor
        THDF5_File();

        //----------------------- Basic file operations --------------------------//
        /// Create the file
        void Create(const char * FileName,
                unsigned int Flags = H5F_ACC_TRUNC );
        /// Open the file
        void Open (const char * FileName,
                unsigned int Flags  = H5F_ACC_RDONLY);
        /**
         * Is the file opened?
         * @return true if the file is opened
         */
        bool IsOpened() const
        {
            return HDF5_FileId != H5I_BADID;
        };

    /**
     * @brief Does the file exist? static method.
     * @details Check if the file exist
     * @param [in] FileName
     * @return true if the file exists
     */
        static bool IsHDF5(const char * FileName)
        {
          #ifdef __linux__
            return (access(FileName, F_OK) == 0);
          #endif

          #ifdef _WIN64
             return (_access_s(FileName, 0) == 0 );
          #endif
        };

        /// Close file
        void Close();

        //----------------------- Group manipulators -----------------------------//
        /// Create a HDF5 group at a specified place in the file tree
        hid_t CreateGroup(const hid_t ParentGroup,
                const char * GroupName);
        /// Open a HDF5 group at a specified place in the file tree
        hid_t OpenGroup  (const hid_t ParentGroup,
                const char * GroupName);
        /// Close group
        void CloseGroup(const hid_t Group);
        /**
         * Get handle to the root group
         * @return  - handle to the root group
         */
        hid_t GetRootGroup() const
        {
            return HDF5_FileId;
        };


        //---------------------- Dataset manipulators ----------------------------//
        /// Open the HDF5 dataset at a specified place in the file tree.
        hid_t OpenDataset  (const hid_t ParentGroup,
                const char * DatasetName);
        /// Create the HDF5 dataset at a specified place in the file tree (3D/4D).
        hid_t CreateFloatDataset(const hid_t ParentGroup,
                const char * DatasetName,
                const TDimensionSizes & DimensionSizes,
                const TDimensionSizes & ChunkSizes,
                const int CompressionLevel);

        /// Create the HDF5 dataset at a specified place in the file tree (3D only).
        hid_t CreateLongDataset(const hid_t ParentGroup,
                const char * DatasetName,
                const TDimensionSizes & DimensionSizes,
                const TDimensionSizes & ChunkSizes,
                const int CompressionLevel);

        /// Close the HDF5 dataset
        void  CloseDataset (const hid_t HDF5_Dataset_id);


        //------------------ Dataset Read/Write operations -----------------------//
        /// Write a hyper-slab into the dataset - float dataset
        void WriteHyperSlab(const hid_t HDF5_Dataset_id,
                const TDimensionSizes & Position,
                const TDimensionSizes & Size,
                const float * Data);
        /// Write a hyper-slab into the dataset - long dataset
        void WriteHyperSlab(const hid_t HDF5_Dataset_id,
                const TDimensionSizes & Position,
                const TDimensionSizes & Size,
                const size_t * Data);

        /// Write a cuboid selected inside MatrixData into a Hyperslab
        void WriteCuboidToHyperSlab(const hid_t HDF5_Dataset_id,
                const TDimensionSizes & HyperslabPosition,
                const TDimensionSizes & CuboidPosition,
                const TDimensionSizes & CuboidSize,
                const TDimensionSizes & MatrixDimensions,
                const float * MatrixData);

        /// Write sensor data selected by the sensor mask - Occasionally very slow, do not use!
        void WriteSensorByMaskToHyperSlab(const hid_t HDF5_Dataset_id,
                const TDimensionSizes & HyperslabPosition,
                const size_t IndexSensorSize,
                const size_t * IndexSensorData,
                const TDimensionSizes & MatrixDimensions,
                const float * MatrixData);

        /// Write the scalar value under a specified group - float value
        void WriteScalarValue(const hid_t ParentGroup,
                const char * DatasetName,
                const float Value);
        /// Write the scalar value under a specified group - long value
        void WriteScalarValue(const hid_t ParentGroup,
                const char * DatasetName,
                const size_t  Value);

        /// Read the scalar value under a specified group - float value.
        void ReadScalarValue(const hid_t ParentGroup,
                             const char * DatasetName,
                             float      & Value);
        /// Read the scalar value under a specified group - index value.
        void ReadScalarValue(const hid_t ParentGroup,
                             const char * DatasetName,
                             size_t & Value);


        /// Read data from the dataset under a specified group - float dataset
        void ReadCompleteDataset(const hid_t ParentGroup,
                const char * DatasetName,
                const TDimensionSizes & DimensionSizes,
                float * Data);
        /// Read data from the dataset under a specified group - long dataset
        void ReadCompleteDataset(const hid_t ParentGroup,
                const char * DatasetName,
                const TDimensionSizes & DimensionSizes,
                size_t * Data);


        //------------------- Attributes Read/Write operations -------------------//

        /// Get dimension sizes of the dataset  under a specified group
        TDimensionSizes GetDatasetDimensionSizes(const hid_t ParentGroup,
                const char * DatasetName);

        /// Get number of dimensions of the dataset  under a specified group
        size_t GetDatasetNumberOfDimensions(const hid_t ParentGroup,
                const char * DatasetName);

        /// Get dataset element count under a specified group
        size_t GetDatasetElementCount(const hid_t ParentGroup,
                const char * DatasetName);


        /// Write matrix data type into the dataset under a specified group
        void WriteMatrixDataType (const hid_t ParentGroup,
                const char * DatasetName,
                const THDF5_MatrixDataType   & MatrixDataType);
        /// Write matrix domain type into the dataset under the root group
        void WriteMatrixDomainType(const hid_t ParentGroup,
                const char * DatasetName,
                const THDF5_MatrixDomainType & MatrixDomainType);

        /// Read matrix data type from the dataset c
        THDF5_File::THDF5_MatrixDataType   ReadMatrixDataType(const hid_t ParentGroup,
                const char * DatasetName);
        /// Read matrix domain type from the dataset under a specified group
        THDF5_File::THDF5_MatrixDomainType ReadMatrixDomainType(const hid_t ParentGroup,
                const char * DatasetName);


        /// Write string attribute into the dataset under the root group
        void   WriteStringAttribute (const hid_t ParentGroup,
                const char * DatasetName,
                const char * AttributeName,
                const string &  Value);
        /// Read string attribute from the dataset under the root group
        string ReadStringAttribute  (const hid_t ParentGroup,
                const char * DatasetName,
                const char * AttributeName);

        /// Destructor
        virtual ~THDF5_File();

    protected:

        /// Copy constructor is not allowed for public
        THDF5_File(const THDF5_File& src);
        /// Operator = is not allowed for public
        THDF5_File & operator = (const THDF5_File& src);


    private:
        /// String representation of the Domain type in the HDF5 file
        static const char * HDF5_MatrixDomainTypeName;
        /// String representation of the Data type in the HDF5 file
        static const char * HDF5_MatrixDataTypeName;

        /// String representation of different domain types
        static const string HDF5_MatrixDomainTypeNames[];
        /// String representation of different data types
        static const string HDF5_MatrixDataTypeNames[];

        /// HDF file handle
        hid_t  HDF5_FileId;
        /// File name
        string FileName;

}; // THDF5_File
//------------------------------------------------------------------------------


/**
 * @class THDF5_FileHeader
 * @brief Class for HDF5 header
 *
 */
class THDF5_FileHeader
{
    public:

        /**
         * @enum THDF5_FileHeaderItems
         * @brief List of all header items
         */
        enum THDF5_FileHeaderItems { hdf5_fhi_created_by =  0,
            hdf5_fhi_creation_date                       =  1,
            hdf5_fhi_file_description                    =  2,
            hdf5_fhi_major_version                       =  3,
            hdf5_fhi_minor_version                       =  4,
            hdf5_fhi_file_type                           =  5,
            hdf5_fhi_host_name                           =  6,
            hdf5_fhi_total_memory_consumption            =  7,
            hdf5_fhi_peak_core_memory_consumption        =  8,
            hdf5_fhi_total_execution_time                =  9,
            hdf5_fhi_data_load_time                      = 10,
            hdf5_fhi_preprocessing_time                  = 11,
            hdf5_fhi_simulation_time                     = 12,
            hdf5_fhi_postprocessing_time                 = 13,
            hdf5_fhi_number_of_cores                     = 14
        };

        /**
         * @enum  THDF5_FileType
         * @brief HDF5 file type
         */
        enum THDF5_FileType         {hdf5_ft_input  = 0,
            hdf5_ft_output = 1,
            hdf5_ft_checkpoint = 2,
            hdf5_ft_unknown = 3};

        /**
         * @enum  THDF5_FileVersion
         * @brief HDF5 file version
         */
        enum THDF5_FileVersion       {hdf5_fv_10 = 0,
            hdf5_fv_11 = 1,
            hdf5_fv_unknown = 2};


        /// Constructor
        THDF5_FileHeader();
        /// Copy constructor
        THDF5_FileHeader(const THDF5_FileHeader & other);
        /// Destructor
        ~THDF5_FileHeader();

        /// Read header from the input file
        void ReadHeaderFromInputFile(THDF5_File & InputFile);
        /// Read Header from output file (necessary for checkpointing)
        void ReadHeaderFromOutputFile(THDF5_File & OutputFile);
        /// Write header to the output file
        void WriteHeaderToOutputFile(THDF5_File & OutputFile);

        /**
         * Set code name
         * @param CodeName - code version
         */
        void SetCodeName(string CodeName)
        {
            HDF5_FileHeaderValues[hdf5_fhi_created_by] = CodeName;
        };

        /// Set creation time
        void SetActualCreationTime();

        /**
         * Get string version of current Major version
         * @return  current version
         */
        static string GetCurrentHDF5_MajorVersion()
        {
            return HDF5_MajorFileVersionsNames[0];
        };

        /**
         * Get string version of current Minor version
         * @return  current minor version
         */
        static string GetCurrentHDF5_MinorVersion()
        {
            return HDF5_MinorFileVersionsNames[1];
        };

        /// Set major file version
        void SetMajorFileVersion()
        {
            HDF5_FileHeaderValues[hdf5_fhi_major_version] = GetCurrentHDF5_MajorVersion();
        };
        /// Set minor file version
        void SetMinorFileVersion()
        {
            HDF5_FileHeaderValues[hdf5_fhi_minor_version] = GetCurrentHDF5_MinorVersion();
        };

        /// Set major file version in a string
        THDF5_FileVersion GetFileVersion();

        /**
         * Check major file version
         * @return true if ok
         */
        bool CheckMajorFileVersion()
        {
            return (HDF5_FileHeaderValues[hdf5_fhi_major_version] == GetCurrentHDF5_MajorVersion());
        };
        /**
         * Check minor file version
         * @return true if ok
         */
        bool CheckMinorFileVersion()
        {
            return (HDF5_FileHeaderValues[hdf5_fhi_minor_version] <= GetCurrentHDF5_MinorVersion());
        };

        /// Get File type
        THDF5_FileHeader::THDF5_FileType GetFileType();
        /// Set file type
        void SetFileType(const THDF5_FileHeader::THDF5_FileType FileType);

        /// Set host name
        void SetHostName();
        /// Set memory consumption
        void SetMemoryConsumption(size_t TotalMemory);
        /// Set execution times
        void SetExecutionTimes(const double TotalTime,
                const double LoadTime,
                const double PreProcessingTime,
                const double SimulationTime,
                const double PostprocessingTime);

        /// Get execution times stored in the output file header
        void GetExecutionTimes(double& TotalTime,
                double& LoadTime,
                double& PreProcessingTime,
                double& SimulationTime,
                double& PostprocessingTime);
        /// Set number of cores
        void SetNumberOfCores();

    private:
        /// Populate the map with the header items
        void PopulateHeaderFileMap();

        /// map for the header values
        map<THDF5_FileHeaderItems, string> HDF5_FileHeaderValues;
        /// map for the header names
        map<THDF5_FileHeaderItems, string> HDF5_FileHeaderNames;

        ///String representation of different file types
        static const string HDF5_FileTypesNames[];
        /// String representations of Major file versions
        static const string HDF5_MajorFileVersionsNames[];
        /// String representations of Major file versions
        static const string HDF5_MinorFileVersionsNames[];

};// THDF5_FileHeader
//------------------------------------------------------------------------------

#endif	/* THDF5_FILE_H */

