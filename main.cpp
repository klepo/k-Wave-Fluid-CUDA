/**
 * @file        main.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The main file for the kspaceFirstOrder3D-CUDA.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 July     2012, 10:57 (created) \n
 *              02 August   2016, 13:50 (revised)
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
 *
 *
 * @mainpage kspaceFirstOrder3D-CUDA
 *
 * @section Overview 1 Overview
 *
 * k-Wave is an open source MATLAB toolbox designed for the time-domain simulation of propagating
 * acoustic waves in 1D, 2D, or 3D. The toolbox has a wide range of functionality, but at its heart
 * is an advanced numerical model that can account for both linear or nonlinear wave propagation,
 * an arbitrary distribution of weakly heterogeneous material parameters, and power law acoustic
 * absorption (http://www.k-wave.org).
 *
 * This project is a part of the k-Wave toolbox accelerating 3D simulations using an optimized
 * CUDA/C++ implementation to run small to moderate grid sizes (64x64x64 to 512x512x512). This
 * code uses a single NVIDIA GPU to accelerate the simulations (AMD GPUs are not supported).
 *
 * Compiled binaries of the CUDA/C++ code are available from  (http://www.k-wave.org/download).
 * Both 64-bit Linux (Ubuntu / Debian) and  64-bit Windows  versions are provided. This Doxygen
 * documentation was created based on the Linux version and provides details on the
 * implementation of the CUDA/C++ code.
 *
 *
 *
 * @section Compilation 2 Compilation
 *
 * The source codes of <tt>kpsaceFirstOrder3D-CUDA</tt> are written using the C++11 standard
 * (optional OpenMP 2.0 library), NVIDIA CUDA 7.5 library and HDF5 1.8.x.
 *
 * There are variety of different C++ compilers that can be used to compile the source codes.
 * We recommend using the GNU C++ compiler (gcc/g++) version 4.8/4.9, the Intel C++ compiler
 * version 15.0, or Visual Studio 2013. The version of the compiler  is limited by the CUDA
 * architecture version. The code was tested with CUDA 6.5, 7.0 and 7.5.
 * The codes can be compiled on 64-bit Linux and Windows. 32-bit systems are not supported due to
 * the the memory requirements even for small simulations.
 *
 * Before compiling the code, it is necessary to install CUDA, C++ compiler and the HDF5 library.
 * The GNU compiler is usually part of Linux distributions and distributed as open source.
 * It can be downloaded from http://gcc.gnu.org/ if necessary. The Intel compiler can be downloaded
 * from http://software.intel.com/en-us/intel-composer-xe/. The Intel compiler is only free for
 * non-commercial use.

 * The CUDA library can be downloaded from https://developer.nvidia.com/cuda-downloads.
 * The supported versions are 6.5, 7.0 and 7.5. We hope the code will work also with newer versions.
 *
 * <b> 2.1 The HDF5 library installation procedure </b>

 * 1. Download the 64-bit HDF5 library  package for your platform
 * (http://www.hdfgroup.org/HDF5/release/obtain5.html). Please use version 1.8.x, the version 1.10.x
 * is not compatible with MATLAB yet.
 *
 * 2. Configure the HDF5 distribution. Enable the high-level library and specify
 * an installation folder by typing:
\verbatim ./configure --enable-hl --prefix=folder_to_install \endverbatim
 * 3. Make the HDF library by typing:
\verbatim make \endverbatim
 * 4. Install the HDF library by typing:
\verbatim make install \endverbatim
 *
 *
 * <b> 2.2 Compiling the CUDA/C++ code </b>
 *
 * When the libraries and the compiler have been installed, you are ready to compile the
 * <tt>kspaceFirstOrder3D-CUDA </tt> code. The Makefile only supports code compilation under CUDA/g++
 * compiler, however using different compilers would be analogous (<tt>-ccbin</tt> parameter for
 * the <tt>nvcc </tt>compiler)
 *

 * 1. Download the <tt>kspaceFirstOrder3D-CUDA </tt> source codes.
 *
 * 2. Open the \c Makefile file.
 *    First, set the paths to CUDA and HDF5 libraries, then select how to link the code. Dynamic
 *    lining is preferred since it does not require reloading the CUDA-GPU driver, however the code
 *    will very likely only run on the machine where compiled. The static linking allows you you
 *     create a self-consistent binary with all libraries linked in.
 *
 * 3. Compile the source code by typing:
\verbatim make -j \endverbatim If you want to clean the distribution, type:
\verbatim make clean \endverbatim
 *
 *
 * @section Parameters 3 Command Line Parameters
 * The CUDA/C++ code requires two mandatory parameters and accepts a few optional parameters and
 * flags.
 *
 * The mandatory parameters \c -i and \c -o specify the input and output file. The file names
 * respect the path conventions for particular operating system. If any of the files is not
 * specified, cannot be found or created, an error message is shown and the code terminates.
 *
 * The \c -t parameter sets the number of threads used, which defaults the system maximum. For the
 * CUDA version, the number of threads is not critical, as the CPU only preprocess data. When
 * running on desktops with multiple GPUs, or clusters with per-GPU resource allocations, it may
 * be useful to limit the number of CPU threads.
 *
 * The \c -g parameter allows to explicitly select a GPU for the execution. The CUDA capable GPUs
 * can be listed by the system command \c nvidia-smi. If the parameter is not specified, the code
 * uses the first free GPU. If the GPUs are set in the CUDA DEFAULT mode, the first CUDA device
 * is selected. In order to get the automatic round-robin GPU selection working (to e.g. execute
 * multiple instances of the code on distinct GPUs), please set the GPUs into PROCESS_EXCLUSIVE mode.
 * On clusters with a PBS  scheduler, this is usually done automatically, so no need to change it
 * by user.
 *
 * The \c -r parameter specifies how often information about the simulation progress is printed out
 * to the command line. By default, the CUDA/C++ code prints out the  progress of the simulation,
 * the elapsed time, and the estimated time of completion in intervals corresponding to 5% of
 * the total number of times steps.
 *
 * The \c -c parameter specifies the compression level used by the ZIP library to reduce the size of
 * the output file. The actual compression rate is highly dependent on the shape of the sensor mask
 * and the range of stored quantities and may be computationally expensive. In general, the output
 * data is very hard to compress, and using higher compression levels can greatly increase the
 * time to save data while not having a large impact on the final file size. That's why we decided
 * to disable compression in default settings.
 *
 * The \c <tt>\--benchmark</tt> parameter enables the total length of simulation (i.e., the number of
 * time steps) to be overridden by setting a new number of time  steps to simulate. This is
 * particularly useful for performance evaluation and benchmarking. As the code performance is
 * relatively stable, 50-100 time steps is  usually enough to predict the simulation duration.
 * This parameter can also be used to quickly check the simulation is set up correctly.
 *
 * The \c <tt>\--verbose</tt> parameter enables to select between three levels of verbosity. For
 * routine simulations, the verbose level of 0 (the default one) is usually sufficient. For more
 * information about the simulation, checking the parameters of the simulation, code version,
 * GPU used, file paths, and debugging running scripts, verbose levels 1 and 2 may be very useful.
 *
 * The \c -h and <tt>\--help</tt> parameters print all the parameters of the C++ code. The
 * <tt>\--version </tt>parameter reports detail information about the code useful for  debugging and
 * bug reports. It prints out the internal version, the build date and time, the git hash allowing
 * us to track the version of the source code, the operating system, the compiler name and version
 * and the instruction set used.
 *
 * For jobs that are expected to run for a very long time, it may be useful to  checkpoint and
 * restart the execution. One motivation is the wall clock limit  per task on clusters where jobs
 * must fit within a given time span (e.g. 24 hours). The second motivation is a level of
 * fault-tolerance, where you can back up the state of the simulation after a predefined period.
 * To enable checkpoint-restart, the user is asked to specify a file to store the actual state of
 * the simulation by  <tt>\--checkpoint_file</tt> and the period in seconds after which the
 * simulation will be interrupted by <tt>\--checkpoint_interval</tt>.  When running on a cluster,
 * please allocate enough time for the checkpoint procedure  that can take a non-negligible amount
 * of time (7 matrices have to be stored in  the checkpoint file and all aggregated quantities are
 * flushed into the output file). Please note, that the checkpoint file name and path is not checked
 * at the beginning of the simulation, but at the time the code starts checkpointing. Thus make sure
 * the file path was correctly specified (otherwise you'll find out after the first leg that the
 * simulation crashed). The rationale behind this is that to keep as high level of fault tolerance
 * as possible, the checkpoint file should be touched even when really necessary.
 *
 * When controlling a multi-leg simulation by a script loop, the parameters of the code remains the
 * same in all legs. The first leg of the simulation creates a checkpoint  file while the last one
 * deletes it. If the checkpoint file is not found the simulation starts from the beginning. In
 * order to find out how many steps have been finished, please open the output file and read
 * the variable <tt>t_index</tt> (e.g. by the h5dump command).
 *
 *
 * The remaining flags specify the output quantities to be recorded during the  simulation and
 * stored on disk analogous to  the sensor.record input. If the \c -p or <tt>\--p\_raw</tt> flags
 * are set (these are equivalent), a time series of  the acoustic pressure at the grid points
 * specified by  the sensor mask is recorded. If the <tt>\--p_rms</tt>, <tt>\--p_max</tt>,
 * <tt>\--p_min</tt> flags  are set, the root mean square and/or maximum and/or minimum values of
 * the pressure at the grid points specified by  the sensor mask are recorded. If the
 * <tt>\--p_final</tt> flag is set, the values for the entire acoustic pressure field in the final
 * time step of the simulation is stored (this will always include the PML, regardless of  the
 * setting for <tt> `PMLInside'</tt>).
 * The flags <tt>\--p_max_all</tt> and <tt>\--p_min_all</tt> allow to calculate the maximum and
 * minimum values over the entire acoustic pressure field, regardless on the shape of the sensor
 * mask. Flags to record the acoustic particle velocity are defined in an analogous fashion. For
 * proper calculation of acoustic intensity, the particle velocity has to be shifted onto the same
 * grid as the acoustic  pressure. This can be done by setting <tt>\--u_non_staggered_raw</tt> flag,
 * that first shifts the  particle velocity and then samples the grid points specified by the sensor
 * mask. Since the  shift operation requires additional FFTs, the impact on the simulation time may
 * be significant.
 *
 * Any combination of <tt>p</tt> and <tt>u</tt> flags is admissible. If no output flag is set,
 * a time-series for the acoustic pressure is recorded. If it is not necessary to collect the output
 * quantities over the entire simulation, the starting time step when the collection begins can
 * be specified using the -s parameter.  Note, the index for the first time step is 1 (this follows
 * the MATLAB indexing convention).
 *
 * The <tt>\--copy_sensor_mask</tt> will copy the sensor from the input file to the output  one at the end
 * of the simulation. This helps in post-processing and visualisation of the outputs.
 *
\verbatim
┌───────────────────────────────────────────────────────────────┐
│                 kspaceFirstOrder3D-CUDA v1.1                  │
├───────────────────────────────────────────────────────────────┤
│                             Usage                             │
├───────────────────────────────────────────────────────────────┤
│                     Mandatory parameters                      │
├───────────────────────────────────────────────────────────────┤
│ -i <file_name>                │ HDF5 input file               │
│ -o <file_name>                │ HDF5 output file              │
├───────────────────────────────┴───────────────────────────────┤
│                      Optional parameters                      │
├───────────────────────────────┬───────────────────────────────┤
│ -t <num_threads>              │ Number of CPU threads         │
│                               │  (default =  4)               │
│ -g <device_number>            │ GPU device to run on          │
│                               │   (default = the first free)  │
│ -r <interval_in_%>            │ Progress print interval       │
│                               │   (default =  5%)             │
│ -c <compression_level>        │ Compression level <0,9>       │
│                               │   (default = 0)               │
│ --benchmark <time_steps>      │ Run only a specified number   │
│                               │   of time steps               │
│ --verbose <level>             │ Level of verbosity <0,2>      │
│                               │   0 - basic, 1 - advanced,    │
│                               │   2 - full                    │
│                               │   (default = basic)           │
│ -h, --help                    │ Print help                    │
│ --version                     │ Print version and build info  │
├───────────────────────────────┼───────────────────────────────┤
│ --checkpoint_file <file_name> │ HDF5 Checkpoint file          │
│ --checkpoint_interval <sec>   │ Checkpoint after a given      │
│                               │   number of seconds           │
├───────────────────────────────┴───────────────────────────────┤
│                          Output flags                         │
├───────────────────────────────┬───────────────────────────────┤
│ -p                            │ Store acoustic pressure       │
│                               │   (default output flag)       │
│                               │   (the same as --p_raw)       │
│ --p_raw                       │ Store raw time series of p    │
│ --p_rms                       │ Store rms of p                │
│ --p_max                       │ Store max of p                │
│ --p_min                       │ Store min of p                │
│ --p_max_all                   │ Store max of p (whole domain) │
│ --p_min_all                   │ Store min of p (whole domain) │
│ --p_final                     │ Store final pressure field    │
├───────────────────────────────┼───────────────────────────────┤
│ -u                            │ Store ux, uy, uz              │
│                               │    (the same as --u_raw)      │
│ --u_raw                       │ Store raw time series of      │
│                               │    ux, uy, uz                 │
│ --u_non_staggered_raw         │ Store non-staggered raw time  │
│                               │   series of ux, uy, uz        │
│ --u_rms                       │ Store rms of ux, uy, uz       │
│ --u_max                       │ Store max of ux, uy, uz       │
│ --u_min                       │ Store min of ux, uy, uz       │
│ --u_max_all                   │ Store max of ux, uy, uz       │
│                               │   (whole domain)              │
│ --u_min_all                   │ Store min of ux, uy, uz       │
│                               │   (whole domain)              │
│ --u_final                     │ Store final acoustic velocity │
├───────────────────────────────┼───────────────────────────────┤
│ -s <time_step>                │ When data collection begins   │
│                               │   (default = 1)               │
└───────────────────────────────┴───────────────────────────────┘
\endverbatim
 *
 *
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
 *  * The HDF5 output file contains a file header with the simulation description as well as
 * performance statistics, such as the simulation  time and memory consumption, stored in string attributes.

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
*/

#include <cstdlib>
#include <iostream>
#include <exception>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <KSpaceSolver/KSpaceFirstOrder3DSolver.h>
#include <Logger/Logger.h>

#include <cstdio>
#include <iostream>

using std::string;

/**
 * The main function of the kspaceFirstOrder3D-CUDA
 * @param [in] argc
 * @param [in] argv
 * @return
 */
int main(int argc, char** argv)
{
  // Create k-Space solver
  TKSpaceFirstOrder3DSolver KSpaceSolver;

  // print header
  TLogger::Log(TLogger::BASIC, OUT_FMT_FIRST_SEPARATOR);
  TLogger::Log(TLogger::BASIC, OUT_FMT_CODE_NAME, KSpaceSolver.GetCodeName().c_str());
  TLogger::Log(TLogger::BASIC, OUT_FMT_SEPARATOR);

  // Create parameters and parse command line
  TParameters& params = TParameters::GetInstance();

  //-------------- Init simulation ----------------//
  try
  {
    // Initialise Parameters by parsing the command line and reading input file scalars
    params.Init(argc, argv);
    // Select GPU
    params.SelectDevice();

    // When we know the GPU, we can print out the code version
    if (params.IsVersion())
    {
      KSpaceSolver.PrintFullNameCodeAndLicense();
      return EXIT_SUCCESS;
    }
  }
  catch (const std::exception &e)
  {
     TLogger::Log(TLogger::BASIC, OUT_FMT_FAILED);
    // must be repeated in case the GPU we want to printout the code version
    // and all GPUs are busy
    if (params.IsVersion())
    {
      KSpaceSolver.PrintFullNameCodeAndLicense();
    }

    if (!params.IsVersion())
    {
      TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);
    }
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),ERR_FMT_PATH_DELIMITERS, 9).c_str());
  }

  // set number of threads and bind them to cores
  #ifdef _OPENMP
    omp_set_num_threads(params.GetNumberOfThreads());
  #endif

  // Print simulation setup
  params.PrintSimulatoinSetup();

  TLogger::Log(TLogger::BASIC, OUT_FMT_INIT_HEADER);

  //-------------- Allocate memory----------------//
  try
  {
    KSpaceSolver.AllocateMemory();
  }
  catch (const std::bad_alloc& e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_FAILED);
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_OUT_OF_MEMORY," ", 9).c_str());
  }
  catch (const std::exception& e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_FAILED);
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),
                                                       ERR_FMT_PATH_DELIMITERS,
                                                       13).c_str());
  }

  //-------------- Load input data ----------------//
  try
  {
    KSpaceSolver.LoadInputData();
  }
  catch (const std::ios::failure& e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_FAILED);
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),
                                                       ERR_FMT_PATH_DELIMITERS,
                                                       9).c_str());
  }
  catch (const std::exception& e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_FAILED);
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);

    const string ErrorMessage = string(ERR_FMT_UNKNOWN_ERROR) + e.what();
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(ErrorMessage,
                                                       ERR_FMT_PATH_DELIMITERS,
                                                       13).c_str());
  }

  TLogger::Log(TLogger::BASIC,
               OUT_FMT_ELAPSED_TIME,
               KSpaceSolver.GetDataLoadTime());


  if (params.Get_t_index() > 0)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_SEPARATOR);
    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_RECOVER_FROM,
                 params.Get_t_index());
  }


  // start computation
  TLogger::Log(TLogger::BASIC, OUT_FMT_SEPARATOR);
  // exception are caught inside due to different log formats
  KSpaceSolver.Compute();



  // summary
  TLogger::Log(TLogger::BASIC, OUT_FMT_SUMMARY_HEADER);

  TLogger::Log(TLogger::BASIC,
               OUT_FMT_HOST_MEMORY_USAGE,
               KSpaceSolver.GetHostMemoryUsageInMB());

  TLogger::Log(TLogger::BASIC,
               OUT_FMT_DEVICE_MEMORY_USAGE,
               KSpaceSolver.GetDeviceMemoryUsageInMB());

TLogger::Log(TLogger::BASIC, OUT_FMT_SEPARATOR);

// Elapsed Time time
if (KSpaceSolver.GetCumulatedTotalTime() != KSpaceSolver.GetTotalTime())
  {
    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_LEG_EXECUTION_TIME,
                 KSpaceSolver.GetTotalTime());

  }
  TLogger::Log(TLogger::BASIC,
               OUT_FMT_TOTAL_EXECUTION_TIME,
               KSpaceSolver.GetCumulatedTotalTime());


  // end of computation
  TLogger::Log(TLogger::BASIC, OUT_FMT_END_OF_SIMULATION);

  return EXIT_SUCCESS;
}// end of main
//------------------------------------------------------------------------------
