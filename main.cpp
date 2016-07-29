/**
 * @file        main.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The main file for the kspaceFirstOrder3D-CUDA.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        11 July     2012, 10:57 (created) \n
 *              14 July     2016, 13:50 (revised)
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
 *
 *
 *
 * @mainpage kspaceFirstOrder3D-CUDA
 *
 * @section Overview 1 Overview
 *
 * k-Wave is an open source MATLAB toolbox designed for the time-domain
 * simulation of propagating acoustic waves in 1D, 2D, or 3D. The toolbox has a
 * wide range of functionality, but at its heart is an advanced numerical model
 * that can account for both linear or nonlinear wave propagation, an arbitrary
 * distribution of weakly heterogeneous material parameters, and power law
 * acoustic absorption (http://www.k-wave.org).
 *
 * This project is a part of the k-Wave toolbox accelerating 3D simulations
 * using an optimized C++ implementation to run moderate to big grid sizes.
 * Compiled binaries of the C++ code for x86 architectures are available from
 * (http://www.k-wave.org/download). Both 64-bit Linux (Ubuntu / Debian) and
 * 64-bit Windows versions are provided. This Doxygen documentation was created
 * based on the Linux version and provides details on the implementation of the
 * C++ code.
 *
 *
 *
 * @section Compilation 2 Compilation
 *
 *
 * The source codes of <tt>kpsaceFirstOrder3D-OMP</tt> are written using the
 * C++03 standard and the OpenMP 2.0 library.  There are variety of different
 * C++ compilers that can be used to compile the source codes. We recommend
 * using either the GNU C++ compiler (gcc/g++) version 4.3 and higher, or the
 * Intel C++ compiler version 11.0 and higher. The codes can be compiled on
 * 64-bit Linux (Mac) and Windows.  32-bit systems are not supported. This
 * section describes the compilation procedure using GNU and Intel compilers on
 * Linux.  (The Windows users are encouraged to download the Visual Studio 2010
 * project and compile it using Intel Compiler from within Visual Studio.)
 *
 * Before compiling the code, it is necessary to install a C++ compiler and
 * several libraries.  The GNU compiler is usually part of Linux distributions
 * and distributed as open source.  It can be downloaded from
 * http://gcc.gnu.org/ if necessary. The Intel compiler can be downloaded from
 * http://software.intel.com/en-us/intel-composer-xe/. This package also
 * includes the Intel MKL (Math Kernel Library) library that contains FFT. The
 * Intel compiler is only free for non-commercial use.
 *
 * The code also relies on several libraries that is to be installed before
 * compiling:
 *
 * \li HDF5 library - Mandatory I/O library, version 1.8 or higher
 * (http://www.hdfgroup.org/HDF5/).  \li FFTW library - Optional library for
 * FFT, version 3.0 or higher (http://www.fftw.org/).  \li MKLlibrary -
 * Optional library for FFT, version 11.0 or higher
 * (http://software.intel.com/en-us/intel-composer-xe/).

 *
 * Although it is possible to use any combination of the FFT library and the
 * compiler, the best performance is observed when using GNU compiler and FFTW,
 * or Intel Compiler and Intel MKL.

 * <b> 2.1 The HDF5 library installation procedure </b>

 * 1. Download the 64-bit HDF5 library  package for your platform
 * (http://www.hdfgroup.org/HDF5/release/obtain5.html).
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
 *
 * <b> 2.2 The FFTW library installation procedure </b>
 *
 * 1. Download the FFTW library  package for your platform
 * (http://www.fftw.org/download.html).
 *
 * 2. Configure the FFTW distribution. Enable OpenMP support, SSE instruction
 * set, single precision data type, and specify an installation folder:
\verbatim ./configure --enable-single --enable-sse --enable-openmp
--prefix=folder_to_install \endverbatim if you intend to use the FFTW library
(and the C++ code) only on a selected machine and want to get the best possible
performance, you may also add processor specific optimisations and AVX
instructions set. Note, the compiled binary code is not likely to be portable
on different CPUs (e.g. even from Intel Sandy Bridge to Intel Nehalem).
\verbatim ./configure --enable-single --enable-avx --enable-openmp
--with-gcc-arch=<arch> --prefix=folder_to_install \endverbatim
 * More information about the installation and customization can be found at
 * http://www.fftw.org/fftw3_doc/Installation-and-Customization.htm.
 *
 * 3. Make the FFTW library by typing:
\verbatim make \endverbatim
 * 4. Install the FFTW library by typing:
\verbatim make install \endverbatim

 * <b> 2.3 The Intel Compiler and MKL installation procedure </b>
 *
 *
 * 1. Download  the Intel Composer XE package for your platform
 * (http://software.intel.com/en-us/intel-compilers).
 *
 * 2. Run the installation script and follow the procedure by typing:
\verbatim ./install.sh \endverbatim
 *
 *
 *
 * <b> 2.4 Compiling the C++ code </b>
 *
 * When the libraries and the compiler have been installed, you are ready to
 * compile the <tt>kspaceFirstOrder3D-OMP </tt> code.
 *

 * 1. Download the <tt>kspaceFirstOrder3D-OMP </tt> souce codes.
 *
 * 2. Open the \c Makefile file.  The Makefile supports code compilation under
 * GNU compiler and FFTW, or Intel compiler with MKL. Uncomment the desired
 * compiler by removing the character of `<tt>#</tt>'.
\verbatim #COMPILER = GNU #COMPILER = Intel \endverbatim *3. Select how to link
the libraries. Static linking is preferred, however, on some systems (HPC
        clusters) it may be better to use dynamic linking and use the system
specific library at runtime.  \verbatim #LINKING = STATIC #LINKING = DYNAMIC
\endverbatim
 *
 * 4. Set installation paths of the libraries (an example is shown bellow).
\verbatim FFT_DIR=/usr/local MKL_DIR=/opt/intel/composer_xe_2011_sp1/mkl
HDF5_DIR=/usr/local/hdf5-1.8.9 \endverbatim
 * 5. The code will be built to work on all CPUs implementing the Streaming SSE2
 * instruction set (Intel Pentium IV, AMD Athlon XP 64 and newer). Higher
 * performance may be obtained by providing more information about the target
 * CPU (SSE 4.2 instructions, AVX instruction, architecture optimization). For
 * more detail, check on the compiler documentation.
 *
 * 6. Compile the source code by typing:
\verbatim make \endverbatim If you want to clean the distribution, type:
\verbatim make clean \endverbatim
 *
 *
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
 * The \c <tt>--benchmark</tt> parameter enables the total length of simulation (i.e., the number of
 * time steps) to be overridden by setting a new number of time  steps to simulate. This is
 * particularly useful for performance evaluation and benchmarking. As the code performance is
 * relatively stable, 50-100 time steps is  usually enough to predict the simulation duration.
 * This parameter can also be used to quickly check the simulation is set up correctly.
 *
 * The \c <tt>--verbose</tt> parameter enables to select between three levels of verbosity. For
 * routine simulations, the verbose level of 0 (the default one) is usually sufficient. For more
 * information about the simulation, checking the parameters of the simulation, code version,
 * GPU used, file paths, and debugging running scripts, verbose levels 1 and 2 may be very useful.
 *
 * The \c -h and \c --help parameters print all the parameters of the C++ code. The
 * <tt> --version </tt>parameter reports detail information about the code useful for  debugging and
 * bug reports. It prints out the internal version, the build date and time, the git hash allowing
 * us to track the version of the source code, the operating system, the compiler name and version
 * and the instruction set used.
 *
 * For jobs that are expected to run for a very long time, it may be useful to  checkpoint and
 * restart the execution. One motivation is the wall clock limit  per task on clusters where jobs
 * must fit within a given time span (e.g. 24 hours). The second motivation is a level of
 * fault-tolerance, where you can back up the state of the simulation after a predefined period.
 * To enable checkpoint-restart, the user is asked to specify a file to store the actual state of
 * the simulation by  <tt>--checkpoint_file</tt> and the period in seconds after which the
 * simulation will be interrupted by <tt>--checkpoint_interval</tt>.  When running on a cluster,
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
 * stored on disk analogous to  the sensor.record input. If the \c -p or \c --p\_raw flags are set
 * (these are equivalent), a time series of  the acoustic pressure at the grid points specified by
 * the sensor mask is recorded. If the \c --p_rms, \c --p_max, \c --p_min flags are set, the root
 * mean square and/or maximum and/or minimum values of the pressure at the grid points specified by
 * the sensor mask are recorded. If the \c --p_final flag is set, the values for the entire acoustic
 * pressure field in the final time step of the simulation is stored (this will always include
 * the PML, regardless of  the setting for <tt> `PMLInside'</tt>).
 * The flags \c --p_max_all and \c --p_min_all allow to calculate the maximum and  minimum values
 * over the entire acoustic pressure field, regardless on the shape of the sensor mask. Flags to
 * record the acoustic particle velocity are defined in an analogous fashion. For proper calculation
 * of acoustic intensity, the particle velocity has to be shifted onto the same grid as the acoustic
 * pressure. This can be done by setting \c --u_non_staggered_raw flag, that first shifts the
 * particle velocity and then samples the grid points specified by the sensor mask. Since the
 * shift operation requires additional FFTs, the impact on the simulation time may be significant.
 *
 * Any combination of <tt>p</tt> and <tt>u</tt> flags is admissible. If no output flag is set,
 * a time-series for the acoustic pressure is recorded. If it is not necessary to collect the output
 * quantities over the entire simulation, the starting time step when the collection begins can
 * be specified using the -s parameter.  Note, the index for the first time step is 1 (this follows
 * the MATLAB indexing convention).
 *
 * The \c --copy_sensor_mask will copy the sensor from the input file to the output  one at the end
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
 *
 * @section HDF5Files 4 HDF5 File Structure
 *
 * As the C++ code has been designed as a standalone application to be able
 * to run without MATLAB (e.g. on servers and supercomputers without MATLAB
 * support), it is necessary to import the simulation data from an
 * external input file and store the simulation outputs into  an output
 * file. Although MATLAB file format could have been used, this format is
 * not opened, it is dependent on MATLAB license, and it is not designed for
 * parallel I/O operations.  Therefore, Hierarchical Data Format (HDF5) has
 * been chosen.
 *
 * HDF5 (http://www.hdfgroup.org/HDF5/) is a data model, library, and file
 * format for storing and managing data. It supports an unlimited variety of
 * datatypes, and is designed for flexible and efficient I/O and for high
 * volume and complex data. HDF5 is portable and extensible.  The HDF5
 * Technology suite includes tools and applications for managing,
 * manipulating, viewing, and analysing data in the HDF5 format.
 *
 * An HDF5 file is a container for storing a variety of scientific data and
 * is composed of two primary types of objects: groups and datasets. HDF5
 * group is a  structure containing zero or more HDF5 objects, together with
 * supporting metadata.  HDF5 group can be seen as a disk folder.  HDF5
 * dataset is a multidimensional array of data elements, together with
 * supporting metadata.  HDF5 dataset can be seen as a disk file.  Any HDF5
 * group or dataset may have an associated attribute list. An HDF5 attribute
 * is a user-defined HDF5 structure that provides extra information about an
 * HDF5 object. More information can be learnt from the HDF5 documentation
 * (http://www.hdfgroup.org/HDF5/doc/index.html).
 *
 * The input file contains a file header with brief description of the
 * simulation stored in string attributes and the root group \c `/' storing
 * all simulation properties in the form of 3D datasets). The output file
 * contains a file header with the simulation description as well as the
 * performance statistics such as simulation time and memory requirements
 * stored in string attributes.  The results of the simulation are stored in
 * the root group in the form of 3D datasets.
 *
 *
 *
 *
\verbatim
==============================================================================================================
Input File Header
=============================================================================================================
created_by                              Short description of the tool that
created this file creation_date                           Date when the
file was created file_description                        Short description
of the content of the file (e.g. simulation name) file_type
Type of the file (`input') major_version                           Major
version of the file definition (`1') minor_version
Minor version of the file definition (`0')
==============================================================================================================
\endverbatim
*
\verbatim
==============================================================================================================
Output File Header
==============================================================================================================
created_by                              Short description of the tool that
created this file creation_date                           Date when the
file was created file_description                        Short description
of the content of the file (e.g. simulation name) file_type
Type of the file (`output') major_version                           Major
version of the file definition ('1') minor_version
Minor version of the file definition (`0')
-------------------------------------------------------------------------------------------------------------
host_names                              List of hosts (computer names) the
simulation was executed on number_of_cpu_cores                     Number
of CPU cores used for the simulation data_loading_phase_execution_time
Time taken to load data from the file pre-processing_phase_execution_time
Time taken to pre-process data simulation_phase_execution_time         Time
taken to run the simulation post-processing_phase_execution_time    Time
taken to complete the post-processing phase total_execution_time
Total execution time peak_core_memory_in_use                 Peak memory
required per core during the simulation total_memory_in_use Total
Peak memory in use
==============================================================================================================
\endverbatim
 *
 *
 *
 * All datasets are always stored as three dimensional with the size defined
 * by a tripled of <tt>(X, Y, Z)</tt>. In order to support scalars, 1D and
 * 2D vectors, the unused dimensions are set to 1 . For example, a scalar
 * variable is stored with the dimension sizes of <tt>(1,1,1)</tt>, a 1D
 * vector oriented in Y dimension is then stored as <tt>(1, Y, 1)</tt>. If
 * the datasets stores a complex variable, it is necessary to double the
 * lowest used dimension size (i.e. \c X for a 3D matrix, \c Y for a 1D
 * vector oriented in Y). Note, the datasets are physically stored
 * in row-major order (in contrast to column-major order used by MATLAB)
 * using either the \c `H5T_IEEE_F32LE' data type for floating point dataset
 * or \c `H5T_STD_U64LE' for integer based datasets.
 *
 * In order to enable compression of big datasets (3D variables, output time
 * series), selected datasets are not stored as monolithic blocks
 * but broken into chunks that are compressed by the ZIP library and stored
 * separately. The chunk size is  defined as follows:
 *
 *
 * \li <tt>(X, Y, 1) </tt> in the case of 3D variables (one 2D slab).  \li
 * <tt>(Nsens, 1, 1) </tt> in the case of the output time series (one time step
 * of the simulation)
 *
 * All datasets have two attributes that specify the content of the dataset.
 * The \c `data_type' attribute specifies the data type of the dataset. The
 * admissible values are either \c `float' or \c `long'. The \c
 * `domain_type' attribute specifies the domain of the dataset. The
 * admissible values are either \c `real' for the real domain or \c
 * `complex' for the complex domain.  The C++ code reads these attributes
 * and checks their values.
 *
 * Complete list of variables in the input HDF5 file and the output HDF5
 * file along with their sizes, data types and conditions of presence are
 * shown bellow.
 *
\verbatim
==============================================================================================================
Input File Datasets
==============================================================================================================
Name                            Size            Data type       Domain Type     Condition of Presence
==============================================================================================================
1. Simulation Flags
--------------------------------------------------------------------------------------------------------------
`ux_source_flag'                (1, 1, 1)       `long'          `real'
`uy_source_flag'                (1, 1, 1)       `long'          `real'
`uz_source_flag'                (1, 1, 1)       `long'          `real'
`p_source_flag'                 (1, 1, 1)       `long'          `real'
`p0_source_flag'                (1, 1, 1)       `long'          `real'
`transducer_source_flag'        (1, 1, 1)       `long'          `real'
`nonuniform_grid_flag'          (1, 1, 1)       `long'          `real'
`nonlinear_flag'		(1, 1, 1)       `long'          `real'
`absorbing_flag'		(1, 1, 1)       `long'          `real'
--------------------------------------------------------------------------------------------------------------
2. Grid Properties
--------------------------------------------------------------------------------------------------------------
`Nx'                            (1, 1, 1)       `long'          `real'
`Ny'                            (1, 1, 1)       `long'          `real'
`Nz'                            (1, 1, 1)       `long'          `real'
`Nt'                            (1, 1, 1)       `long'          `real'
`dt'                            (1, 1, 1)       `float'         `real'
`dx'                            (1, 1, 1)       `float'         `real'
`dy'                            (1, 1, 1)       `float'         `real'
`dz'                            (1, 1, 1)       `float'         `real'
--------------------------------------------------------------------------------------------------------------
3.1 Medium Properties
--------------------------------------------------------------------------------------------------------------
`rho0'                          (Nx, Ny, Nz)    `float'         `real'
(1, 1, 1)       `float'         `real'          if `rho0' is scalar
`rho0_sgx'                      (Nx, Ny, Nz)    `float'         `real'
(1, 1, 1)       `float'         `real'          if `rho0' is scalar
`rho0_sgy'                      (Nx, Ny, Nz)    `float'         `real'
(1, 1, 1)       `float'         `real'          if `rho0' is scalar
`rho0_sgz'                      (Nx, Ny, Nz)    `float'         `real'
(1, 1, 1)       `float'         `real'          if `rho0' is scalar
`c0'                            (Nx, Ny, Nz)    `float'         `real'
(1, 1, 1)       `float'         `real'          if `c0' is scalar
`c_ref'                         (1, 1, 1)       `float'         `real'
--------------------------------------------------------------------------------------------------------------
3.2 Nonlinear Medium Properties
These are only defined if (nonlinear_flag == 1)
--------------------------------------------------------------------------------------------------------------
`BonA'                          (Nx, Ny, Nz)    `float'         `real'
--------------------------------------------------------------------------------------------------------------
3.3 Absorbing Medium Properties
These are only defined if (absorbing_flag == 1)
--------------------------------------------------------------------------------------------------------------
`alpha_coeff'                   (Nx, Ny, Nz)    `float'         `real'
(1, 1, 1)       `float'         `real'          if `alpha_coeff' is scalar
`alpha_power'                   (1, 1, 1)       `float'         `real'
--------------------------------------------------------------------------------------------------------------
4. Sensor Variables
--------------------------------------------------------------------------------------------------------------
`sensor_mask_index'             (Nsens, 1, 1)   `long'          `real'
--------------------------------------------------------------------------------------------------------------
5.1 Velocity Source Terms
These are only defined if (ux_source_flag == 1) || (uy_source_flag == 1) || (uz_source_flag == 1)
--------------------------------------------------------------------------------------------------------------
`u_source_mode'                 (1, 1, 1)          `long'       `real'
`u_source_many'                 (1, 1, 1)          `long'       `real'
`u_source_index'                (Nsrc, 1, 1)       `long'       `real'
`ux_source_input'               (Nsrc, Nt_src, 1)  `float'      `real'          if (ux_source_flag == 1) &&
(u_source_many == 1)
(Nt_src, 1 , 1)    `float'      `real'          if (ux_source_flag == 1) &&
(u_source_many == 0)
`uy_source_input'               (Nsrc, Nt_src, 1)  `float'      `real'          if (uy_source_flag == 1) &&
(u_source_many == 1)
(Nt_src, 1, 1)     `float'      `real'          if (uy_source_flag == 1) &&
(u_source_many == 0)
`uz_source_input'               (Nsrc, Nt_src, 1)  `float'      `real'          if (uz_source_flag == 1) &&
(u_source_many == 1)
(Nt_src, 1, 1)     `float'      `real'          if (uz_source_flag == 1) &&
(u_source_many == 0)
--------------------------------------------------------------------------------------------------------------
5.2 Pressure Source Terms
These are only defined if (p_source_flag == 1)
--------------------------------------------------------------------------------------------------------------
`p_source_mode'                 (1, 1, 1)          `long'       `real'
`p_source_many'                 (1, 1, 1)          `long'       `real'
`p_source_index'                (Nsrc, 1, 1)       `long'       `real'
`p_source_input'                (Nsrc, Nt_src, 1)  `float'      `real'          if (p_source_many == 1)
(Nt_src, 1, 1)     `float'      `real'          if (p_source_many == 0)
--------------------------------------------------------------------------------------------------------------
5.3 Transducer Source Terms
These are only defined if (transducer_source_flag == 1)
--------------------------------------------------------------------------------------------------------------
`u_source_index'        	(Nsrc, 1, 1)    `long'          `real'
`transducer_source_input'       (Nt_src, 1, 1)  `float'         `real'
`delay_mask'			(Nsrc, 1, 1)    `float'         `real'
--------------------------------------------------------------------------------------------------------------
5.4 IVP Source Terms
These are only defined if ( p0_source_flag ==1)
--------------------------------------------------------------------------------------------------------------
`p0_source_input'               (Nx, Ny, Nz)    `float'         `real'
--------------------------------------------------------------------------------------------------------------
6. Non-uniform Grids
These are only defined if (`nonuniform_grid_flag ==1')
--------------------------------------------------------------------------------------------------------------
`dxudxn'        		(Nx, 1, 1)      `float'         `real'
`dxudxn_sgx'                    (Nx, 1, 1)      `float'         `real'
`dyudyn'                        (1, Ny, 1)      `float'         `real'
`dyudyn_sgy'                    (1, Ny, 1)      `float'         `real'
`dzudzn'                        (1, 1, Nz)      `float'         `real'
`dzudzn_sgz'            	(1, 1, Nz)      `float'         `real'
--------------------------------------------------------------------------------------------------------------
7. K-space and Shift Variables
--------------------------------------------------------------------------------------------------------------
`ddx_k_shift_pos_r'     	(Nx/2 + 1, 1, 1)  `float'       `complex'
`ddx_k_shift_neg_r'             (Nx/2 + 1, 1, 1)  `float'       `complex'
`ddy_k_shift_pos'               (1, Ny, 1)        `float'       `complex'
`ddy_k_shift_neg'               (1, Ny, 1)        `float'       `complex'
`ddz_k_shift_pos'               (1, 1, Nz)        `float'       `complex
`ddz_k_shift_neg'               (1, 1, Nz)        `float'       `complex'
--------------------------------------------------------------------------------------------------------------
8. PML Variables
--------------------------------------------------------------------------------------------------------------
`pml_x_size'                    (1, 1, 1)       `long'          `real'
`pml_y_size'                    (1, 1, 1)       `long'          `real'
`pml_z_size'                    (1, 1, 1)       `long'          `real'
`pml_x_alpha'                   (1, 1, 1)       `float'         `real'
`pml_y_alpha'                   (1, 1, 1)       `float'         `real'
`pml_z_alpha'                   (1, 1, 1)       `float'         `real'

`pml_x'                         (Nx, 1, 1)      `float'         `real'
`pml_x_sgx'                     (Nx, 1, 1)      `float'         `real'
`pml_y'                         (1, Ny, 1)      `float'         `real'
`pml_y_sgy'                     (1, Ny, 1)      `float'         `real'
`pml_z'                         (1, 1, Nz)      `float'         `real'
`pml_z_sgz'                     (1, 1, Nz)      `float'         `real'
==============================================================================================================
\endverbatim


\verbatim
==============================================================================================================
Output File Datasets
==============================================================================================================
Name                            Size           Data type        Domain Type     Condition of Presence
==============================================================================================================
1. Simulation Flags
--------------------------------------------------------------------------------------------------------------
`ux_source_flag'                (1, 1, 1)       `long'          `real'
`uy_source_flag'                (1, 1, 1)       `long'          `real'
`uz_source_flag'                (1, 1, 1)       `long'          `real'
`p_source_flag'                 (1, 1, 1)       `long'          `real'
`p0_source_flag'                (1, 1, 1)       `long'          `real'
`transducer_source_flag'        (1, 1, 1)       `long'          `real'
`nonuniform_grid_flag'          (1, 1, 1)       `long'          `real'
`nonlinear_flag'		(1, 1, 1)       `long'          `real'
`absorbing_flag'		(1, 1, 1)       `long'          `real'
--------------------------------------------------------------------------------------------------------------
2. Grid Properties
--------------------------------------------------------------------------------------------------------------
`Nx'                            (1, 1, 1)       `long'          `real'
`Ny'                            (1, 1, 1)       `long'          `real'
`Nz'                            (1, 1, 1)       `long'          `real'
`Nt'                            (1, 1, 1)       `long'          `real'
`dt'                            (1, 1, 1)       `float'         `real'
`dx'                            (1, 1, 1)       `float'         `real'
`dy'                            (1, 1, 1)       `float'         `real'
`dz'                            (1, 1, 1)       `float'         `real'
-------------------------------------------------------------------------------------------------------------
3. PML Variables
--------------------------------------------------------------------------------------------------------------
`pml_x_size'                    (1, 1, 1)       `long'          `real'
`pml_y_size'                    (1, 1, 1)       `long'          `real'
`pml_z_size'                    (1, 1, 1)       `long'          `real'
`pml_x_alpha'                   (1, 1, 1)       `float'         `real'
`pml_y_alpha'                   (1, 1, 1)       `float'         `real'
`pml_z_alpha'                   (1, 1, 1)       `float'         `real'

`pml_x'                         (Nx, 1, 1)      `float'         `real'
`pml_x_sgx'                     (Nx, 1, 1)      `float'         `real'
`pml_y'                         (1, Ny, 1)      `float'         `real'
`pml_y_sgy'                     (1, Ny, 1)      `float'         `real'
`pml_z'                         (1, 1, Nz)      `float'         `real'
`pml_z_sgz'                     (1, 1, Nz)      `float'         `real'
--------------------------------------------------------------------------------------------------------------
4.1 Medium Properties
--------------------------------------------------------------------------------------------------------------
`c_ref'                         (1, 1, 1)       `float'         `real'
--------------------------------------------------------------------------------------------------------------
4.2 Velocity Source Terms
These are only defined if (ux_source_flag == 1) || (uy_source_flag == 1) || (uz_source_flag == 1)
--------------------------------------------------------------------------------------------------------------
`u_source_mode'                 (1, 1, 1)       `long'          `real'
`u_source_many'                 (1, 1, 1)       `long'          `real'
--------------------------------------------------------------------------------------------------------------
4.3 Pressure Source Terms
These are only defined if (p_source_flag == 1)
--------------------------------------------------------------------------------------------------------------
`p_source_mode'                 (1, 1, 1)       `long'          `real'
`p_source_many'                 (1, 1, 1)       `long'          `real'
--------------------------------------------------------------------------------------------------------------
5.1 Simulation Results - Acoustic Pressure
--------------------------------------------------------------------------------------------------------------
`p'                             (Nsens, Nt - s, 1) `float'      `real'          if (-p) || (--p_raw)
`p_rms'                         (Nsens, 1, 1)      `float'      `real'          if (--p_rms)
`p_max'                         (Nsens, 1, 1)      `float'      `real'          if (--p_max)
`p_final'                       (Nx, Ny, Nz)       `float'      `real'          if (--p_final)
---------------------------------------z-----------------------------------------------------------------------
5.2 Simulation Results - Particle Velocity
--------------------------------------------------------------------------------------------------------------
`ux'                            (Nsens, Nt - s, 1) `float'      `real'          if (-u) || (--u_raw)
`uy'                            (Nsens, Nt - s, 1) `float'      `real'          if (-u) || (--u_raw)
`uz'                            (Nsens, Nt - s, 1) `float'      `real'          if (-u) || (--u_raw)

`ux_rms'                        (Nsens, 1, 1)      `float'      `real'          if (--u_rms)
`uy_rms'                        (Nsens, 1, 1)      `float'      `real'          if (--u_rms)
`uz_rms'                        (Nsens, 1, 1)      `float'      `real'          if (--u_rms)

`ux_max'                        (Nsens, 1, 1)      `float'      `real'          if (--u_max)
`uy_max'                        (Nsens, 1, 1)      `float'      `real'          if (--u_max)
`uz_max'                        (Nsens, 1, 1)      `float'      `real'          if (--u_max)

`ux_final'                      (Nx, Ny, Nz)       `float'      `real'          if (--u_final)
`uy_final'                      (Nx, Ny, Nz)       `float'      `real'          if (--u_final)
`uz_final'                      (Nx, Ny, Nz)       `float'      `real'          if (--u_final)
--------------------------------------------------------------------------------------------------------------
5.3 Simulation Results -  Acoustic Intensity
--------------------------------------------------------------------------------------------------------------
`Ix_avg'                        (Nsens, 1, 1)      `float'      `real'          if (-I) || (--I_avg)
`Iy_avg'                        (Nsens, 1, 1)      `float'      `real'          if (-I) || (--I_avg)
`Iz_avg'                        (Nsens, 1, 1)      `float'      `real'          if (-I) || (--I_avg)

`Ix_max'                        (Nsens, 1, 1)      `float'      `real'          if (--I_max)
`Iy_max'                        (Nsens, 1, 1)      `float'      `real'          if (--I_max)
`Iz_max'                        (Nsens, 1, 1)      `float'      `real'          if (--I_max)
==============================================================================================================
\endverbatim
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
