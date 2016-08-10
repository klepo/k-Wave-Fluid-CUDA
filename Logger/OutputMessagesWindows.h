/**
 * @file        OutputMessagesWindows.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing all messages going to the standard output,
 *              windows version.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        13 July     2016, 12:35 (created) \n
 *              29 July     2016, 16:44 (revised)
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

#ifndef OUTPUT_MESSAGES_WINDOWS_H
#define OUTPUT_MESSAGES_WINDOWS_H

/**
 * @typedef TOutputMessage
 * @brief   Datatype for output messages.
 * @details Datatype for output messages.
 */
typedef const std::string TOutputMessage;


//--------------------------------------- Common outputs -----------------------------------------//
/// Output message - first separator
TOutputMessage OUT_FMT_FIRST_SEPARATOR
        = "+---------------------------------------------------------------+\n";
/// Output message  - separator
TOutputMessage OUT_FMT_SEPARATOR
        = "+---------------------------------------------------------------+\n";
/// Output message -last separator
TOutputMessage OUT_FMT_LAST_SEPARATOR
        = "+---------------------------------------------------------------+\n";

/// Output message - new line
TOutputMessage OUT_FMT_NEW_LINE
        = "\n";
/// Output message - Done with two spaces
TOutputMessage OUT_FMT_DONE
        = "  Done |\n";
/// Output message - finish line without done
TOutputMessage OUT_FMT_FINSIH_NO_DONE
        = "       |\n";
/// Output message - failed message
TOutputMessage OUT_FMT_FAILED
        = "Failed |\n" ;
/// Output message - vertical line
TOutputMessage OUT_FMT_VERTICAL_LINE
        = "|";

/// Output message
TOutputMessage OUT_FMT_CODE_NAME
        = "|                 %s                  |\n";
/// Output message
TOutputMessage OUT_FMT_NUMBER_OF_THREADS
        = "| Number of CPU threads:                              %9lu |\n";
/// Output message
TOutputMessage OUT_FMT_SIMULATION_DETAIL_TITLE
        = "+---------------------------------------------------------------+\n"
          "|                      Simulation details                       |\n"
          "+---------------------------------------------------------------+\n";
/// Output message
TOutputMessage OUT_FMT_INIT_HEADER
        = "+---------------------------------------------------------------+\n"
          "|                        Initialization                         |\n"
          "+---------------------------------------------------------------+\n";
/// Output message
TOutputMessage OUT_FMT_COMP_RESOURCES_HEADER
        = "+---------------------------------------------------------------+\n"
          "|                    Computational resources                    |\n"
          "+---------------------------------------------------------------+\n";
/// Output message
TOutputMessage OUT_FMT_SIMULATION_HEADER
        = "+---------------------------------------------------------------+\n"
          "|                          Simulation                           |\n"
          "+----------+----------------+--------------+--------------------+\n"
          "| Progress |  Elapsed time  |  Time to go  |  Est. finish time  |\n"
          "+----------+----------------+--------------+--------------------+\n";
/// Output message
TOutputMessage OUT_FMT_CHECKPOINT_HEADER
        = "+---------------------------------------------------------------+\n"
          "|                         Checkpointing                         |\n"
          "+---------------------------------------------------------------+\n";
/// Output message
TOutputMessage OUT_FMT_SUMMARY_HEADER
        = "+---------------------------------------------------------------+\n"
          "|                            Summary                            |\n"
          "+---------------------------------------------------------------+\n";
/// Output message
TOutputMessage OUT_FMT_END_OF_SIMULATION  =
        "+---------------------------------------------------------------+\n"
        "|                       End of computation                      |\n"
        "+---------------------------------------------------------------+\n";

///Output message
TOutputMessage OUT_FMT_ELAPSED_TIME
        = "| Elapsed time:                                    %11.2fs |\n";
///Output message
TOutputMessage OUT_FMT_RECOVER_FROM
        = "| Recovered from time step:                            %8ld |\n";
///Output message
TOutputMessage OUT_FMT_HOST_MEMORY_USAGE
        = "| Peak host memory in use:                           %8luMB |\n";
///Output message
TOutputMessage OUT_FMT_DEVICE_MEMORY_USAGE
        = "| Peak device memory in use:                         %8luMB |\n";
///Output message
TOutputMessage OUT_FMT_TOTAL_EXECUTION_TIME
        = "| Total execution time:                               %8.2fs |\n";
///Output message
TOutputMessage OUT_FMT_LEG_EXECUTION_TIME
        = "| This leg execution time:                            %8.2fs |\n";

///Output message
TOutputMessage OUT_FMT_READING_CONFIGURATION
        = "| Reading simulation configuration:                      ";
///Output message
TOutputMessage OUT_FMT_SELECTED_DEVICE
        = "| Selected GPU device id:                                ";
///Output message
TOutputMessage OUT_FMT_DEVICE_ID
        = "%6d |\n";
///Output message
TOutputMessage OUT_FMT_DEVICE_NAME
        = "| GPU device name: %44s |\n";
///Output message
TOutputMessage OUT_FMT_DOMAIN_SIZE
        = "| Domain dimensions: %42s |\n";
///Output message
TOutputMessage OUT_FMT_DOMAIN_SIZE_FORMAT
        = "%lu x %lu x %lu";


///Output message
TOutputMessage OUT_FMT_SIMULATION_LENGTH
        = "| Simulation time steps:                              %9lu |\n";
///Output message
TOutputMessage OUT_FMT_SENSOR_MASK_INDEX
        = "| Sensor mask type:                                       Index |\n";
///Output message
TOutputMessage OUT_FMT_SENSOR_MASK_CUBOID
        = "| Sensor mask type:                                      Cuboid |\n";
///Output message
TOutputMessage OUT_FMT_GIT_HASH_LEFT
        = "| Git hash:            %s |\n";


///Output message
TOutputMessage OUT_FMT_KWAVE_VERSION
        = "kspaceFirstOrder3D-CUDA v1.1";

///Output message
TOutputMessage OUT_FMT_FFT_PLANS
        = "| FFT plans creation:                                    ";
///Output message
TOutputMessage OUT_FMT_PRE_PROCESSING
        = "| Pre-processing phase:                                  ";
///Output message
TOutputMessage OUT_FMT_DATA_LOADING
        = "| Data loading:                                          ";
///Output message
TOutputMessage OUT_FMT_MEMORY_ALLOCATION
        = "| Memory allocation:                                     ";
///Output message
TOutputMessage OUT_FMT_CURRENT_HOST_MEMORY
        = "| Current host memory in use:                        %8luMB |\n";
///Output message
TOutputMessage OUT_FMT_CURRENT_DEVICE_MEMORY
        = "| Current device memory in use:                      %8luMB |\n";

///Output message
TOutputMessage OUT_FMT_CUDA_GRID_SHAPE_FORMAT
        = "%d x %d";
///Output message
TOutputMessage OUT_FMT_CUDA_SOLVER_GRID_SHAPE
        = "| CUDA solver grid size [blocks x threads]: %19s |\n";
///Output message
TOutputMessage OUT_FMT_CUDA_SAMPLER_GRID_SHAPE
        = "| CUDA sampler grid size [blocks x threads]: %18s |\n";

///Output message
TOutputMessage OUT_FMT_SIMULATION_PROGRESS
        ="|    %2li%c   |    %9.3fs  |  %9.3fs  |  %02i/%02i/%02i %02i:%02i:%02i |\n";

///Output message
TOutputMessage OUT_FMT_SIMULATOIN_END_SEPARATOR
        = "+----------+----------------+--------------+--------------------+\n";
///Output message
TOutputMessage OUT_FMT_SIMULATION_FINAL_SEPARATOR
        = "+----------+----------------+--------------+--------------------+\n";

///Output message
TOutputMessage OUT_FMT_CHECKPOINT_TIME_STEPS
        = "| Number of time steps completed:                    %10u |\n";
///Output message
TOutputMessage OUT_FMT_CREATING_CHECKPOINT
        = "| Creating checkpoint:                                   ";
///Output message
TOutputMessage OUT_FMT_POST_PROCESSING
        = "| Sampled data post-processing:                          ";
///Output message
TOutputMessage OUT_FMT_STORING_CHECKPOINT_DATA
        = "| + Storing checkpoint data:                             ";
///Output message
TOutputMessage OUT_FMT_STORING_SENSOR_DATA
        = "| + Storing sensor data:                                 ";
///Output message
TOutputMessage OUT_FMT_READING_INPUT_FILE
        = "| + Reading input file:                                  ";
///Output message
TOutputMessage OUT_FMT_READING_CHECKPOINT_FILE
        = "| + Reading checkpoint file:                             ";
///Output message
TOutputMessage OUT_FMT_READING_OUTPUT_FILE
        = "| + Reading output file:                                 ";
///Output message
TOutputMessage OUT_FMT_CREATING_OUTPUT_FILE
        = "| + Creating output file:                                ";
///Output message
TOutputMessage OUT_FMT_INPUT_FILE
        = "Input file:  ";
///Output message
TOutputMessage OUT_FMT_OUTPUT_FILE
        = "Output file: ";
///Output message
TOutputMessage OUT_FMT_CHECKPOINT_FILE
        = "Check file:  ";
///Output message
TOutputMessage OUT_FMT_CHECKPOINT_INTERVAL
        = "| Checkpoint interval:                                %8lus |\n";
///Output message
TOutputMessage OUT_FMT_COMPRESSION_LEVEL
        = "| Compression level:                                   %8lu |\n";
///Output message
TOutputMessage OUT_FMT_PRINT_PROGRESS_INTERVAL
        = "| Print progress interval:                            %8lu%% |\n";
///Output message
TOutputMessage OUT_FMT_BENCHMARK_TIME_STEP
        = "| Benchmark time steps:                                %8lu |\n";
///Output message
TOutputMessage OUT_FMT_SAMPLING_FLAGS
        = "+---------------------------------------------------------------+\n"
          "|                        Sampling flags                         |\n"
          "+---------------------------------------------------------------+\n";
///Output message
TOutputMessage OUT_FMT_SAMPLING_BEGINS_AT
        = "| Sampling begins at time step:                        %8lu |\n";
///Output message
TOutputMessage OUT_FMT_COPY_SENSOR_MASK
        = "| Copy sensor mask to output file:                          Yes |\n";



//-------------------------------------- Print code version --------------------------------------//
/// Print version output message
TOutputMessage OUT_FMT_BUILD_NO_DATE_TIME
        = "+---------------------------------------------------------------+\n"
          "|                       Build information                       |\n"
          "+---------------------------------------------------------------+\n"
          "| Build number:     kspaceFirstOrder3D v3.4                     |\n"
          "| Build date:       %*.*s                                 |\n"
          "| Build time:       %*.*s                                    |\n";

/// Print version output message
TOutputMessage OUT_FMT_VERSION_GIT_HASH
        = "| Git hash:         %s    |\n";

/// Print version output message
TOutputMessage OUT_FMT_LINUX_BUILD
        = "| Operating system: Linux x64                                   |\n";
/// Print version output message
TOutputMessage OUT_FMT_WINDOWS_BUILD
        = "| Operating system: Windows x64                                 |\n";
/// Print version output message
TOutputMessage OUT_FMT_MAC_OS_BUILD
        = "| Operating system: Mac OS X x64                                |\n";

/// Print version output message
TOutputMessage OUT_FMT_GNU_COMPILER
        = "| Compiler name:    GNU C++ %.19s                               |\n";
/// Print version output message
TOutputMessage OUT_FMT_INTEL_COMPILER
        = "| Compiler name:    Intel C++ %d                              |\n";
/// Print version output message
TOutputMessage OUT_FMT_VISUAL_STUDIO_COMPILER
        = "| Compiler name:    Visual Studio C++ %d                      |\n";

/// Print version output message
TOutputMessage OUT_FMT_AVX2
        = "| Instruction set:  Intel AVX 2                                 |\n";
/// Print version output message
TOutputMessage OUT_FMT_AVX
        = "| Instruction set:  Intel AVX                                   |\n";
/// Print version output message
TOutputMessage OUT_FMT_SSE42
        = "| Instruction set:  Intel SSE 4.2                               |\n";
/// Print version output message
TOutputMessage OUT_FMT_SSE41
        = "| Instruction set:  Intel SSE 4.1                               |\n";
/// Print version output message
TOutputMessage OUT_FMT_SSE3
        = "| Instruction set:  Intel SSE 3                                 |\n";
/// Print version output message
TOutputMessage OUT_FMT_SSE2
        = "| Instruction set:  Intel SSE 2                                 |\n";

/// Print version output message
TOutputMessage OUT_FMT_CUDA_RUNTIME_NA
        = "| CUDA runtime:     N/A                                         |\n";
/// Print version output message
TOutputMessage OUT_FMT_CUDA_RUNTIME
        = "| CUDA runtime:     %d.%d                                         |\n";
/// Print version output message
TOutputMessage OUT_FMT_CUDA_DRIVER
        = "| CUDA driver:      %d.%d                                         |\n";

/// Print version output message
TOutputMessage OUT_FMT_CUDA_DEVICE_INFO_NA
        = "| CUDA code arch:   N/A                                         |\n"
          "+---------------------------------------------------------------+\n"
          "| CUDA device id:   N/A                                         |\n"
          "| CUDA device name: N/A                                         |\n"
          "| CUDA capability:  N/A                                         |\n";

/// Print version output message
TOutputMessage OUT_FMT_CUDA_CODE_ARCH
        = "| CUDA code arch:   %1.1f                                         |\n";
/// Print version output message
TOutputMessage OUT_FMT_CUDA_DEVICE
        = "| CUDA device id:   %d                                           |\n";
/// Print version output message
TOutputMessage OUT_FMT_CUDA_DEVICE_NAME
        = "| CUDA device name: %s %.*s|\n";
/// Print version output message
TOutputMessage OUT_FMT_CUDA_DEVICE_NAME_PADDING
        =  "                                        ";
/// Print version output message
TOutputMessage OUT_FMT_CUDA_CAPABILITY
        = "| CUDA capability:  %d.%d                                         |\n";
/// Print version output message
TOutputMessage OUT_FMT_LICENCE
        = "+---------------------------------------------------------------+\n"
          "| Contact email:    jarosjir@fit.vutbr.cz                       |\n"
          "| Contact web:      http://www.k-wave.org                       |\n"
          "+---------------------------------------------------------------+\n"
          "|       Copyright (C) 2016 Jiri Jaros and Bradley Treeby        |\n"
          "+---------------------------------------------------------------+\n";




//------------------------------ Usage ----------------------------------------//
/// Usage massage
TOutputMessage OUT_FMT_USAGE_PART_1
        = "|                             Usage                             |\n"
          "+---------------------------------------------------------------+\n"
          "|                     Mandatory parameters                      |\n"
          "+---------------------------------------------------------------+\n"
          "| -i <file_name>                | HDF5 input file               |\n"
          "| -o <file_name>                | HDF5 output file              |\n"
          "+-------------------------------+-------------------------------+\n"
          "|                      Optional parameters                      |\n"
          "+-------------------------------+-------------------------------+\n";

/// Usage massage
TOutputMessage OUT_FMT_USAGE_PART_2
        = "| -g <device_number>            | GPU device to run on          |\n"
          "|                               |   (default = the first free)  |\n"
          "| -r <interval_in_%%>            | Progress print interval       |\n"
          "|                               |   (default = %2ld%%)             |\n"
          "| -c <compression_level>        | Compression level <0,9>       |\n"
          "|                               |   (default = %1ld)               |\n"
          "| --benchmark <time_steps>      | Run only a specified number   |\n"
          "|                               |   of time steps               |\n"
          "| --verbose <level>             | Level of verbosity <0,2>      |\n"
          "|                               |   0 - basic, 1 - advanced,    |\n"
          "|                               |   2 - full                    |\n"
          "|                               |   (default = basic)           |\n"
          "| -h, --help                    | Print help                    |\n"
          "| --version                     | Print version and build info  |\n"
          "+-------------------------------+-------------------------------+\n"
          "| --checkpoint_file <file_name> | HDF5 checkpoint file          |\n"
          "| --checkpoint_interval <sec>   | Checkpoint after a given      |\n"
          "|                               |   number of seconds           |\n"
          "+-------------------------------+-------------------------------+\n"
          "|                          Output flags                         |\n"
          "+-------------------------------+-------------------------------+\n"
          "| -p                            | Store acoustic pressure       |\n"
          "|                               |   (default output flag)       |\n"
          "|                               |   (the same as --p_raw)       |\n"
          "| --p_raw                       | Store raw time series of p    |\n"
          "| --p_rms                       | Store rms of p                |\n"
          "| --p_max                       | Store max of p                |\n"
          "| --p_min                       | Store min of p                |\n"
          "| --p_max_all                   | Store max of p (whole domain) |\n"
          "| --p_min_all                   | Store min of p (whole domain) |\n"
          "| --p_final                     | Store final pressure field    |\n"
          "+-------------------------------+-------------------------------+\n"
          "| -u                            | Store ux, uy, uz              |\n"
          "|                               |    (the same as --u_raw)      |\n"
          "| --u_raw                       | Store raw time series of      |\n"
          "|                               |    ux, uy, uz                 |\n"
          "| --u_non_staggered_raw         | Store non-staggered raw time  |\n"
          "|                               |   series of ux, uy, uz        |\n"
          "| --u_rms                       | Store rms of ux, uy, uz       |\n"
          "| --u_max                       | Store max of ux, uy, uz       |\n"
          "| --u_min                       | Store min of ux, uy, uz       |\n"
          "| --u_max_all                   | Store max of ux, uy, uz       |\n"
          "|                               |   (whole domain)              |\n"
          "| --u_min_all                   | Store min of ux, uy, uz       |\n"
          "|                               |   (whole domain)              |\n"
          "| --u_final                     | Store final acoustic velocity |\n"
          "+-------------------------------+-------------------------------+\n"
          "| -s <time_step>                | When data collection begins   |\n"
          "|                               |   (default = 1)               |\n"
          "+-------------------------------+-------------------------------+\n";

/// Usage massage
TOutputMessage OUT_FMT_USAGE_THREADS
        = "| -t <num_threads>              | Number of CPU threads         |\n"
          "|                               |  (default = %2d)               |\n";

#endif /* OUTPUT_MESSAGES_WINDOWS_H */

