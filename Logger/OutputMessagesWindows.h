/**
 * @file      OutputMessagesWindows.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing all windows specific messages going to the standard output.
 *
 * @version   kspaceFirstOrder3D 3.6
 *
 * @date      13 July      2016, 12:35 (created) \n
 *            22 February  2019, 21:09 (revised)
 *
 * @copyright Copyright (C) 2019 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#ifndef OUTPUT_MESSAGES_WINDOWS_H
#define OUTPUT_MESSAGES_WINDOWS_H

/**
 * @brief   Datatype for output messages.
 * @details Datatype for output messages.
 */
using OutputMessage = const std::string;


//------------------------------------------------- Common outputs ---------------------------------------------------//
/// Output message - first separator
OutputMessage kOutFmtFirstSeparator
  = "+---------------------------------------------------------------+\n";
/// Output message  - separator
OutputMessage kOutFmtSeparator
  = "+---------------------------------------------------------------+\n";
/// Output message -last separator
OutputMessage kOutFmtLastSeparator
  = "+---------------------------------------------------------------+\n";

/// Output message - new line
OutputMessage kOutFmtNewLine
  = "\n";
/// Output message - Done with two spaces
OutputMessage kOutFmtDone
  = "  Done |\n";
/// Output message - finish line without done
OutputMessage kOutFmtNoDone
  = "       |\n";
/// Output message - failed message
OutputMessage kOutFmtFailed
  = "Failed |\n" ;
/// Output message - vertical line
OutputMessage kOutFmtVerticalLine
  = "|";

/// Output message
OutputMessage kOutFmtCodeName
  = "|                 %s                  |\n";
/// Output message
OutputMessage kOutFmtNumberOfThreads
  = "| Number of CPU threads:                              %9lu |\n";
/// Output message
OutputMessage kOutFmtSimulationDetailsTitle
  = "+---------------------------------------------------------------+\n"
    "|                      Simulation details                       |\n"
    "+---------------------------------------------------------------+\n";
/// Output message
OutputMessage kOutFmtInitializationHeader
  = "+---------------------------------------------------------------+\n"
    "|                        Initialization                         |\n"
    "+---------------------------------------------------------------+\n";
/// Output message
OutputMessage kOutFmtCompResourcesHeader
  = "+---------------------------------------------------------------+\n"
    "|                    Computational resources                    |\n"
    "+---------------------------------------------------------------+\n";
/// Output message
OutputMessage kOutFmtSimulationHeader
  = "+---------------------------------------------------------------+\n"
    "|                          Simulation                           |\n"
    "+----------+----------------+--------------+--------------------+\n"
    "| Progress |  Elapsed time  |  Time to go  |  Est. finish time  |\n"
    "+----------+----------------+--------------+--------------------+\n";
/// Output message
OutputMessage kOutFmtCheckpointHeader
  = "+---------------------------------------------------------------+\n"
    "|                         Checkpointing                         |\n"
    "+---------------------------------------------------------------+\n";
/// Output message
OutputMessage kOutFmtSummaryHeader
  = "+---------------------------------------------------------------+\n"
    "|                            Summary                            |\n"
    "+---------------------------------------------------------------+\n";
/// Output message
OutputMessage kOutFmtEndOfSimulation
  = "+---------------------------------------------------------------+\n"
    "|                       End of computation                      |\n"
    "+---------------------------------------------------------------+\n";

///Output message
OutputMessage kOutFmtElapsedTime
  = "| Elapsed time:                                    %11.2fs |\n";
///Output message
OutputMessage kOutFmtRecoveredFrom
  = "| Recovered from time step:                            %8ld |\n";
///Output message
OutputMessage kOutFmtHostMemoryUsage
  = "| Peak host memory in use:                           %8luMB |\n";
///Output message
OutputMessage kOutFmtDeviceMemoryUsage
  = "| Peak device memory in use:                         %8luMB |\n";
///Output message
OutputMessage kOutFmtTotalExecutionTime
  = "| Total execution time:                               %8.2fs |\n";
///Output message
OutputMessage kOutFmtLegExecutionTime
  = "| This leg execution time:                            %8.2fs |\n";

///Output message
OutputMessage kOutFmtReadingConfiguration
  = "| Reading simulation configuration:                      ";
///Output message
OutputMessage kOutFmtSelectedDevice
  = "| Selected GPU device id:                                ";
///Output message
OutputMessage kOutFmtDeviceId
  = "%6d |\n";
///Output message
OutputMessage kOutFmtDeviceName
  = "| GPU device name: %44s |\n";
///Output message
OutputMessage kOutFmtDomainSize
  = "| Domain dimensions: %42s |\n";
///Output message
OutputMessage kOutFmtDomainSizeFormat
  = "%lu x %lu x %lu";


///Output message
OutputMessage kOutFmtSimulatoinLenght
  = "| Simulation time steps:                              %9lu |\n";
///Output message
OutputMessage kOutFmtSensorMaskIndex
  = "| Sensor mask type:                                       Index |\n";
///Output message
OutputMessage kOutFmtSensorMaskCuboid
  = "| Sensor mask type:                                      Cuboid |\n";
///Output message
OutputMessage kOutFmtGitHashLeft
  = "| Git hash:            %s |\n";


///Output message
OutputMessage kOutFmtKWaveVersion
  = "kspaceFirstOrder3D-CUDA v1.3";

///Output message
OutputMessage kOutFmtFftPlans
  = "| FFT plans creation:                                    ";
///Output message
OutputMessage kOutFmtPreProcessing
  = "| Pre-processing phase:                                  ";
///Output message
OutputMessage kOutFmtDataLoading
  = "| Data loading:                                          ";
///Output message
OutputMessage kOutFmtMemoryAllocation
  = "| Memory allocation:                                     ";
///Output message
OutputMessage kOutFmtCurrentHostMemory
  = "| Current host memory in use:                        %8luMB |\n";
///Output message
OutputMessage kOutFmtCurrentDeviceMemory
  = "| Current device memory in use:                      %8luMB |\n";

///Output message
OutputMessage kOutFmtCudaGridShapeFormat
  = "%d x %d";
///Output message
OutputMessage kOutFmtCudaSolverGridShape
  = "| CUDA solver grid size [blocks x threads]: %19s |\n";
///Output message
OutputMessage kOutFmtCudaSamplerGridShape
  = "| CUDA sampler grid size [blocks x threads]: %18s |\n";

///Output message
OutputMessage kOutFmtSimulationProgress
  ="|    %2li%c   |    %9.3fs  |  %9.3fs  |  %02i/%02i/%02i %02i:%02i:%02i |\n";

///Output message
OutputMessage kOutFmtSimulationEndSeparator
  = "+----------+----------------+--------------+--------------------+\n";
///Output message
OutputMessage kOutFmtSimulatoinFinalSeparator
  = "+----------+----------------+--------------+--------------------+\n";

///Output message
OutputMessage kOutFmtCheckpointTimeSteps
  = "| Number of time steps completed:                    %10u |\n";
///Output message
OutputMessage kOutFmtCreatingCheckpoint
  = "| Creating checkpoint:                                   ";
///Output message
OutputMessage kOutFmtPostProcessing
  = "| Sampled data post-processing:                          ";
///Output message
OutputMessage kOutFmtStoringCheckpointData
  = "| + Storing checkpoint data:                             ";
///Output message
OutputMessage kOutFmtStoringSensorData
  = "| + Storing sensor data:                                 ";
///Output message
OutputMessage kOutFmtReadingInputFile
  = "| + Reading input file:                                  ";
///Output message
OutputMessage kOutFmtReadingCheckpointFile
  = "| + Reading checkpoint file:                             ";
///Output message
OutputMessage kOutFmtReadingOutputFile
  = "| + Reading output file:                                 ";
///Output message
OutputMessage kOutFmtCreatingOutputFile
  = "| + Creating output file:                                ";
///Output message
OutputMessage kOutFmtInputFile
  = "Input file:  ";
///Output message
OutputMessage kOutFmtOutputFile
  = "Output file: ";
///Output message
OutputMessage kOutFmtCheckpointFile
  = "Check file:  ";
///Output message
OutputMessage kOutFmtCheckpointInterval
  = "| Checkpoint interval:                                %8lus |\n";
///Output message
OutputMessage kOutFmtCompressionLevel
  = "| Compression level:                                   %8lu |\n";
///Output message
OutputMessage kOutFmtPrintProgressIntrerval
  = "| Print progress interval:                            %8lu%% |\n";
///Output message
OutputMessage kOutFmtBenchmarkTimeStep
  = "| Benchmark time steps:                                %8lu |\n";
///Output message
OutputMessage kOutFmtSamplingFlags
  = "+---------------------------------------------------------------+\n"
    "|                        Sampling flags                         |\n"
    "+---------------------------------------------------------------+\n";
///Output message
OutputMessage kOutFmtSamplingStartsAt
  = "| Sampling begins at time step:                        %8lu |\n";
///Output message
OutputMessage kOutFmtCopySensorMask
  = "| Copy sensor mask to output file:                          Yes |\n";



//------------------------------------------------ Print code version ------------------------------------------------//
/// Print version output message
OutputMessage kOutFmtBuildNoDataTime
  = "+---------------------------------------------------------------+\n"
    "|                       Build information                       |\n"
    "+---------------------------------------------------------------+\n"
    "| Build number:     kspaceFirstOrder3D v3.6                     |\n"
    "| Build date:       %*.*s                                 |\n"
    "| Build time:       %*.*s                                    |\n";

/// Print version output message
OutputMessage kOutFmtVersionGitHash
  = "| Git hash:         %s    |\n";

/// Print version output message
OutputMessage kOutFmtLinuxBuild
  = "| Operating system: Linux x64                                   |\n";
/// Print version output message
OutputMessage kOutFmtWindowsBuild
  = "| Operating system: Windows x64                                 |\n";
/// Print version output message
OutputMessage kOutFmtMacOsBuild
  = "| Operating system: Mac OS X x64                                |\n";

/// Print version output message
OutputMessage kOutFmtGnuCompiler
  = "| Compiler name:    GNU C++ %.19s                               |\n";
/// Print version output message
OutputMessage kOutFmtIntelCompiler
  = "| Compiler name:    Intel C++ %d                              |\n";
/// Print version output message
OutputMessage kOutFmtVisualStudioCompiler
  = "| Compiler name:    Visual Studio C++ %d                      |\n";

/// Print version output message
OutputMessage kOutFmtAVX2
  = "| Instruction set:  Intel AVX 2                                 |\n";
/// Print version output message
OutputMessage kOutFmtAVX
  = "| Instruction set:  Intel AVX                                   |\n";
/// Print version output message
OutputMessage kOutFmtSSE42
  = "| Instruction set:  Intel SSE 4.2                               |\n";
/// Print version output message
OutputMessage kOutFmtSSE41
  = "| Instruction set:  Intel SSE 4.1                               |\n";
/// Print version output message
OutputMessage kOutFmtSSE3
  = "| Instruction set:  Intel SSE 3                                 |\n";
/// Print version output message
OutputMessage kOutFmtSSE2
  = "| Instruction set:  Intel SSE 2                                 |\n";

/// Print version output message
OutputMessage kOutFmtCudaRuntimeNA
  = "| CUDA runtime:     N/A                                         |\n";
/// Print version output message
OutputMessage kOutFmtCudaRuntime
  = "│ CUDA runtime:     %d.%d                                         │\n";
/// Print version output message
OutputMessage kOutFmtCudaRuntime10
  = "│ CUDA runtime:     %d.%d                                        │\n";
/// Print version output message
OutputMessage kOutFmtCudaDriver
  = "│ CUDA driver:      %d.%d                                         │\n";
/// Print version output message
OutputMessage kOutFmtCudaDriver10
  = "│ CUDA driver:      %d.%d                                        │\n";
/// Print version output message
OutputMessage kOutFmtCudaDeviceInfoNA
  = "| CUDA code arch:   N/A                                         |\n"
    "+---------------------------------------------------------------+\n"
    "| CUDA device id:   N/A                                         |\n"
    "| CUDA device name: N/A                                         |\n"
    "| CUDA capability:  N/A                                         |\n";

/// Print version output message
OutputMessage kOutFmtCudaCodeArch
  = "| CUDA code arch:   %1.1f                                         |\n";
/// Print version output message
OutputMessage kOutFmtCudaDevice
  = "| CUDA device id:   %d                                           |\n";
/// Print version output message
OutputMessage kOutFmtCudaDeviceName
  = "| CUDA device name: %s %.*s|\n";
/// Print version output message
OutputMessage kOutFmtCudaDeviceNamePadding
  =  "                                        ";
/// Print version output message
OutputMessage kOutFmtCudaCapability
  = "| CUDA capability:  %d.%d                                         |\n";
/// Print version output message
OutputMessage kOutFmtLicense
  = "+---------------------------------------------------------------+\n"
    "| Contact email:    jarosjir@fit.vutbr.cz                       |\n"
    "| Contact web:      http://www.k-wave.org                       |\n"
    "+---------------------------------------------------------------+\n"
    "|       Copyright (C) 2019 Jiri Jaros and Bradley Treeby        |\n"
    "+---------------------------------------------------------------+\n";




//------------------------------------------------- Usage ------------------------------------------------------------//
/// Usage massage
OutputMessage kOutFmtUsagePart1
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
OutputMessage kOutFmtUsagePart2
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
    "| --copy_sensor_mask            | Copy sensor mask to the       |\n"
    "|                               |    output file                |\n"
    "+-------------------------------+-------------------------------+\n";

/// Usage massage
OutputMessage kOutFmtUsageThreads
  = "| -t <num_threads>              | Number of CPU threads         |\n"
    "|                               |  (default = %2d)               |\n";

#endif /* OUTPUT_MESSAGES_WINDOWS_H */
