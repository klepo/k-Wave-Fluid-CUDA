/**
 * @file        OutputMessages.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing all messages going to the standard
 *              output, windows version.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        13 July     2016, 12:35 (created) \n
 *              13 July     2016, 11235 (revised)
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

#ifndef OUTPUT_MESSAGES_WINDOWS_H
#define OUTPUT_MESSAGES_WINDOWS_H

/**
 * @typedef TOutputMessage
 * @brief Datatype for output messages
 */
typedef const char* const TOutputMessage;


//----------------------------- Common outputs -------------------------------//
/// Output message - first separator
TOutputMessage OUT_FMT_FirstSeparator = "+---------------------------------------------------------------+\n";
/// Output message  - separator
TOutputMessage OUT_FMT_Separator      = "+---------------------------------------------------------------+\n";
/// Output message -last separator
TOutputMessage OUT_FMT_LastSeparator  = "+---------------------------------------------------------------+\n";

/// Output message - new line
TOutputMessage OUT_FMT_NewLine = "\n";
/// Output message - Done with two spaces and
TOutputMessage OUT_FMT_Done = "  Done |\n";
/// Output message - finish line without Done
TOutputMessage OUT_FMT_FinishLineNoDone = "       |\n";

/// Output message - failed message
TOutputMessage OUT_FMT_Failed = "Failed |\n" ;

/// Output message - vertical line
TOutputMessage OUT_FMT_VerticalLine = "|";

//----------------------------- Main module outputs --------------------------//

/// Main file output message
TOutputMessage OUT_FMT_CodeName  = "|                 %s                  |\n";
/// Main file output message
TOutputMessage OUT_FMT_NumberOfThreads = "| Number of CPU threads:                              %9lu |\n";
/// Format message for simulation details title
TOutputMessage OUT_FMT_SimulationDetailsTitle =
        "+---------------------------------------------------------------+\n"
        "|                      Simulation details                       |\n"
        "+---------------------------------------------------------------+\n";


/// Main file output message
TOutputMessage OUT_FMT_InitialisatoinHeader  =
        "+---------------------------------------------------------------+\n"
        "|                        Initialization                         |\n"
        "+---------------------------------------------------------------+\n";

/// Main file output message
TOutputMessage OUT_FMT_ComputationalResourcesHeader  =
        "+---------------------------------------------------------------+\n"
        "|                    Computational resources                    |\n"
        "+---------------------------------------------------------------+\n";

TOutputMessage OUT_FMT_SimulationHeader =
        "+---------------------------------------------------------------+\n"
        "|                          Simulation                           |\n"
        "+----------+----------------+--------------+--------------------+\n"
        "| Progress |  Elapsed time  |  Time to go  |  Est. finish time  |\n"
        "+----------+----------------+--------------+--------------------+\n";

TOutputMessage OUT_FMT_CheckpointHeader =
        "+---------------------------------------------------------------+\n"
        "|                         Checkpointing                         |\n"
        "+---------------------------------------------------------------+\n";

/// Main file output message
TOutputMessage OUT_FMT_SummaryHeader =
        "+---------------------------------------------------------------+\n"
        "|                            Summary                            |\n"
        "+---------------------------------------------------------------+\n";

TOutputMessage OUT_FMT_EndOfComputation  =
        "+---------------------------------------------------------------+\n"
        "|                       End of computation                      |\n"
        "+---------------------------------------------------------------+\n";

/// Main file output message
TOutputMessage OUT_FMT_ElapsedTime = "| Elapsed time:                                    %11.2fs |\n";
/// Main file output message
TOutputMessage OUT_FMT_RecoveredForm = "| Recovered from time step:                            %8ld |\n";

/// Main file output message
TOutputMessage OUT_FMT_HostMemoryUsage  = "| Peak host memory in use:                           %8luMB |\n";
/// Main file output message
TOutputMessage OUT_FMT_DeviceMemoryUsage= "| Peak device memory in use:                         %8luMB |\n";
/// Main file output message
TOutputMessage OUT_FMT_TotalExecutionTime= "| Total execution time:                               %8.2fs |\n";
/// Main file output message
TOutputMessage OUT_FMT_LegExecutionTime  = "| This leg execution time:                            %8.2fs |\n";
/// Main file output message



//----------------------- Parameters module outputs --------------------------//
/// Parameter module log message
TOutputMessage OUT_FMT_ReadingConfiguration = "| Reading simulation configuration:                      ";
/// Parameter module log message
TOutputMessage OUT_FMT_SelectedDeviceId = "| Selected GPU device id:                                ";
/// Parameter module log message
TOutputMessage OUT_FMT_DeviceId         = "%6d |\n";
/// Parameter module log message
TOutputMessage OUT_FMT_DeviceName       = "| GPU device name: %44s |\n";
/// Parameter module log message
TOutputMessage OUT_FMT_DomainSize       = "| Domain dimensions: %42s |\n";
/// Parameter module log message
TOutputMessage OUT_FMT_DomainSizeFormat = "%lu x %lu x %lu";


/// Parameter module log message
TOutputMessage OUT_FMT_SimulationLength  = "| Simulation time steps:                              %9lu |\n";
/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_SensorMaskTypeIndex  = "| Sensor mask type:                                       Index |\n";
/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_SensorMaskTypeCuboid = "| Sensor mask type:                                      Cuboid |\n";
/// ComandlineParamerers module log message
TOutputMessage TParamereres_OUT_FMT_GitHash = "| Git hash:            %s |\n";


//------------------ TKSpaceFirstOrder3DSolver module outputs ----------------//
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_Version       = "kspaceFirstOrder3D-CUDA v1.1";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_FFTPlans      = "| FFT plans creation:                                    ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_PreProcessing = "| Pre-processing phase:                                  ";
/// Main file output message
TOutputMessage OUT_FMT_DataLoading      = "| Data loading:                                          ";
/// Main file output message
TOutputMessage OUT_FMT_MemoryAllocation = "| Memory allocation:                                     ";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CurrentHostMemory   = "| Current host memory in use:                        %8luMB |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CurrentDeviceMemory = "| Current device memory in use:                      %8luMB |\n";


TOutputMessage OUT_FMT_CUDAGridShapeFormat = "%d x %d";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDASolverGridShape     = "| CUDA solver grid size [blocks x threads]: %19s |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDASamplerGridShape    = "| CUDA sampler grid size [blocks x threads]: %18s |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_SimulatoinProgress  ="|    %2li%c   |    %9.3fs  |  %9.3fs  |  %02i/%02i/%02i %02i:%02i:%02i |\n";

/// Output message to finish simulation
TOutputMessage OUT_FMT_SimulatoinEndSeparator = "+----------+----------------+--------------+--------------------+\n";
/// Output message to finish simulation when an error met
TOutputMessage OUT_FMT_SimulatoinFinalSeparator = "+----------+----------------+--------------+--------------------+\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CheckpointTimeSteps  = "| Number of time steps completed:                    %10u |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CreatingCheckpoint = "| Creating checkpoint:                                   ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_PostProcessing     = "| Sampled data post-processing:                          ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_StoringCheckpointData  = "| + Storing checkpoint data:                             ";

TOutputMessage OUT_FMT_StoringSensorData      = "| + Storing sensor data:                                 ";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_ReadingInputFile       = "| + Reading input file:                                  ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_ReadingCheckpointFile  = "| + Reading checkpoint file:                             ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_ReadingOuptutFile      = "| + Reading output file:                                 ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CreatingOutputFile     = "| + Creating output file:                                ";




//------------------ TKSpaceFirstOrder3DSolver module Print code version ----------------//
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_BuildNoDateTime =
    "+---------------------------------------------------------------+\n"
    "|                       Build information                       |\n"
    "+---------------------------------------------------------------+\n"
    "| Build Number:     kspaceFirstOrder3D v3.4                     |\n"
    "| Build date:       %*.*s                                 |\n"
    "| Build time:       %*.*s                                    |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_VersionGitHash =
    "| Git hash:         %s    |\n";

/// KSpaceFirstOrder3DSolver module log message

TOutputMessage OUT_FMT_LinuxBuild    = "| Operating system: Linux x64                                   |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_WindowsBuild  = "| Operating system: Windows x64                                 |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_MacOSBuild    = "| Operating system: Mac OS X x64                                |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_GNUCompiler   = "| Compiler name:    GNU C++ %.19s                               |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_IntelCompiler = "| Compiler name:    Intel C++ %d                              |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_VisualStudioCompiler = "| Compiler name:    Visual Studio C++ %d                      |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_AVX2  = "| Instruction set:  Intel AVX 2                                 |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_AVX   = "| Instruction set:  Intel AVX                                   |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_SSE42 = "| Instruction set:  Intel SSE 4.2                               |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_SSE41 = "| Instruction set:  Intel SSE 4.1                               |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_SSE3  = "| Instruction set:  Intel SSE 3                                 |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_SSE2  = "| Instruction set:  Intel SSE 2                                 |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDARuntimeNA = "| CUDA Runtime:     N/A                                         |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDARuntime   = "| CUDA Runtime:     %d.%d                                         |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDADriver    = "| CUDA Driver:      %d.%d                                         |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDADeviceInfoNA  =
    "| CUDA code arch:   N/A                                         |\n"
    "+---------------------------------------------------------------+\n"
    "| CUDA device id:   N/A                                         |\n"
    "| CUDA device name: N/A                                         |\n"
    "| CUDA capability:  N/A                                         |\n";


/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDACodeArch    = "| CUDA code arch:   %1.1f                                         |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDADevice      = "| CUDA device id:   %d                                           |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDADeviceName  = "| CUDA device name: %s %.*s|\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDADeviceNamePadding =  "                                        ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_CUDADCapability = "| CUDA capability:  %d.%d                                         |\n";


/// KSpaceFirstOrder3DSolver module log message
TOutputMessage OUT_FMT_Licence =
    "+---------------------------------------------------------------+\n"
    "| Contact email:    jarosjir@fit.vutbr.cz                       |\n"
    "| Contact web:      http://www.k-wave.org                       |\n"
    "+---------------------------------------------------------------+\n"
    "|       Copyright (C) 2016 Jiri Jaros and Bradley Treeby        |\n"
    "+---------------------------------------------------------------+\n";


//------------ TCommandlineParamerers module Print code version --------------//
/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_InputFile  = "Input file:  ";

/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_OutputFile = "Output file: ";

/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_CheckpointFile = "Check file:  ";

/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_CheckpointInterval     = "| Checkpoint interval:                                %8lus |\n";
/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_CompressionLevel       = "| Compression level:                                   %8lu |\n";
/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_PrintProgressInterval  = "| Print progress interval:                            %8lu%% |\n";
/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_BenchmarkTimeStepCount = "| Benchmark time steps:                                %8lu |\n";

/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_QuantitySampling       =
        "+---------------------------------------------------------------+\n"
        "|                        Sampling flags                         |\n"
        "+---------------------------------------------------------------+\n";

/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_SamplingBeginsAt     = "| Sampling begins at time step:                        %8lu |\n";
/// ComandlineParamerers module log message
TOutputMessage OUT_FMT_CopySensorMaskYes    = "| Copy sensor mask to output file:                          Yes |\n";


//------------------------------ Usage ----------------------------------------//
/// Usage log massage
TOutputMessage OUT_FMT_UsagePart1 =
        "|                             Usage                             |\n"
        "+---------------------------------------------------------------+\n"
        "|                     Mandatory parameters                      |\n"
        "+---------------------------------------------------------------+\n"
        "| -i <file_name>                | HDF5 input file               |\n"
        "| -o <file_name>                | HDF5 output file              |\n"
        "+-------------------------------+-------------------------------+\n"
        "|                      Optional parameters                      |\n"
        "+-------------------------------+-------------------------------+\n";

/// Usage log massage
TOutputMessage OUT_FMT_UsagePart2 =
        "| -g <device_number>            | GPU device to run on          |\n"
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

/// Usage log massage
TOutputMessage OUT_FMT_UsageThreads =
        "| -t <num_threads>              | Number of CPU threads         |\n"
        "|                               |  (default = %2d)               |\n";

#endif /* OUTPUT_MESSAGES_WINDOWS_H */

