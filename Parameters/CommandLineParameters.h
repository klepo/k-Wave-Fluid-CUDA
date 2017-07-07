/**
 * @file        CommandLineParameters.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the command line parameters.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        29 August   2012, 11:25 (created) \n
 *              07 July     2017, 19:06 (revised)
 *
 * @section Params Command Line Parameters
 *
 * The CUDA/C++ code requires two mandatory parameters and accepts a few optional parameters and
 * flags. Ill parameters, bad simulation files, and runtime errors such as out-of-memory problems,
 * lead to an exception followed by an error message shown and execution termination.
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
 * The \c <tt>\--benchmark</tt> parameter enables the total length of simulation (i.e., the number
 * of time steps) to be overridden by setting a new number of time  steps to simulate. This is
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
 * the file path was correctly specified ((otherwise you will not find out the simulation crashed
 * until the first leg of the simulation finishes)). The rationale behind this is that to keep as
 * high level of fault tolerance as possible, the checkpoint file should be touched even when really
 * necessary.
 *
 * When controlling a multi-leg simulation by a script loop, the parameters of the code remains the
 * same in all legs. The first leg of the simulation creates a checkpoint  file while the last one
 * deletes it. If the checkpoint file is not found the simulation starts from the beginning. In
 * order to find out how many steps have been finished, please open the output file and read
 * the variable <tt>t_index</tt> and compare it with <tt>Nt</tt> (e.g. by the h5dump command).
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

#ifndef TCOMMAND_LINE_PARAMETERS_H
#define TCOMMAND_LINE_PARAMETERS_H

#include <cstdlib>
#include <string>

/**
 * @class TCommandLineParameters
 * @brief The class to parse and store command line parameters.
 * @details The class to parse and store command line parameters.
 */
class TCommandLineParameters
{
  public:
    /// Only TParameters can create this class.
    friend class TParameters;

    /// Copy constructor not allowed.
    TCommandLineParameters(const TCommandLineParameters&) = delete;

    /// Destructor.
    virtual ~TCommandLineParameters() {};

    /// operator= not allowed.
    TCommandLineParameters& operator= (const TCommandLineParameters&) = delete;

    /// Get input file name.
    const std::string& GetInputFileName()      const {return inputFileName;};
    /// Get output file name.
    const std::string& GetOutputFileName()     const {return outputFileName;};
    /// Get Checkpoint file name.
    const std::string& GetCheckpointFileName() const {return checkpointFileName;};

    /// Get GPU device ID specified by the user (not necessary the one the code runs on).
    int GetCUDADeviceIdx()              const {return cudaDeviceIdx;};

    /// Is --benchmark flag set?
    bool IsBenchmarkFlag()              const {return benchmarkFlag;};
    /// Is --version flag set?
    bool IsVersion()                    const {return printVersion; };
    /// Get benchmark time step count.
    size_t GetBenchmarkTimeStepsCount() const {return benchmarkTimeStepCount;};

    /// Get compression level.
    size_t GetCompressionLevel()        const {return compressionLevel;};
    /// Get number of threads.
    size_t GetNumberOfThreads()         const {return numberOfThreads;};
    /// Get progress print interval.
    size_t GetProgressPrintInterval()   const {return progressPrintInterval;};
    /// Get start time index when sensor data collection begins.
    size_t GetStartTimeIndex()          const {return startTimeStep;};

    /// Is checkpoint enabled?
    bool IsCheckpointEnabled()          const {return (checkpointInterval > 0); };
    /// Get checkpoint interval
    size_t GetCheckpointInterval()      const {return checkpointInterval; };

    /// Is --p_raw set?
    bool IsStore_p_raw()                const {return store_p_raw;};
    /// Is --p_rms set?
    bool IsStore_p_rms()                const {return store_p_rms;};
    /// Is --p_max set?
    bool IsStore_p_max()                const {return store_p_max;};
    /// Is --p_min set?
    bool IsStore_p_min()                const {return store_p_min;};
    /// Is --p_max_all set?
    bool IsStore_p_max_all()            const {return store_p_max_all;};
    /// Is --p_min_all set?
    bool IsStore_p_min_all()            const {return store_p_min_all;};
    /// Is --p_final set?
    bool IsStore_p_final()              const {return store_p_final;};

    /// Is --u_raw set?
    bool IsStore_u_raw()                const {return store_u_raw;};
    /// Is --u_non_staggered_raw set?
    bool IsStore_u_non_staggered_raw()  const {return store_u_non_staggered_raw;};
    /// Is --u_rms set?
    bool IsStore_u_rms()                const {return store_u_rms;};
    /// Is --u_max set?
    bool IsStore_u_max()                const {return store_u_max;};
    /// Is --u_min set?
    bool IsStore_u_min()                const {return store_u_min;};
    /// Is --u_max_all set?
    bool IsStore_u_max_all()            const {return store_u_max_all;};
    /// Is --u_min set?
    bool IsStore_u_min_all()            const {return store_u_min_all;};
    /// Is --u_final set?
    bool IsStore_u_final()              const {return store_u_final;};

    /// Is --copy_mask set set?
    bool IsCopySensorMask()             const {return copySensorMask;};

    /// Print usage..
    void PrintUsage();
    /// Print setup.
    void PrintComandlineParamers();
    /// Parse command line.
    void ParseCommandLine(int argc, char** argv);

  protected:
    /// Default constructor - only friend class can create an instance.
    TCommandLineParameters();

  private:
    /// Input file name.
    std::string inputFileName;
    /// Output file name.
    std::string outputFileName;
    /// Checkpoint file name.
    std::string checkpointFileName;

    /// Number of CPU threads value.
    size_t numberOfThreads;

    /// Id of selected GPU devices.
    int  cudaDeviceIdx;

    /// ProgressInterval value.
    size_t progressPrintInterval;
    /// CompressionLevel value.
    size_t compressionLevel;

    /// BenchmarkFlag value.
    bool   benchmarkFlag;
    /// BenchmarkTimeStepsCount value.
    size_t benchmarkTimeStepCount;
    /// Checkpoint interval in seconds
    size_t checkpointInterval;

    /// print version of the code and exit.
    bool printVersion;

    /// Store_p_raw value.
    bool store_p_raw;
    /// Store_p_rms value.
    bool store_p_rms;
    /// Store_p_max value.
    bool store_p_max;
    /// Store_p_min value.
    bool store_p_min;
    /// Store_p_max_all value.
    bool store_p_max_all;
    /// Store_p_min_all value.
    bool store_p_min_all;
    /// Store_p_final value.
    bool store_p_final;

    /// Store_u_raw value.
    bool store_u_raw;
    /// Store_u_non_staggered_raw value.
    bool store_u_non_staggered_raw;
    /// Store_u_rms value.
    bool store_u_rms;
    /// Store_u_max value.
    bool store_u_max;
    /// Store_u_min value.
    bool store_u_min;
    /// Store_u_max_all value.
    bool store_u_max_all;
    /// Store_u_min_all value.
    bool store_u_min_all;
    /// Store_u_final value.
    bool store_u_final;

    /// Copy sensor mask to the output file.
    bool copySensorMask;
    /// StartTimeStep value.
    size_t startTimeStep;

    /// Default compression level.
    static constexpr size_t DEFAULT_COMPRESSION_LEVEL = 0;
    /// Default progress print interval.
    static constexpr size_t DEFAULT_PROGRESS_PRINT_INTERVAL  = 5;
};// end of class TCommandLineParameters
//--------------------------------------------------------------------------------------------------

#endif /* TCOMMAND_LINE_PARAMETERS_H */

