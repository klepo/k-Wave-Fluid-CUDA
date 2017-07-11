/**
 * @file        TimeMeasure.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file for class with time measurement
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        15 August   2012, 09:35 (created) \n
 *              11 July     2017, 11:59 (revised)
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


#ifndef TimeMeasureH
#define	TimeMeasureH

#ifdef _OPENMP
  #include <omp.h>
#else
  // Linux build
  #ifdef __linux__
    #include <sys/time.h>
  #endif
  // Windows build
  #ifdef _WIN64
    #include <Windows.h>
    #include <time.h>
  #endif
#endif

/**
 * @class  TimeMeasure
 * @brief  Class measuring elapsed time.
 * @brief  Class measuring elapsed time, even over multiple simulation legs.
 */
class TimeMeasure
{
  public:

    /// Default constructor.
    TimeMeasure() :
      mStartTime(0.0),
      mStopTime(0.0),
      mElapsedTimeOverPreviousLegs(0.0)
    { };

    /// Destructor.
    virtual ~TimeMeasure() {};

    /**
     * @brief   Copy constructor.
     * @details Copy constructor.
     * @param [in] src - The other class to copy from
     */
    TimeMeasure(const TimeMeasure& src) :
      mStartTime(src.mStartTime),
      mStopTime (src.mStopTime),
      mElapsedTimeOverPreviousLegs(src.mElapsedTimeOverPreviousLegs)
    { };

    /**
     * @brief  operator=
     * @details operator=
     * @param [in] src - source
     * @return
     */
    TimeMeasure& operator=(const TimeMeasure& src)
    {
      if (this != &src)
      {
        mStartTime = src.mStartTime;
        mStopTime  = src.mStopTime;
        mElapsedTimeOverPreviousLegs = src.mElapsedTimeOverPreviousLegs;
      }
      return *this;
    };

    /// Get start timestamp.
    inline void start()
    {
      #ifdef _OPENMP
        mStartTime = omp_get_wtime();
      #else
        // Linux build
        #ifdef __linux__
          timeval actTime;
          gettimeofday(&actTime, nullptr);
          mStartTime = actTime.tv_sec + actTime.tv_usec * 1.0e-6;
        #endif
        #ifdef _WIN64
          mStartTime = clock() / (double) CLOCKS_PER_SEC;
        #endif
      #endif
    };

    /// Get stop timestamp.
    inline void stop()
    {
      #ifdef _OPENMP
        mStopTime = omp_get_wtime();
      #else
        // Linux build
        #ifdef __linux__
          timeval actTime;
          gettimeofday(&actTime, nullptr);
          mStopTime = actTime.tv_sec + actTime.tv_usec * 1.0e-6;
        #endif
        #ifdef _WIN64
          mStopTime = clock() / (double) CLOCKS_PER_SEC;
        #endif
      #endif
    };

    /**
     * @brief Get elapsed time.
     * @details Get elapsed time.
     * @return elapsed time between start timestamp and stop timestamp.
     */
    inline double getElapsedTime() const
    {
      return mStopTime - mStartTime;
    };

    /**
     * @brief Get cumulated elapsed time over all simulation legs.
     * @details Get cumulated elapsed time over all simulation legs.
     * @return elapsed time all (including this one) legs.
     */
    inline double getElapsedTimeOverAllLegs() const
    {
      return mElapsedTimeOverPreviousLegs + (mStopTime - mStartTime);
    };

    /**
     * @brief Get time spent in previous legs.
     * @detail Get time spent in previous legs.
     * @return elapsed time over previous legs.
     */
    inline double getElapsedTimeOverPreviousLegs() const
    {
      return mElapsedTimeOverPreviousLegs;
    };

    /**
     * @brief Set elapsed time in previous legs of the simulation.
     * @details Set elapsed time in previous legs of the simulation.
     * @param [in] elapsedTime - Elapsed time
     */
    void SetElapsedTimeOverPreviousLegs(const double elapsedTime)
    {
      mElapsedTimeOverPreviousLegs = elapsedTime;
    }

  private:
    /// Start timestamp of the interval
    double mStartTime;
    /// Stop timestamp of the interval
    double mStopTime;
    /// Elapsed time in previous simulation legs
    double mElapsedTimeOverPreviousLegs;

};// end of TTimeMeasure
//----------------------------------------------------------------------------------------------------------------------

#endif	/* TimeMeasureH */
