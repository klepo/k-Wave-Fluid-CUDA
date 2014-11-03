/**
 * @File        TimeMeasure.h
 * @Author      Jiri Jaros
 * @Affiliation CECS,ANU, Australia
 * @Email       jiri.jaros@anu.edu.au
 *
 * @Comments    Class with time measurement
 *
 * @Tool        kspaceFirstOrder3D 2.13
 * @Created     15 August 2012, 9:35 AM
 * @LastModif   17 September 2012, 15:35PM
 *
 * @License:
 * This file is part of the C++ extension of the k-Wave Toolbox (http://www.k-wave.org)
 * Copyright (C) 2012 Jiri Jaros and Bradley Treeby
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it
 * and/or modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the License,
 * or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
 *  */


#ifndef TIMEMEASURE_H
#define	TIMEMEASURE_H

#include <omp.h>

/**
 * @class  Time Messsure
 */
class TTimeMeasure{
    private: 
        /// Start timestamp of the interval
        double StartTime;
        /// Stop timestamp of the interval
        double StopTime;
        /// Elapsed time in previous simulation legs
        double CumulatedElapsedTimeOverPreviousLegs;
    
    public:
        ///Default constructor
        TTimeMeasure() {
            StartTime = 0.0;
            StopTime = 0.0;
            CumulatedElapsedTimeOverPreviousLegs = 0.0;
        };

        /**
         * @brief Copy constructor
         * @param [in] src - the other class to copy from
         */
        TTimeMeasure(const TTimeMeasure & orig) {
            StartTime = orig.StartTime;
            StopTime  = orig.StopTime;
            CumulatedElapsedTimeOverPreviousLegs = orig.CumulatedElapsedTimeOverPreviousLegs;
        };

        /**
         * @brief operator =
         * @param [in] src - source
         * @return
         */
        TTimeMeasure& operator = (const TTimeMeasure & orig){
            if (this != &orig){
                StartTime = orig.StartTime;
                StopTime  = orig.StopTime;
                CumulatedElapsedTimeOverPreviousLegs =
                    orig.CumulatedElapsedTimeOverPreviousLegs;
            }
            return *this;
        };

        ///Destructor
        virtual ~TTimeMeasure() {};

        ///Get start timestamp
        void Start(){
            StartTime = omp_get_wtime();
        };

        ///Get the stop timestamp
        void Stop(){
            StopTime = omp_get_wtime();
        };

        /**
         * Get elapsed time
         * @return elapsed time between start timestamp and stop timestamp
         */
        double GetElapsedTime() const {
            return StopTime - StartTime;
        };
        
        /**
         * Get cumulated elapsed time over all simulation legs
         * @return
         */
        double GetCumulatedElapsedTimeOverAllLegs() const
        {
            return CumulatedElapsedTimeOverPreviousLegs + (StopTime - StartTime);
        };

        /**
         * Get time spent in previous legs
         * @return
         */
        double GetCumulatedElapsedTimePreviousAllLegs() const
        {
            return CumulatedElapsedTimeOverPreviousLegs;
        };

        /**
         * @brief Set elapsed time in previous legs of the simulation
         * @param [in] ElapsedTime
         */
        void SetCumulatedElapsedTimeOverPreviousLegs(const double ElapsedTime)
        {
            CumulatedElapsedTimeOverPreviousLegs = ElapsedTime;
        }

};// end of TTimeMeasure
//------------------------------------------------------------------------------

#endif	/* TIMEMEASURE_H */
