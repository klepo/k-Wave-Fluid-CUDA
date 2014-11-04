/**
 * @file        MemoryMeasure.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file for memory measure class.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        04 November 2014, 17:30 (revised) \n
 *              04 November 2014, 17:30 (revised)
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

#ifndef MEMORYMEASURE_H
#define MEMORYMEASURE_H

#if DEBUG_MEMORY_USAGE && CUDA_VERSION
#define LogPreAlloc(){PreProcessorLogPreAlloc();}
#define LogPostAlloc(){PreProcessorLogPostAlloc();}
#define LogAlloc(size_in_bytes){PreProcessorLogAlloc(size_in_bytes);}

#include <sys/resource.h>
#include <cuda_runtime.h>
#include <iostream>

inline void PreProcessorLogPreAlloc()
{

    struct rusage mem_usage;
    size_t free, total;

    getrusage(RUSAGE_SELF, &mem_usage);
    cudaMemGetInfo(&free,&total);

    cout << "\n********************" << endl;
    cout << "Pre alloc" << endl;
    cout << "CPU memory (MB) = " << (mem_usage.ru_maxrss >> 10) << endl;
    cout << "GPU memory (MB) = " << ((total-free) >> 20) << endl;
}

inline void PreProcessorLogPostAlloc()
{

    struct rusage mem_usage;
    size_t free, total;

    getrusage(RUSAGE_SELF, &mem_usage);
    cudaMemGetInfo(&free,&total);

    cout << "Post alloc" << endl;
    cout << "CPU memory (MB) = " << (mem_usage.ru_maxrss >> 10) << endl;
    cout << "GPU memory (MB) = " << ((total-free) >> 20) << endl;
    cout << "********************\n" << endl;
}

inline void PreProcessorLogAlloc(size_t size_in_bytes)
{
    cout << "allocating " <<  (size_in_bytes >> 20) << " MB" << endl;
}

#else
#define LogPreAlloc()
#define LogPostAlloc()
#define LogAlloc(size_in_bytes)
#endif



#endif

