
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

