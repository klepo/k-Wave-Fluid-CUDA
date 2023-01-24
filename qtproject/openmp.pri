win32 {
    QMAKE_CXXFLAGS += /openmp:llvm
    QMAKE_CFLAGS += /openmp:llvm
}

unix {
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_CFLAGS += -fopenmp
    QMAKE_LFLAGS += -fopenmp
    LIBS += -fopenmp
}
