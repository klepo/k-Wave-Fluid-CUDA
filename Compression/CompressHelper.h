/**
 * @file      CompressHelper.h
 *
 * @author    Petr Kleparnik \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            ikleparnik@fit.vutbr.cz
 *
 * @brief     The header file containing CompressHelper class declaration.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      8  September 2016, 12:00 (created) \n
 *            08 February  2023, 12:00 (revised)
 *
 * @copyright Copyright (C) 2019 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave
 * Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can
 * redistribute it and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with k-Wave. If not, see
 * [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#ifndef COMPRESSHELPER_H
#define COMPRESSHELPER_H

#include <iostream>
#include <string>
#include <algorithm> // std::sort

//#define _USE_MATH_DEFINES // for C++
#ifndef M_PI
  /// M_PI definition
  #define M_PI 3.14159265358979323846
#endif

#ifndef FLT_EPSILON
  #define FLT_EPSILON __FLT_EPSILON__
#endif

#include <cmath>
#include <complex>
#include <omp.h>
#include <vector>

/// Float complex datatype
using FloatComplex = std::complex<float>;
/// Unsigned long long datatype
using hsize_t      = unsigned long long;
/// Long long datatype
using hssize_t     = long long;

const std::string kCompressSuffix = "_c";

/**
 * @brief The CompressHelper class represents wrapper for the ultrasound signals compression
 */
class CompressHelper
{
  public:
    void init(float period, hsize_t mos, hsize_t harmonics, bool normalize = false);
    ~CompressHelper();
    static CompressHelper& getInstance();

    static float findPeriod(const float* dataSrc, hsize_t length);
    static void convert40bToFloatC(uint8_t* iValues, FloatComplex& cValue, const int32_t e);
    static void convertFloatCTo40b(FloatComplex cValue, uint8_t* iValues, const int32_t e);

    const FloatComplex* getBE() const;
    const FloatComplex* getBEShifted() const;
    const FloatComplex* getBE_1() const;
    const FloatComplex* getBE_1Shifted() const;
    hsize_t getOSize() const;
    hsize_t getBSize() const;
    float getPeriod() const;
    hsize_t getMos() const;
    hsize_t getHarmonics() const;

    static const int kMaxExpP = 138;
    static const int kMaxExpU = 114;

  private:
    CompressHelper();
    CompressHelper(const CompressHelper&);
    CompressHelper& operator=(const CompressHelper&);

    static void xcorr(
      const float* dataSrc1, const float* dataSrc2, float* dataDst, hsize_t lengthSrc1, hsize_t lengthSrc2);
    static void conv(
      const float* dataSrc1, const float* dataSrc2, float* dataDst, hsize_t lengthSrc1, hsize_t lengthSrc2);
    static void findPeaks(const float* dataSrc, float* locsDst, float* peaksDst, hsize_t length, hsize_t& lengthDst);
    static void diff(const float* dataSrc, float* dataDst, hsize_t length);
    static void diff(const hsize_t* dataSrc, hsize_t* dataDst, hsize_t length);
    static float mean(const float* dataSrc, hsize_t length);
    static hsize_t mean(const hsize_t* dataSrc, hsize_t length);
    static float median(const float* dataSrc, hsize_t length);
    static hsize_t median(const hsize_t* dataSrc, hsize_t length);

    void generateFunctions(hsize_t bSize,
      hsize_t oSize,
      float period,
      hsize_t harmonics,
      float* b,
      FloatComplex* e,
      FloatComplex* bE,
      FloatComplex* bE_1,
      bool normalize = false,
      bool shift     = false) const;
    void triangular(hsize_t oSize, float* w) const;
    void hann(hsize_t oSize, float* w) const;
    void generateE(float period, hsize_t ih, hsize_t h, hsize_t bSize, FloatComplex* e, bool shift = false) const;
    void generateBE(hsize_t ih,
      hsize_t bSize,
      hsize_t oSize,
      const float* b,
      const FloatComplex* e,
      FloatComplex* bE,
      FloatComplex* bE_1,
      bool normalize = false) const;

    /// Overlap size
    hsize_t mOSize     = 0;
    /// Base size
    hsize_t mBSize     = 0;
    /// Period
    float mPeriod      = 0.0f;
    /// Multiple of overlap size
    hsize_t mMos       = 1;
    /// Number of harmonics
    hsize_t mHarmonics = 1;

    // Memory for helper functions data, 2D arrays for harmonics
    /// Window basis
    float* mB                  = nullptr;
    /// Complex exponential basis
    FloatComplex* mE           = nullptr;
    /// Shifted complex exponential basis
    FloatComplex* mEShifted    = nullptr;
    /// Complex exponential window basis
    FloatComplex* mBE          = nullptr;
    /// Shifted Complex exponential window basis
    FloatComplex* mBEShifted   = nullptr;
    /// Inverted complex exponential window basis
    FloatComplex* mBE_1        = nullptr;
    /// Inverted shifted complex exponential window basis
    FloatComplex* mBE_1Shifted = nullptr;

    /// Singleton flag
    static bool sCompressHelperInstanceFlag;
    /// Singleton instance
    static CompressHelper* sCompressHelperInstance;
};

#endif // COMPRESSHELPER_H
