/**
 * @file      CompressHelper.cpp
 *
 * @author    Petr Kleparnik \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            ikleparnik@fit.vutbr.cz
 *
 * @brief     The implementation file containing CompressHelper class definition.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      8  September 2016, 12:00 (created) \n
 *            08 February  2023, 12:00 (revised)
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

#include <Compression/CompressHelper.h>
#include <immintrin.h>

// Initialization of the singleton instance flag
bool CompressHelper::sCompressHelperInstanceFlag = false;

// Initialization of the instance
CompressHelper* CompressHelper::sCompressHelperInstance = nullptr;

/**
 * @brief Initialization
 * @param[in] period Period
 * @param[in] mos Multiple of overlap size
 * @param[in] harmonics Number of harmonics
 * @param[in] normalize Normalizes basis functions for compression (optional)
 */
void CompressHelper::init(float period, hsize_t mos, hsize_t harmonics, bool normalize)
{
  mOSize       = hsize_t(period * mos);
  mBSize       = mOSize * 2 + 1;
  mPeriod      = period;
  mMos         = mos;
  mHarmonics   = harmonics;
  mB           = new float[mBSize]();
  mE           = new FloatComplex[harmonics * mBSize]();
  mEShifted    = new FloatComplex[harmonics * mBSize]();
  mBE          = new FloatComplex[harmonics * mBSize]();
  mBEShifted   = new FloatComplex[harmonics * mBSize]();
  mBE_1        = new FloatComplex[harmonics * mBSize]();
  mBE_1Shifted = new FloatComplex[harmonics * mBSize]();

  generateFunctions(mBSize, mOSize, period, harmonics, mB, mE, mBE, mBE_1, normalize);
  generateFunctions(mBSize, mOSize, period, harmonics, mB, mEShifted, mBEShifted, mBE_1Shifted, normalize, true);
}

/**
 * @brief Destructor
 */
CompressHelper::~CompressHelper()
{
  if (mB)
  {
    delete[] mB;
  }
  mB = nullptr;

  if (mE)
  {
    delete[] mE;
  }
  mE = nullptr;

  if (mEShifted)
  {
    delete[] mEShifted;
  }
  mEShifted = nullptr;

  if (mBE)
  {
    delete[] mBE;
  }
  mBE = nullptr;

  if (mBEShifted)
  {
    delete[] mBEShifted;
  }
  mBEShifted = nullptr;

  if (mBE_1)
  {
    delete[] mBE_1;
  }
  mBE_1 = nullptr;

  if (mBE_1Shifted)
  {
    delete[] mBE_1Shifted;
  }
  mBE_1Shifted = nullptr;

  sCompressHelperInstanceFlag = false;
  if (sCompressHelperInstance)
  {
    delete sCompressHelperInstance;
  }
  sCompressHelperInstance = nullptr;
}

/**
 * @brief Get instance of singleton class
 * @return Instance of singleton class
 */
CompressHelper& CompressHelper::getInstance()
{
  if (!sCompressHelperInstanceFlag)
  {
    sCompressHelperInstance     = new CompressHelper();
    sCompressHelperInstanceFlag = true;
    return *sCompressHelperInstance;
  }
  else
  {
    return *sCompressHelperInstance;
  }
}

/**
 * @brief Computes period from data
 * @param[in] dataSrc Source data
 * @param[in] length Source length
 * @return Period
 */
float CompressHelper::findPeriod(const float* dataSrc, hsize_t length)
{
  float* dataTmp  = new float[length]();
  float* peaksTmp = new float[length]();
  float* locsTmp  = new float[length]();
  hsize_t peaksCount;
  float period;

  // xcorr(dataSrc, dataSrc, dataTmp, length, length);
  findPeaks(dataSrc, locsTmp, peaksTmp, length, peaksCount);

  float* newLocsTmp = new float[peaksCount]();
  float* newLocs    = new float[peaksCount]();

  // Find max peak
  float m = std::numeric_limits<float>::min();
  for (hsize_t i = 0; i < peaksCount; i++)
  {
    if (peaksTmp[i] > m)
    {
      m = peaksTmp[i];
    }
  }

  // Filter peaks under 0.5 * max
  hsize_t j = 0;
  for (hsize_t i = 0; i < peaksCount; i++)
  {
    if (peaksTmp[i] > 0.5f * m)
    {
      newLocsTmp[j] = locsTmp[i];
      j++;
    }
  }

  float* locs = new float[j - 1]();
  diff(newLocsTmp, locs, j);
  period = median(locs, j - 1);

  if (dataTmp)
  {
    delete[] dataTmp;
    dataTmp = nullptr;
  }
  if (peaksTmp)
  {
    delete[] peaksTmp;
    peaksTmp = nullptr;
  }
  if (locsTmp)
  {
    delete[] locsTmp;
    locsTmp = nullptr;
  }
  if (newLocsTmp)
  {
    delete[] newLocsTmp;
    newLocsTmp = nullptr;
  }
  if (newLocs)
  {
    delete[] newLocs;
    newLocs = nullptr;
  }
  if (locs)
  {
    delete[] locs;
    locs = nullptr;
  }
  return period;
}

/**
 * @brief Convert 40-bit coded complex number to float complex number
 * @param[in] iValues 40-bit coded complex number
 * @param[out] cValue float complex number
 * @param[in] e Exponent constant, usually 138 for acoustic pressure and 114 for particle velocity
 */
void CompressHelper::convert40bToFloatC(uint8_t* iValues, FloatComplex& cValue, const int32_t e)
{
  // Get mantissas, signs and exponent
  uint32_t mR = (*reinterpret_cast<uint8_t*>(&iValues[0]) & 0x20U) << 11 | *reinterpret_cast<uint16_t*>(&iValues[1]);
  uint32_t mI = (*reinterpret_cast<uint8_t*>(&iValues[0]) & 0x10U) << 12 | *reinterpret_cast<uint16_t*>(&iValues[3]);
  uint32_t sR = *reinterpret_cast<uint8_t*>(&iValues[0]) >> 7;
  uint32_t sI = (*reinterpret_cast<uint8_t*>(&iValues[0]) & 0x40) >> 6;
  uint8_t eS  = (*reinterpret_cast<uint8_t*>(&iValues[0]) & 0xF);
  // Add 6 bits, now we have 23 bit mantissas
  mR <<= 6;
  mI <<= 6;
  // Add e constant (138 or 114)
  int32_t eR = eS + e;
  int32_t eI = eR;
  // Zero mantissa means zero float number
  if (mR != 0)
  {
// Find index of most left one bit
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    int index = 0;
    index     = 31 ^ __builtin_clz(mR);
#elif defined _WIN32
    unsigned long index = 0;
    _BitScanReverse(&index, mR);
#else
    uint32_t index = 0;
    _BitScanReverse(&index, mR);
#endif
    // Shift left by the index
    mR <<= 23 - index;
    // Recompute final exponent by the index
    eR -= 22 - index;
  }
  else
  {
    eR = 0;
  }

  if (mI != 0)
  {
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    int index = 0;
    index     = 31 ^ __builtin_clz(mI);
#elif defined _WIN32
    unsigned long index = 0;
    _BitScanReverse(&index, mI);
#else
    uint32_t index = 0;
    _BitScanReverse(&index, mI);
#endif
    mI <<= 23 - index;
    eI -= 22 - index;
  }
  else
  {
    eI = 0;
  }
  uint32_t ccR = (sR << 31) | (eR << 23) | (mR & 0x007fffff);
  uint32_t ccI = (sI << 31) | (eI << 23) | (mI & 0x007fffff);
  cValue       = FloatComplex(*reinterpret_cast<float*>(&ccR), *reinterpret_cast<float*>(&ccI));
}

/**
 * @brief Convert float complex number to 40-bit coded complex number
 * @param[in] cValue float complex number
 * @param[out] iValues 40-bit coded complex number
 * @param[in] e Exponent constant, usually 138 for acoustic pressure and 114 for particle velocity
 */
void CompressHelper::convertFloatCTo40b(FloatComplex cValue, uint8_t* iValues, const int32_t e)
{
  // Get real and imaginary part
  float cR    = cValue.real();
  float cI    = cValue.imag();
  uint32_t mR = *reinterpret_cast<uint32_t*>(&cR);
  uint32_t mI = *reinterpret_cast<uint32_t*>(&cI);
  // Get 1-bit signs
  uint8_t sR  = mR >> 31;
  uint8_t sI  = mI >> 31;
  // e = 138 p, max exponent is 2^26 (15 + 138 = 153, 153 - 127 = 26)
  // p max value is pow(2, 26 - 16) * 0x1FFFF = 134216704
  // p min value is pow(2, 26 - 16 - 15) * 0x1 = 0.0312500000
  // e = 114 u, max exponent is 2^2  (15 + 114 = 129, 129 - 127 = 2)
  // u max value is pow(2, 2 - 16) * 0x1FFFF = 7.99993896484375
  // u max value is pow(2, 2 - 16 - 15) * 0x1 = 0.000000001862645149230957031250
  //  Get 8-bit exponents, subtracts e, (exponent will have 4 bits, values from 0 to 15)
  int32_t eRS = ((mR & 0x7f800000) >> 23) - e;
  int32_t eIS = ((mI & 0x7f800000) >> 23) - e;
  int32_t eS  = eRS;
  // Get 23-bit mantissas
  mR          = (mR & 0x007fffff);
  mI          = (mI & 0x007fffff);
  // Right shifts of real and imaginary part, we drop 6 bits, the mantissa will have 16 + 1 bits
  uint8_t rSR = 6;
  uint8_t rSI = 6;
  // Find the higher exponent, smaller part will be shifted to right
  if (eRS > eIS)
  {
    // Add right shift of imaginary part
    rSI += (eRS - eIS);
    eS = eRS;
  }
  else if (eIS > eRS)
  {
    // Add right shift of real part
    rSR += (eIS - eRS);
    eS = eIS;
  }
  // Crop small values
  if (eS < 0)
  {
    // Shift back to left
    rSR += -(eS);
    rSI += -(eS);
    // Exponent will be 0
    eS = 0;
  }
  // Shift overflow
  if (rSR > 23)
  {
    rSR = 23;
  }
  if (rSI > 23)
  {
    rSI = 23;
  }
  // Shift right
  mR >>= rSR;
  mI >>= rSI;
  // Rounding
  if (mR > 0)
  {
    // Check possible overflow
    if (mR != (0x7FFFFFU >> rSR))
    {
      mR++;
    }
  }
  if (mI > 0)
  {
    if (mI != (0x7FFFFFU >> rSI))
    {
      mI++;
    }
  }
  // Set 1 left flag bit
  mR |= 1UL << (23 - rSR);
  // Align to 17 bits
  mR >>= 1;
  mI |= 1UL << (23 - rSI);
  mI >>= 1;

  // Exponent overflow, set maximum values
  if (eS > 0xF)
  {
    mR = 0xFFFF;
    mI = 0xFFFF;
    eS = 0xF;
  }
  // bits: | 1         | 1              | 17            | 17                 | 4                |
  //       | real sign | imaginary sign | real mantissa | imaginary mantissa | shifted exponent |
  // Mantissa is composed from: 0-16 zero bits, 1 flag bit, 0-16 data (mantissa or fraction) bits
  // Number of zero bits means exponent shift from the stored exponent eS
  iValues[0] = (sR << 7) | (sI << 6) | ((mR & 0x10000) >> 11) | ((mI & 0x10000) >> 12) | (eS & 0xF);
  *reinterpret_cast<uint16_t*>(&iValues[1]) = mR;
  *reinterpret_cast<uint16_t*>(&iValues[3]) = mI;
}

/**
 * @brief Returns complex exponential window basis
 * @return Complex exponential window basis
 */
const FloatComplex* CompressHelper::getBE() const
{
  return mBE;
}

/**
 * @brief Returns shifted complex exponential window basis
 * @return Shifted complex exponential window basis
 */
const FloatComplex* CompressHelper::getBEShifted() const
{
  return mBEShifted;
}

/**
 * @brief Returns inverted complex exponential window basis
 * @return Inverted complex exponential window basis
 */
const FloatComplex* CompressHelper::getBE_1() const
{
  return mBE_1;
}

/**
 * @brief Returns inverted shifted complex exponential window basis
 * @return Inverted shifted complex exponential window basis
 */
const FloatComplex* CompressHelper::getBE_1Shifted() const
{
  return mBE_1Shifted;
}

/**
 * @brief Returns overlap size
 * @return Overlap size
 */
hsize_t CompressHelper::getOSize() const
{
  return mOSize;
}

/**
 * @brief Return base size
 * @return Base size
 */
hsize_t CompressHelper::getBSize() const
{
  return mBSize;
}

/**
 * @brief Returns period
 * @return Period
 */
float CompressHelper::getPeriod() const
{
  return mPeriod;
}

/**
 * @brief Returns multiple of overlap size
 * @return Multiple of overlap size
 */
hsize_t CompressHelper::getMos() const
{
  return mMos;
}

/**
 * @brief Return number of harmonics
 * @return Number of harmonics
 */
hsize_t CompressHelper::getHarmonics() const
{
  return mHarmonics;
}

/**
 * @brief Constructor
 */
CompressHelper::CompressHelper()
{
}

/**
 * @brief Cross correlation
 * @param[in] dataSrc1 Source data 1
 * @param[in] dataSrc2 Source data 2
 * @param[out] dataDst Destination
 * @param[in] lengthSrc1 Source length 1
 * @param[in] lengthSrc2 Source length 2
 */
void CompressHelper::xcorr(
  const float* dataSrc1, const float* dataSrc2, float* dataDst, hsize_t lengthSrc1, hsize_t lengthSrc2)
{
  hsize_t i, j;
  hssize_t i1;

  for (i = 0; i < lengthSrc1 + lengthSrc2 - 1; i++)
  {
    dataDst[i] = 0.0f;
    i1         = hssize_t(i);
    float tmp  = 0.0f;
    for (j = 0; j < lengthSrc2; j++)
    {
      if (i1 >= 0 && i1 < hssize_t(lengthSrc1))
      {
        tmp = tmp + (dataSrc1[i1] * dataSrc2[lengthSrc2 - 1 - j]);
      }
      i1         = i1 - 1;
      dataDst[i] = tmp;
    }
  }
}

/**
 * @brief Convolution
 * @param[in] dataSrc1 Source data 1
 * @param[in] dataSrc2 Source data 2
 * @param[out] dataDst Destination
 * @param[in] lengthSrc1 Source length 1
 * @param[in] lengthSrc2 Source length 2
 */
void CompressHelper::conv(
  const float* dataSrc1, const float* dataSrc2, float* dataDst, hsize_t lengthSrc1, hsize_t lengthSrc2)
{
  hsize_t i, j;
  hssize_t i1;

  for (i = 0; i < lengthSrc1 + lengthSrc2 - 1; i++)
  {
    dataDst[i] = 0.0f;
    i1         = hssize_t(i);
    float tmp  = 0.0f;
    for (j = 0; j < lengthSrc2; j++)
    {
      if (i1 >= 0 && i1 < hssize_t(lengthSrc1))
      {
        tmp = tmp + (dataSrc1[i1] * dataSrc2[j]);
      }
      i1         = i1 - 1;
      dataDst[i] = tmp;
    }
  }
}

/**
 * @brief Finds peaks in source signal
 * @param[in] dataSrc Source data
 * @param[out] locsDst Destination for locations
 * @param[out] peaksDst Destination for peaks
 * @param[in] lengthSrc Source length
 * @param[out] lengthDst Destination length
 */
void CompressHelper::findPeaks(
  const float* dataSrc, float* locsDst, float* peaksDst, hsize_t lengthSrc, hsize_t& lengthDst)
{
  for (hsize_t i = 0; i < lengthSrc; i++)
  {
    locsDst[i]  = 0.0f;
    peaksDst[i] = 0.0f;
  }

  lengthDst = 0;

  for (hsize_t i = 0; i < lengthSrc; i++)
  {
    // Peaks between
    if (i > 0 && i < lengthSrc - 1 && lengthSrc > 2 && dataSrc[i] > dataSrc[i - 1] && dataSrc[i] >= dataSrc[i + 1])
    {
      float d1            = dataSrc[i] - dataSrc[i - 1];
      float d2            = dataSrc[i] - dataSrc[i + 1];
      locsDst[lengthDst]  = i + d1 / (d1 + d2) - 0.5f;
      peaksDst[lengthDst] = dataSrc[i];
      lengthDst++;
    }
  }
}

/**
 * @brief Differences
 * @param[in] dataSrc Source data
 * @param[out] dataDst Destination
 * @param[in] length Source length
 */
void CompressHelper::diff(const float* dataSrc, float* dataDst, hsize_t length)
{
  for (hsize_t i = 0; i < length - 1; i++)
  {
    dataDst[i] = dataSrc[i + 1] - dataSrc[i];
  }
}

/**
 * @brief Differences
 * @param[in] dataSrc Source data
 * @param[out] dataDst Destination
 * @param[in] length Source length
 */
void CompressHelper::diff(const hsize_t* dataSrc, hsize_t* dataDst, hsize_t length)
{
  for (hsize_t i = 0; i < length - 1; i++)
  {
    dataDst[i] = dataSrc[i + 1] - dataSrc[i];
  }
}

/**
 * @brief Mean
 * @param[in] dataSrc Source data
 * @param[in] length Source length
 * @return Mean
 */
float CompressHelper::mean(const float* dataSrc, hsize_t length)
{
  float sum = 0.0f;
  for (hsize_t i = 0; i < length; i++)
  {
    sum += dataSrc[i];
  }
  return sum / length;
}

/**
 * @brief Mean
 * @param[in] dataSrc Source data
 * @param[in] length Source length
 * @return Mean
 */
hsize_t CompressHelper::mean(const hsize_t* dataSrc, hsize_t length)
{
  float sum = 0.0f;
  for (hsize_t i = 0; i < length; i++)
  {
    sum += dataSrc[i];
  }
  return hsize_t(round(sum / length));
}

/**
 * @brief Median
 * @param[in] dataSrc Source data
 * @param[in] length Source length
 * @return Median
 */
float CompressHelper::median(const float* dataSrc, hsize_t length)
{
  std::vector<float> dataSrcVector(dataSrc, dataSrc + length);
  std::sort(dataSrcVector.begin(), dataSrcVector.end());
  return dataSrcVector[size_t(length / 2)];
}

/**
 * @brief Median
 * @param[in] dataSrc Source data
 * @param[in] length Source length
 * @return Median
 */
hsize_t CompressHelper::median(const hsize_t* dataSrc, hsize_t length)
{
  std::vector<hsize_t> dataSrcVector(dataSrc, dataSrc + length);
  std::sort(dataSrcVector.begin(), dataSrcVector.end());
  return dataSrcVector[size_t(length / 2)];
}

/**
 * @brief Generates basis function values
 * @param[in] bSize Base size
 * @param[in] oSize Overlap size
 * @param[in] period Period
 * @param[in] harmonics Harmonics
 * @param[out] b Window basis
 * @param[out] e Exponential basis
 * @param[out] bE Complex exponential window basis
 * @param[out] bE_1 Inverted complex exponential window basis
 * @param[in] normalize Normalization flag (optional)
 */
void CompressHelper::generateFunctions(hsize_t bSize,
  hsize_t oSize,
  float period,
  hsize_t harmonics,
  float* b,
  FloatComplex* e,
  FloatComplex* bE,
  FloatComplex* bE_1,
  bool normalize,
  bool shift) const
{
  // Generate basis function (window)
  triangular(oSize, b); // Triangular window
  // hann(oSize, b);        // Hann window

  // Generate complex exponential window basis functions
  for (hsize_t ih = 0; ih < harmonics; ih++)
  {
    generateE(period, ih, ih + 1, bSize, e, shift);
    generateBE(ih, bSize, oSize, b, e, bE, bE_1, normalize);
  }
}

/**
 * @brief Generates triangular window
 * @param[in] oSize Overlap size
 * @param[out] w Window
 */
void CompressHelper::triangular(hsize_t oSize, float* w) const
{
  for (hsize_t x = 0; x < oSize; x++)
  {
    w[x] = float(x) / oSize;
  }
  for (hsize_t x = oSize; x < 2 * oSize + 1; x++)
  {
    w[x] = 2.0f - float(x) / oSize;
  }
}

/**
 * @brief Generates Hann window
 * @param[in] oSize Overlap size
 * @param[out] w Window
 */
void CompressHelper::hann(hsize_t oSize, float* w) const
{
  for (hsize_t x = 0; x < 2 * oSize + 1; x++)
  {
    w[x] = float(pow(sin(float(M_PI) * x / (2.0f * oSize)), 2));
  }
}

/**
 * @brief Generates complex exponential basis
 * @param[in] period Period
 * @param[in] ih Harmonics index
 * @param[in] h Harmonic multiple
 * @param[in] bSize Base size
 * @param[out] e Exponential basis
 */
void CompressHelper::generateE(float period, hsize_t ih, hsize_t h, hsize_t bSize, FloatComplex* e, bool shift) const
{
  FloatComplex i(0.0f, -1.0f);
  for (hsize_t x = 0; x < bSize; x++)
  {
    hsize_t hx = ih * bSize + x;
    e[hx]      = std::exp(i * (2.0f * float(M_PI) / (period / float(h))) * float(x));
    if (shift)
    {
      e[hx] *= std::exp(-i * float(M_PI) / (period / float(h)));
    }
  }
}

/**
 * @brief Generates complex exponential window basis
 * @param[in] ih Harmonics index
 * @param[in] bSize Base size
 * @param[in] oSize Overlap size
 * @param[in] b Window basis
 * @param[in] e Exponential basis
 * @param[out] bE Complex exponential window basis
 * @param[out] bE_1 Inverted complex exponential window basis
 * @param[in] normalize Normalization flag (optional)
 */
void CompressHelper::generateBE(hsize_t ih,
  hsize_t bSize,
  hsize_t oSize,
  const float* b,
  const FloatComplex* e,
  FloatComplex* bE,
  FloatComplex* bE_1,
  bool normalize) const
{
  for (hsize_t x = 0; x < bSize; x++)
  {
    hsize_t hx = ih * bSize + x;
    bE[hx]     = b[x] * e[hx];
    bE_1[hx]   = b[(x + oSize) % (bSize - 1)] * e[ih * bSize + ((x + oSize) % (bSize - 1))];
    if (normalize)
    {
      bE[hx] *= (2.0f / float(oSize));
      bE_1[hx] *= (2.0f / float(oSize));
    }
  }
}
