#ifndef DECONV_H
#define DECONV_H

#include <stdbool.h>

/**
 * Headers for deconvolution algorithm implementation in C++
 * Algorithm equations taken from:
 *   "Development of Iterative Algorithms for Industrial Tomography"
 *   A.V.Likhachov, V.V.Pickalov, N.V.Chugunova, V.A.Baranov
 *   1st World Congress on Industrial Process Tomography, Buxton, Greater Manchester, April 14-17, 1999.
 * Keywords: Algebraic Reconstruction Technique, Kachmazh method.
 */
typedef struct PatternData
{
    float *patternX;
    float *patternY;
    unsigned int ptrnCount;
    float *ptrnMatrix;
    unsigned int innerSigCount;
    int zeroPatternIndex;
    float *norms;
    bool isInitialized;
} PatternData;

void deconv(short *signal, short *ptrnx, short *pattern, int sigCount, int ptrnCount, float lambda, int maxIterations);

/** Fast 1D deconvolution with using preloaded pattern */
int fastDeconv(const PatternData *const patternData, short *signal, int sigCount, float lambda, int maxIterations);

void deconvImage(short *image, short *ptrnx, short *pattern, int imageWidth, int imageHeight, int ptrnCount,
        float lambda, int maxIterations);

/** Load pattern data */
int loadPattern(PatternData *const patternData, short *pattern, short *ptrnx, int ptrnCount, int sigCount);

/** Remove pattern from library memory */
void unloadPattern(PatternData *const patternData);

#endif // DECONV_H
