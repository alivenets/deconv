#include "deconv.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int ALIGN = 4;

static float vecMultiply(float *vec1, float *vec2, unsigned int length);
static float vecNormSqr(float *vec, unsigned int length);
static inline void zeroNegativeElements(float *vec, unsigned int length);
static void innerDeconv(short *innerSignal, float *innerDeconvSignal, float *norms, unsigned int innerSigCount,
        float *ptrnMatrix, float lambda, unsigned int maxIterations);

static inline unsigned int nextMultiply(unsigned int k, unsigned int align)
{
    if (k % align != 0)
        return (k / align) * align + align;
    else
        return k;
}

static inline unsigned int prevMultiply(unsigned int k, unsigned int align)
{
    return (k / align) * align;
}

static inline float vecMultiply(float *vec1, float *vec2, unsigned int length)
{
    float sum = 0;
    unsigned int i = 0;
    for (i = 0; i < length; i += 4)
        sum += vec1[i] * vec2[i] + vec1[i + 1] * vec2[i + 1] + vec1[i + 2] * vec2[i + 2] + vec1[i + 3] * vec2[i + 3];
    return sum;
}

static inline float vecNormSqr(float *vec, unsigned int length)
{
    return vecMultiply(vec, vec, length);
}

static inline void zeroNegativeElements(float *vec, unsigned int length)
{
    for (unsigned int i = 0; i < length; i += 4) {
        if (vec[i] < 0)
            vec[i] = 0;
        if (vec[i + 1] < 0)
            vec[i + 1] = 0;
        if (vec[i + 2] < 0)
            vec[i + 2] = 0;
        if (vec[i + 3] < 0)
            vec[i + 3] = 0;
    }
}

static short normalize(short *img, int len)
{
    int maxVal = img[0];
    for (int i = 1; i < len; i++)
        if (img[i] > maxVal) {
            maxVal = img[i];
        }
    for (int i = 0; i < len; i++)
        img[i] /= maxVal;

    return maxVal;
}

static void multiply(short *img, int len, short maxVal)
{
    for (int i = 0; i < len; i++)
        img[i] *= maxVal;
}

static void normalizePattern(float *innerPattern, float *innerPatternX, unsigned int len)
{
    float sum = 0;
    float maxPtrnVal = innerPattern[0];
    unsigned int i = 0;
    for (i = 0; i < len - 1; i++) {
        sum += (innerPattern[i] + innerPattern[i + 1]) * (innerPatternX[i + 1] - innerPatternX[i]) / 2.;
        if (innerPattern[i] > maxPtrnVal)
            maxPtrnVal = innerPattern[i];
    }
    if (innerPattern[i] > maxPtrnVal)
        maxPtrnVal = innerPattern[i];

    if (fabs(sum - 1.) > 1e-10)
        for (i = 0; i < len; i++)
            innerPattern[i] /= (sum);
}

void deconvImage(short *image, short *ptrnx, short *pattern, int imageWidth, int imageHeight, int ptrnCount,
        float lambda, int maxIterations)
{
    int i = 0, j = 0, row = 0;
    int zeroPatternIndex = 0;

    float *innerPattern = malloc(sizeof(float) * ptrnCount);
    float *innerPatternX = malloc(sizeof(float) * ptrnCount);
    short *innerImage = malloc(sizeof(short) * imageWidth * imageHeight);
    float *innerDeconvSignal2D = malloc(sizeof(float) * imageWidth * imageHeight);
    float *ptrnMatrix = malloc(sizeof(float) * imageWidth * imageWidth);
    float *norms = malloc(sizeof(float) * imageWidth);

    memcpy(innerImage, image, imageWidth * imageHeight * sizeof(short));

    memset(ptrnMatrix, 0, imageWidth * imageWidth * sizeof(float));

    memset(innerDeconvSignal2D, 0, imageWidth * imageHeight * sizeof(float));

    for (i = 0; i < ptrnCount; i++) {
        innerPatternX[i] = -ptrnx[ptrnCount - 1 - i];
        innerPattern[i] = pattern[ptrnCount - 1 - i];
    }

    normalizePattern(innerPattern, innerPatternX, ptrnCount);

    zeroPatternIndex = innerPatternX[0];

    for (i = 0; i < ptrnCount; i++)
        if (innerPatternX[i] == 0) {
            zeroPatternIndex = i;
            break;
        }

    // form convolution matrix
    for (i = 0; i < imageWidth; i++) {
        for (j = 0; j < ptrnCount; j++) {
            if (i + j - zeroPatternIndex < 0)
                continue;
            if (i + j - zeroPatternIndex >= imageWidth)
                break;
            ptrnMatrix[i + j - zeroPatternIndex + i * imageWidth] = innerPattern[j];
        }
    }

    for (i = 0, row = 0; i < imageWidth; i++, row += imageWidth)
        norms[i] = vecNormSqr(&ptrnMatrix[row], imageWidth);

    for (i = 0, row = 0; i < imageHeight; i++, row += imageWidth) {
        innerDeconv(&innerImage[row], &innerDeconvSignal2D[row], norms, imageWidth, ptrnMatrix, lambda, maxIterations);
    }

    // copy to output signal in short
    for (i = 0; i < imageHeight; i++) {
        for (j = 0; j < imageWidth; j += 4) {
            innerImage[j + i * imageWidth] = round(innerDeconvSignal2D[j + i * imageWidth]);
            innerImage[j + i * imageWidth + 1] = round(innerDeconvSignal2D[j + i * imageWidth + 1]);
            innerImage[j + i * imageWidth + 2] = round(innerDeconvSignal2D[j + i * imageWidth + 2]);
            innerImage[j + i * imageWidth + 3] = round(innerDeconvSignal2D[j + i * imageWidth + 3]);
        }
    }

    memcpy(image, innerImage, imageWidth * imageHeight * sizeof(short));

    free(innerPattern);
    free(innerPatternX);
    free(ptrnMatrix);
    free(norms);
    free(innerDeconvSignal2D);
    free(innerImage);
}

void innerDeconv(short *innerSignal, float *innerDeconvSignal, float *norms, unsigned int innerSigCount,
        float *ptrnMatrix, float lambda, unsigned int maxIterations)
{
    float temp = 0, l = 0;

    unsigned int i = 0, j = 0, it = 0, row = 0;

    // ART method ( Kaczmarz method)
    for (it = 0; it < maxIterations; it++) {
        l = lambda;

        for (i = 0, row = 0; i < innerSigCount; i++, row += innerSigCount) {

            if (norms[i] != 0.0f) {
                temp = l * (innerSignal[i] - vecMultiply(&ptrnMatrix[row], innerDeconvSignal, innerSigCount)) /
                        norms[i];
                for (j = 0; j < innerSigCount; j += 4) {
                    innerDeconvSignal[j] = innerDeconvSignal[j] + temp * ptrnMatrix[row + j];
                    innerDeconvSignal[j + 1] = innerDeconvSignal[j + 1] + temp * ptrnMatrix[row + j + 1];
                    innerDeconvSignal[j + 2] = innerDeconvSignal[j + 2] + temp * ptrnMatrix[row + j + 2];
                    innerDeconvSignal[j + 3] = innerDeconvSignal[j + 3] + temp * ptrnMatrix[row + j + 3];
                }
            }
        }

        zeroNegativeElements(innerDeconvSignal, innerSigCount);
    }
}

int loadPattern(PatternData *const patternData, short *pattern, short *ptrnx, int ptrnCount, int sigCount)
{
    int innerSigCount = nextMultiply(sigCount, ALIGN);
    int i = 0, j = 0, row = 0;

    if (!patternData)
        return -1;

    if (patternData->isInitialized)
        return -2;

    patternData->ptrnCount = ptrnCount;
    patternData->innerSigCount = innerSigCount;
    patternData->patternX = malloc(sizeof(float) * ptrnCount);
    patternData->patternY = malloc(sizeof(float) * ptrnCount);
    patternData->ptrnMatrix = malloc(sizeof(float) * innerSigCount * innerSigCount);
    patternData->norms = malloc(sizeof(float) * innerSigCount);

    memset(patternData->ptrnMatrix, 0, innerSigCount * innerSigCount * sizeof(float));

    for (i = 0; i < ptrnCount; i++) {
        patternData->patternX[i] = -ptrnx[ptrnCount - 1 - i];
        patternData->patternY[i] = pattern[ptrnCount - 1 - i];
    }

    normalizePattern(patternData->patternY, patternData->patternX, ptrnCount);

    patternData->zeroPatternIndex = patternData->patternX[0];

    for (i = 0; i < ptrnCount; i++)
        if (patternData->patternX[i] == 0) {
            patternData->zeroPatternIndex = i;
            break;
        }

    // form convolution matrix
    for (i = 0, row = 0; i < innerSigCount; i++, row += innerSigCount) {
        for (j = 0; j < ptrnCount; j++) {
            if (i + j - patternData->zeroPatternIndex < 0)
                continue;
            if (i + j - patternData->zeroPatternIndex >= innerSigCount)
                break;
            patternData->ptrnMatrix[i + j - patternData->zeroPatternIndex + row] = patternData->patternY[j];
        }
    }

    for (i = 0, row = 0; i < innerSigCount; i++, row += innerSigCount) {
        patternData->norms[i] = vecNormSqr(&patternData->ptrnMatrix[row], innerSigCount);
    }

    patternData->isInitialized = true;

    return 0;
}

void unloadPattern(PatternData *const patternData)
{
    free(patternData->patternX);
    free(patternData->patternY);
    free(patternData->ptrnMatrix);
    free(patternData->norms);

    patternData->isInitialized = false;
}

void deconv(short *signal, short *ptrnx, short *pattern, int sigCount, int ptrnCount, float lambda, int maxIterations)
{
    int i = 0, j = 0, row = 0;
    int zeroPatternIndex = 0;
    const int innerSigCount = nextMultiply(sigCount, ALIGN);

    short *innerSignal = malloc(sizeof(short) * innerSigCount);
    float *innerDeconvSignal = malloc(sizeof(float) * innerSigCount);
    float *innerPattern = malloc(sizeof(float) * ptrnCount);
    float *innerPatternX = malloc(sizeof(float) * ptrnCount);
    float *ptrnMatrix = malloc(sizeof(float) * innerSigCount * innerSigCount);
    float *norms = malloc(sizeof(float) * innerSigCount);

    memcpy(innerSignal, signal, sigCount * sizeof(short));
    if (innerSigCount - sigCount > 0)
        memset(innerSignal + sigCount, 0, (innerSigCount - sigCount) * sizeof(short));

    memset(ptrnMatrix, 0, innerSigCount * innerSigCount * sizeof(float));
    memset(innerDeconvSignal, 0, innerSigCount * sizeof(float));

    for (i = 0; i < ptrnCount; i++) {
        innerPatternX[i] = -ptrnx[ptrnCount - 1 - i];
        innerPattern[i] = pattern[ptrnCount - 1 - i];
    }

    normalizePattern(innerPattern, innerPatternX, ptrnCount);

    zeroPatternIndex = innerPatternX[0];

    for (i = 0; i < ptrnCount; i++)
        if (innerPatternX[i] == 0) {
            zeroPatternIndex = i;
            break;
        }

    // form convolution matrix
    for (i = 0; i < innerSigCount; i++) {
        for (j = 0; j < ptrnCount; j++) {
            if (i + j - zeroPatternIndex < 0)
                continue;
            if (i + j - zeroPatternIndex >= innerSigCount)
                break;
            ptrnMatrix[i + j - zeroPatternIndex + i * innerSigCount] = innerPattern[j];
        }
    }

    for (i = 0, row = 0; i < innerSigCount; i++, row += innerSigCount) {
        norms[i] = vecNormSqr(&ptrnMatrix[row], innerSigCount);
    }

    innerDeconv(innerSignal, innerDeconvSignal, norms, innerSigCount, ptrnMatrix, lambda, maxIterations);

    for (i = 0; i < sigCount; i++)
        signal[i] = round(innerDeconvSignal[i]);

    free(innerPattern);
    free(innerPatternX);
    free(ptrnMatrix);
    free(norms);
    free(innerSignal);
    free(innerDeconvSignal);
}

int fastDeconv(const PatternData *const patternData, short *signal, int sigCount, float lambda, int maxIterations)
{
    int i = 0;

    const int innerSigCount = nextMultiply(sigCount, ALIGN);
    const int lessSigCount = prevMultiply(sigCount, ALIGN);

    if (!patternData)
        return -1;

    if (innerSigCount != patternData->innerSigCount)
        return -2;

    short *innerSignal = malloc(sizeof(short) * innerSigCount);
    float *innerDeconvSignal = malloc(sizeof(float) * innerSigCount);

    memcpy(innerSignal, signal, sigCount * sizeof(short));
    memset(innerSignal + sigCount, 0, (innerSigCount - sigCount) * sizeof(short));

    memset(innerDeconvSignal, 0, innerSigCount * sizeof(float));

    innerDeconv(innerSignal, innerDeconvSignal, patternData->norms, innerSigCount, patternData->ptrnMatrix, lambda,
            maxIterations);

    for (i = 0; i < lessSigCount; i += 4) {
        signal[i] = (int)innerDeconvSignal[i];
        signal[i + 1] = (int)innerDeconvSignal[i + 1];
        signal[i + 2] = (int)innerDeconvSignal[i + 2];
        signal[i + 3] = (int)innerDeconvSignal[i + 3];
    }

    for (; i < sigCount; i++)
        signal[i] = (int)innerDeconvSignal[i];

    free(innerSignal);
    free(innerDeconvSignal);

    return 0;
}