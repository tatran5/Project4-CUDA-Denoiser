#pragma once

#include <vector>
#include "scene.h"
#include <chrono>

enum class DisplayType {DEFAULT, GBUFFER_DEFAULT, GBUFFER_NORMAL, GBUFFER_POSITION, GBUFFER_COLOR, DENOISE};

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, DisplayType displayType, int filterSize, float cPhi, float nPhi, float pPhi);
void showGBuffer(uchar4* pbo, DisplayType displayType);
void showImage(uchar4* pbo, int iter);

void aTrousWaveletFilter(int filterSize, int camResX, int camResY, 
	GBufferPixelVec3* gbufferPos, GBufferPixelVec3* gbufferNor,
	GBufferPixelVec3* gbufferCol, GBufferPixelVec3* gbufferCol1, 
	float cPhi, float nPhi, float pPhi);
void generateGaussianFilter();