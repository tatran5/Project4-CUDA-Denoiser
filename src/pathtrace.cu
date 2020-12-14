#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <iostream>
#include <iomanip>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ANTIALIASING 0
#define TIMEPATHTRACE 1 // Measure performance
#define DEPTHOFFIELD 0
#define MOTIONBLUR 0

#define SORTPATHSBYMATERIAL 1 // Improve performance
#define CACHEFIRSTINTERSECTIONS 0 // Improve performance

#define ERRORCHECK 1

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static GBufferPixelVec3* dev_gBufferNor = NULL;
static GBufferPixelVec3* dev_gBufferPos = NULL;
static GBufferPixelVec3* dev_gBufferCol = NULL;
static GBufferPixelVec3* dev_gBufferCol1 = NULL;
static float* dev_gaussianFilter = NULL;
static int gaussianFilterSize = 5;

static ShadeableIntersection* dev_firstIntersections = NULL; // Cache first bounce of first iter to be re-use in other iters
static Triangle* dev_tris = NULL; // Store triangle information for meshes
static glm::vec2* dev_samples = NULL;

static std::chrono::steady_clock::time_point timePathTrace; // Measure performance

// Depth of field
static float lensRadius = 0.5f;
static float focalDist = 10.f;

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

// --------------------------------------------------------------
// ----------------- GBUFFER TO PBO -----------------------------
// --------------------------------------------------------------

__global__ void gbufferVec3TestToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixelVec3* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		/// <summary>
		/// 
		/// </summary>
		/// <param name="pbo"></param>
		/// <param name="resolution"></param>
		/// <param name="gBuffer"></param>
		/// <returns></returns>
		pbo[index].w = 0;
		float scalar = 32.f;
		pbo[index].x = glm::clamp(glm::abs(gBuffer[index].v[0]) * scalar, 0.f, 255.f);
		pbo[index].y = glm::clamp(glm::abs(gBuffer[index].v[1]) * scalar, 0.f, 255.f);
		pbo[index].z = glm::clamp(glm::abs(gBuffer[index].v[2]) * scalar, 0.f, 255.f);
	}
}

__global__ void gbufferVec3ToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixelVec3* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		
		glm::vec3 gBufferVal = gBuffer[index].v;
		pbo[index].w = 0;
		pbo[index].x = gBufferVal[0];
		pbo[index].y = gBufferVal[1];
		pbo[index].z = gBufferVal[2];
	}
}


__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		float timeToIntersect = gBuffer[index].t * 256.0;

		pbo[index].w = 0;
		pbo[index].x = timeToIntersect;
		pbo[index].y = timeToIntersect;
		pbo[index].z = timeToIntersect;
	}
}

//---------------------------------------------------------------------
//----------- GAUSSIAN FILTER GENERATION ------------------------------
//---------------------------------------------------------------------

__global__ void generateGaussianFilterNotNormalized(int filterSize, float* filter) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= filterSize || y >= filterSize) {
		return;
	}

	float sigma = 1.f; // Intialising standard deviation to 1.0 
	float s = 2.f * sigma * sigma;
	int index = x + (y * filterSize);
	int halfFilterSize = filterSize / 2;
	// Adjusted the x and y index so that the center of the filter is 
	// at coordinate (0, 0) instead of (halfFilterSize, halfFilterSize)
	int newX = x - halfFilterSize;
	int newY = y - halfFilterSize;
	float r = (float)glm::sqrt((float)(newX * newX + newY * newY));
	float nom = exp(-(r * r) / s);
	float denom = glm::pi<float>() * s;
	float val = nom / denom;
	filter[index] = val;
}

__global__ void normalizeFilter(int filterSize, float* filter, float sum) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < filterSize && y < filterSize) {
		int index = x + (y * filterSize);
		filter[index] /= sum;
	}
}


// Wraps around the two kernels above to generate a gaussian filter
void generateGaussianFilter() {
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(gaussianFilterSize + blockSize2d.x - 1) / blockSize2d.x,
		(gaussianFilterSize + blockSize2d.y - 1) / blockSize2d.y);
	generateGaussianFilterNotNormalized << <blocksPerGrid2d, blockSize2d >> > (gaussianFilterSize, dev_gaussianFilter);
	int countGaussianFilterElm = gaussianFilterSize * gaussianFilterSize;
	float* testGaussianFilter = new float[countGaussianFilterElm];
	cudaMemcpy(testGaussianFilter, dev_gaussianFilter, countGaussianFilterElm * sizeof(float), cudaMemcpyDeviceToHost);

	float sumGaussianNotNormalized =
		thrust::reduce(thrust::device, dev_gaussianFilter, dev_gaussianFilter + countGaussianFilterElm, 0.f);
	normalizeFilter << <blocksPerGrid2d, blockSize2d >> > (gaussianFilterSize, dev_gaussianFilter, sumGaussianNotNormalized);
	delete testGaussianFilter;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_tris, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
	cudaMalloc(&dev_gBufferPos, pixelcount * sizeof(GBufferPixelVec3));
	cudaMalloc(&dev_gBufferNor, pixelcount * sizeof(GBufferPixelVec3));
	cudaMalloc(&dev_gBufferCol, pixelcount * sizeof(GBufferPixelVec3));
	cudaMalloc(&dev_gBufferCol1, pixelcount * sizeof(GBufferPixelVec3));

	int countGaussianFilterElm = gaussianFilterSize * gaussianFilterSize;
	cudaMalloc(&dev_gaussianFilter, (countGaussianFilterElm) * sizeof(float));
	generateGaussianFilter();

	cudaMalloc(&dev_firstIntersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_firstIntersections, 0, pixelcount * sizeof(ShadeableIntersection));

	/*const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(samples1D + blockSize2d.x - 1) / blockSize2d.x,
		(samples1D + blockSize2d.y - 1) / blockSize2d.y);
	*/
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_gBuffer);
	cudaFree(dev_gBufferPos);
	cudaFree(dev_gBufferNor);
	cudaFree(dev_gBufferCol);
	cudaFree(dev_gBufferCol1);
	cudaFree(dev_gaussianFilter);
	cudaFree(dev_tris);
	cudaFree(dev_firstIntersections);

	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		float addedJitter0 = 0.f;
		float addedJitter1 = 0.f;
#ifdef ANTIALIASING
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> uo(-0.5, 0.5);
		addedJitter0 = uo(rng);
		addedJitter1 = uo(rng);
#endif

#ifdef MOTIONBLUR
		thrust::uniform_real_distribution<float> u01(0, 1);
		float interpolateBlurVal = u01(rng);
		glm::vec3 blurredView = cam.view;
		if (MOTIONBLUR) {
			blurredView -= cam.move * interpolateBlurVal;
		}
		segment.ray.direction = glm::normalize(blurredView
			- cam.right * cam.pixelLength.x * ((float)x + addedJitter0 - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + addedJitter1 - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + addedJitter0 - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + addedJitter1 - (float)cam.resolution.y * 0.5f)
		);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, Triangle* tris
	, int geoms_size
	, ShadeableIntersection* intersections
)
{
#if TIMEPATHTRACE
#endif
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == MESH) {
				t = meshIntersectionTest(tris, geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment path = pathSegments[idx];
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f && pathSegments[idx].remainingBounces > 0) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				glm::vec3 isectLoc = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
				pathSegments[idx].remainingBounces -= 1;

				if (pathSegments[idx].remainingBounces == 0) {
					pathSegments[idx].color = glm::vec3(0.f);
				}
				else {
					scatterRay(pathSegments[idx], isectLoc, intersection.surfaceNormal, material, rng);
				}
			}
		}
		else {
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

//---------------------------------------------------------------------
//--------------------GENERATE G-BUFFERS-------------------------------
//---------------------------------------------------------------------

__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixel* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		gBuffer[idx].t = shadeableIntersections[idx].t;
	}
}

__global__ void generateGBufferPositions(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixelVec3* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		Ray ray = pathSegments[idx].ray;
		glm::vec3 pos = ray.origin + ray.direction * shadeableIntersections[idx].t;
		gBuffer[idx].v = pos;
	}
}

__global__ void generateGBufferNormals(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixelVec3* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		gBuffer[idx].v = shadeableIntersections[idx].surfaceNormal;
	}
}


__global__ void generateGBufferColors(int nPaths, GBufferPixelVec3* gBuffer, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		glm::vec3 col = iterationPath.color;
		gBuffer[iterationPath.pixelIndex].v = col;
	}
}

__global__ void copyBufferToImage(int num_paths, const GBufferPixelVec3* gBuffer, glm::vec3* image) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		image[idx] = gBuffer[idx].v;
	}
}

//---------------------------------------------------------------------
//-------------------- FINAL GATHER -----------------------------------
//---------------------------------------------------------------------

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		glm::vec3 col = iterationPath.color;
		image[iterationPath.pixelIndex] += col;
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGatherWithDenoise(int nPaths, glm::vec3* image, GBufferPixelVec3* curIterationDenoised)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		image[index] += curIterationDenoised[index].v;
	}
}

// ----------------------------------------------------------------------------------------------
// ---------------------------- STREAM COMPACTION -----------------------------------------------
// ----------------------------------------------------------------------------------------------

struct pathIsAlive {
	__host__ __device__
		bool operator()(const PathSegment& path) {
		return path.remainingBounces > 0;
	}
};

struct compareMaterial {
	__host__ __device__
		bool operator()(const ShadeableIntersection& isect1, const ShadeableIntersection& isect2) {
		return isect1.materialId > isect2.materialId;
	}
};

// ----------------------------------------------------------------------------------------------
// ---------------------------- DEPTH OF FIELD --------------------------------------------------
// ----------------------------------------------------------------------------------------------

// Assume that the input is a vec2 where each component is a uniform random number [-1, 1]
__host__ __device__ glm::vec2 sampleConcentricDisk(const glm::vec2& u) {
	// Handle degeneracy at the origin
	if (u.x == 0 && u.y == 0) {
		return glm::vec2(0.f);
	}
	float theta = 0.f;
	float r = 0.f;
	if (glm::abs(u.x) > glm::abs(u.y)) {
		r = u.x;
		theta = glm::pi<float>() / 4.f * u.y / u.x;
	}
	else {
		r = u.y;
		theta = glm::pi<float>() / 2.f - glm::pi<float>() / 4.f * u.x / u.y;
	}
	return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}

// Based on PBRT 6.2.3 
__global__ void updateRaysForDepthOfField(
	int iter,
	int numPaths,
	float camResX, float camResY,
	float lensRadius, float focalDist,
	PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= camResX || y >= camResY) {
		return;
	}
	int index = x + (y * camResX);

	// Sample point on lens
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	thrust::uniform_real_distribution<float> urd(-1, 1);
	float sampleX = urd(rng);
	float sampleY = urd(rng);
	glm::vec2 sample = glm::vec2(sampleX, sampleY);
	glm::vec2 diskSample = sampleConcentricDisk(sample);  // Warp from square to uniform
	glm::vec3 offset = lensRadius * glm::vec3(diskSample.x, diskSample.y, 0.f);

	//Compute point on plane of focus
	Ray rayCopy = pathSegments[index].ray;
	float ft = glm::abs(focalDist / rayCopy.direction.z);
	glm::vec3 pFocus = rayCopy.origin + rayCopy.direction * ft;

	// Update ray for effect of lens
	glm::vec3 newOrigin = rayCopy.origin + offset;
	glm::vec3 newDirection = glm::normalize(pFocus - newOrigin);
	pathSegments[index].ray.origin = newOrigin;
	pathSegments[index].ray.direction = newDirection;
}

// ----------------------------------------------------------------------------------------------
// ---------------------------- DENOISER --------------------------------------------------------
// ----------------------------------------------------------------------------------------------

__global__ void test(
	const GBufferPixelVec3* dataPos, const GBufferPixelVec3* dataNor,
	const GBufferPixelVec3* dataCol, GBufferPixelVec3* dataCol1)
{}

__global__ void applySparseFilter(
	int filterSize, float* filter,
	const GBufferPixelVec3* dataPos, const GBufferPixelVec3* dataNor,
	const GBufferPixelVec3* dataCol, GBufferPixelVec3* dataCol1,
	int offset, float cPhi, float nPhi, float pPhi,
	int camResX, int camResY)
{

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < camResX && y < camResY) {
		int indexPix = x + (y * camResX);

		glm::vec3 sum = glm::vec3(0.f);
		glm::vec3 cval = dataCol[indexPix].v;
		glm::vec3 nval = dataNor[indexPix].v;
		glm::vec3 pval = dataPos[indexPix].v;

		float cum_w = 0.f;
		int halfFilterSize = filterSize / 2;
		int pixSampledCount = 0;
		for (int unfactoredOffsetX = -halfFilterSize; unfactoredOffsetX <= halfFilterSize; unfactoredOffsetX++) {
			for (int unfactoredOffsetY = -halfFilterSize; unfactoredOffsetY <= halfFilterSize; unfactoredOffsetY++) {
				// Get the location of the sampled pixel
				int sampleX = x + unfactoredOffsetX * offset;
				int sampleY = y + unfactoredOffsetY * offset;

				if (sampleX >= 0 && sampleX < camResX && sampleY >= 0 && sampleY < camResY) {
					pixSampledCount++;
					int indexPixSample = sampleX + sampleY * camResX;

					// Find the color weight by this pixel
					glm::vec3 ctmp = dataCol[indexPixSample].v;
					glm::vec3 t = cval - ctmp;
					float dist2 = glm::dot(t, t);
					float cwSample = glm::min(glm::exp(-dist2 / cPhi), 1.f);

					glm::vec3 ntmp = dataNor[indexPixSample].v;
					t = nval - ntmp;
					dist2 = glm::max(glm::dot(t, t) / (offset * offset), 0.f);
					float nwSample = glm::min(glm::exp(-dist2 / nPhi), 1.f);

					glm::vec3 ptmp = dataPos[indexPixSample].v;
					t = pval - ptmp;
					dist2 = glm::dot(t, t);
					float pwSample = glm::min(glm::exp(-dist2 / pPhi), 1.f);

					float weight = cwSample * nwSample * pwSample;

					int filterPosX = unfactoredOffsetX + halfFilterSize;
					int filterPosY = unfactoredOffsetY + halfFilterSize;
					int filterIdx = filterPosX + filterPosY * filterSize;
					sum += ctmp * weight * filter[filterIdx];
					cum_w += weight * filter[filterIdx];
				}
			}
		}
		if (cum_w > 0) {
			dataCol1[indexPix].v = sum / cum_w;
		}
	}
}

// Filter the image with iterations
void aTrousWaveletFilter(int filterSize, int camResX, int camResY,
	GBufferPixelVec3* gBufferPos, GBufferPixelVec3* gBufferNor,
	GBufferPixelVec3* gBufferCol, GBufferPixelVec3* gBufferCol1,
	float cPhi, float nPhi, float pPhi) 
{
	int blurIteration = (int)(glm::log2(filterSize / 2) + 1.f); 
	int offset = 1;

	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(camResX + blockSize2d.x - 1) / blockSize2d.x,
		(camResY + blockSize2d.y - 1) / blockSize2d.y);

	for (int i = 0; i < blurIteration; i++) { 
		applySparseFilter << <blocksPerGrid2d, blockSize2d >> > 
			(gaussianFilterSize, dev_gaussianFilter,
			gBufferPos, gBufferNor,
			gBufferCol, gBufferCol1,
			offset, cPhi, nPhi, pPhi, camResX, camResY);
	
		// Ping-pong the two color buffers
		GBufferPixelVec3* temp = gBufferCol;
		gBufferCol = gBufferCol1;
		gBufferCol1 = temp;

		offset *= 2;
	}
}

// ----------------------------------------------------------------------------------------------
// ---------------------------- MAIN ------------------------------------------------------------
// ----------------------------------------------------------------------------------------------

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, DisplayType displayType, int filterSize, float cPhi, float nPhi, float pPhi) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;


	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

#if TIMEPATHTRACE
	timePathTrace = std::chrono::high_resolution_clock::now();
#endif

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

#if DEPTHOFFIELD
	updateRaysForDepthOfField << <blocksPerGrid2d, blockSize2d >> > (iter, pixelcount, cam.resolution.x, cam.resolution.y, lensRadius, focalDist, dev_paths);
	checkCUDAError("depth of field");
#endif


	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	// Empty gbuffer
	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));
	cudaMemset(dev_gBufferPos, 0, pixelcount * sizeof(GBufferPixelVec3));
	cudaMemset(dev_gBufferNor, 0, pixelcount * sizeof(GBufferPixelVec3));
	cudaMemset(dev_gBufferCol, 0, pixelcount * sizeof(GBufferPixelVec3));
	cudaMemset(dev_gBufferCol1, 0, pixelcount * sizeof(GBufferPixelVec3));

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	bool iterationComplete = false;
	while (!iterationComplete) {
		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHEFIRSTINTERSECTIONS
		if (depth == 0) {
			if (iter <= 1) {
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
					, dev_tris
					, hst_scene->geoms.size()
					, dev_firstIntersections
					);
				checkCUDAError("trace first bounce, first iter");
			}
			cudaDeviceSynchronize();
			cudaMemcpy(dev_intersections, dev_firstIntersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, dev_tris
				, hst_scene->geoms.size()
				, dev_intersections
				);
			checkCUDAError("trace one bounce");
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, dev_tris
			, hst_scene->geoms.size()
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
#endif

		cudaDeviceSynchronize();
		if (depth == 0) {
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
			generateGBufferPositions << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBufferPos);
			generateGBufferNormals << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBufferNor);
		}

		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

#if SORTPATHSBYMATERIAL
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareMaterial());
#endif

		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);

		dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_path_end, pathIsAlive());
		num_paths = dev_path_end - dev_paths;
		

		iterationComplete = num_paths == 0;
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	if (displayType == DisplayType::DENOISE) {
		generateGBufferColors << < numBlocksPixels, blockSize1d >> > (pixelcount, dev_gBufferCol, dev_paths);
		aTrousWaveletFilter(filterSize, cam.resolution.x, cam.resolution.y,
			dev_gBufferPos, dev_gBufferNor, dev_gBufferCol, dev_gBufferCol1, cPhi, nPhi, pPhi);
		finalGatherWithDenoise << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_gBufferCol);
	}
	else {
		finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);
	}
	
#if TIMEPATHTRACE
	double ms = (double)std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - timePathTrace).count();
	std::cout << ms << std::endl;
#endif
	///////////////////////////////////////////////////////////////////////////

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}

void showGBuffer(uchar4* pbo, DisplayType displayType) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	if (displayType == DisplayType::GBUFFER_DEFAULT) {
		gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
	}
	else if (displayType == DisplayType::GBUFFER_NORMAL) {
		gbufferVec3TestToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBufferNor);
	}
	else if (displayType == DisplayType::GBUFFER_POSITION) {
		gbufferVec3TestToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBufferPos);
	} 
	else if (displayType == DisplayType::GBUFFER_COLOR) {
		gbufferVec3ToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBufferCol);
	}
}

void showImage(uchar4* pbo, int iter) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}