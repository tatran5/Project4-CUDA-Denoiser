#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
	SPHERE,
	CUBE,
	MESH
};

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

struct Geom {
	enum GeomType type;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	glm::mat4 invTranspose;
	// Mesh-related
	int triangleIdxStart = -1;
	int triangleIdxEnd = -1;
	glm::vec3 minPos;
	glm::vec3 maxPos;
};

struct Triangle {
	glm::vec3 v[3];
	glm::vec3 n[3];
};

struct Material {
	glm::vec3 color;
	struct {
		float exponent;
		glm::vec3 color;
	} specular;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float emittance;
	// Refraction
	float ior0;
	float ior1;
};

struct Camera {
	glm::ivec2 resolution;
	glm::vec3 position;
	glm::vec3 lookAt;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec2 fov;
	glm::vec2 pixelLength;
	// Motion blur
	glm::vec3 move;
};

struct RenderState {
	Camera camera;
	unsigned int iterations;
	int traceDepth;
	std::vector<glm::vec3> image;
	std::string imageName;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
	float t;
	glm::vec3 surfaceNormal;
	int materialId;
};

struct GBufferPixel {
	float t;
};

struct GBufferPixelVec3 {
	glm::vec3 v;
};