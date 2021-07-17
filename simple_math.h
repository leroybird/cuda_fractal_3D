/*
  Simple maths operators. Header, only file. 
  
*/
#include "math.h"
#include "cuda_runtime.h"


__host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator*(const float3 &a, const float &b)
{
  return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float3 operator*(const float &b, const float3 &a)
{
  return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float length(const float3 &vec)
{
  return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__host__ __device__ float3 norm(const float3 vec)
{
  return vec * (1.0f / length(vec));
}

__host__ __device__ float dot(const float3 &a, const float3 &b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 cross(const float3 &a, const float3 &b)
{
  return make_float3((a.y * b.z - a.z * b.y),
                     (-(a.x * b.z - a.z * b.x)),
                     (a.x * b.y - a.y * b.x));
}
