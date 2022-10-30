#ifndef MAT_H
#define MAT_H
#include <hip/hip_runtime.h>
#include <vector>

class Matrix {
public:
  /* Member Data */
  int width;
  int height;
};

class Vector {
public:
  /* Member Data */
  int width;
};

class CpuMatrix;
class CpuMatrix3;
class GpuMatrix;
class GpuMatrix3;
class CpuVector;
class GpuVector;

class GpuMatrix : public Matrix {
public:
  float *elements;

  /* Constructor */
  GpuMatrix(const int w, const int h) {
    width = w;
    height = h;
    hipMalloc(&elements, width * height * sizeof(float));
  }

  void load(const CpuMatrix oldMatrix);
  void deAllocate();
};

class GpuMatrix3 : public Matrix {
public:
  float3 *elements;

  /* Constructor */
  GpuMatrix3(const int w, const int h) {
    width = w;
    height = h;
    hipMalloc(&elements, width * height * sizeof(float3));
  }

  void load(const CpuMatrix3 oldMatrix);
  void deAllocate();
};


class GpuVector : public Vector {
public:
  float *elements;

  /* Constructor */
  GpuVector(const int w) {
    width = w;
    hipMalloc(&elements, width * sizeof(float));
  }

  void load(const CpuVector oldVector);
  void deAllocate();
};

class CpuMatrix : public Matrix {
public:
  std::vector<float> elements;
  /* Constructor */
  CpuMatrix(const int w, const int h, const float val = 0) {
    width = w;
    height = h;
    elements.resize(w * h, val);
  }

  void load(const GpuMatrix oldMatrix);
  void deAllocate();
};

class CpuMatrix3 : public Matrix {
public:
  std::vector<float3> elements;
  /* Constructor */
  CpuMatrix3(const int w, const int h, const float val = 0) {
    width = w;
    height = h;
    elements.resize(w * h);
  }

  void load(const GpuMatrix3 oldMatrix);
  void deAllocate();
};

class CpuVector : public Vector {
public:
  std::vector<float> elements;
  /* Constructor */
  CpuVector(const int w, const float val = 0) {
    width = w;
    elements.resize(w, val);
  }

  void load(const GpuVector oldVector);
  void deAllocate();
};

#endif
