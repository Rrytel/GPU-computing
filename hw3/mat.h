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
class GpuMatrix;
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
