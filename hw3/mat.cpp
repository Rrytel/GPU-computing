#include "mat.h"

void GpuMatrix::load(const CpuMatrix oldMatrix) {
  size_t size = width * height * sizeof(float);
  hipMemcpy(elements, oldMatrix.elements.data(), size, hipMemcpyHostToDevice);
}

void GpuMatrix3::load(const CpuMatrix3 oldMatrix) {
  size_t size = width * height * sizeof(float3);
  hipMemcpy(elements, oldMatrix.elements.data(), size, hipMemcpyHostToDevice);
}

void GpuVector::load(const CpuVector oldVector) {
  size_t size = width * sizeof(float);
  hipMemcpy(elements, oldVector.elements.data(), size, hipMemcpyHostToDevice);
}

void GpuMatrix::deAllocate() { hipFree(elements); }

void GpuMatrix3::deAllocate() { hipFree(elements); }

void GpuVector::deAllocate() { hipFree(elements); }

void CpuMatrix::load(const GpuMatrix oldMatrix) {
  size_t size = width * height * sizeof(float);
  hipMemcpy(elements.data(), oldMatrix.elements, size, hipMemcpyDeviceToHost);
}

void CpuMatrix3::load(const GpuMatrix3 oldMatrix) {
  size_t size = width * height * sizeof(float3);
  hipMemcpy(elements.data(), oldMatrix.elements, size, hipMemcpyDeviceToHost);
}

void CpuVector::load(const GpuVector oldVector) {
  size_t size = width * sizeof(float);
  hipMemcpy(elements.data(), oldVector.elements, size, hipMemcpyDeviceToHost);
}

void CpuMatrix::deAllocate() { elements.clear(); }

void CpuMatrix3::deAllocate() { elements.clear(); }

void CpuVector::deAllocate() { elements.clear(); }
