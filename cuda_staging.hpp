//////////////////////////
// cuda_staging.hpp
//////////////////////////

#ifndef CUDA_STAGING_HEADER
#define CUDA_STAGING_HEADER

#include <cuda_runtime.h>
#include <new>
#include <stdexcept>
#include <utility>

inline void gevolution_cuda_check(cudaError_t status, const char * context)
{
	if (status != cudaSuccess)
		throw std::runtime_error(context);
}

template <typename T>
class DeviceStagingBuffer
{
public:
	DeviceStagingBuffer() : data_(nullptr), count_(0) {}

	explicit DeviceStagingBuffer(size_t count) : data_(nullptr), count_(count)
	{
		if (count_ > 0)
			gevolution_cuda_check(cudaMalloc((void **) &data_, count_ * sizeof(T)), "CUDA staging allocation failed");
	}

	DeviceStagingBuffer(const T * host_data, size_t count) : DeviceStagingBuffer(count)
	{
		copy_from_host(host_data, count);
	}

	~DeviceStagingBuffer()
	{
		if (data_ != nullptr)
			cudaFree(data_);
	}

	DeviceStagingBuffer(const DeviceStagingBuffer &) = delete;
	DeviceStagingBuffer & operator=(const DeviceStagingBuffer &) = delete;

	T * data() { return data_; }
	const T * data() const { return data_; }

	void copy_from_host(const T * host_data, size_t count)
	{
		if (count > count_)
			throw std::runtime_error("CUDA staging buffer overflow");
		if (count > 0)
			gevolution_cuda_check(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice), "CUDA staging copy to device failed");
	}

	void copy_to_host(T * host_data, size_t count) const
	{
		if (count > count_)
			throw std::runtime_error("CUDA staging buffer overflow");
		if (count > 0)
			gevolution_cuda_check(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost), "CUDA staging copy to host failed");
	}

private:
	T * data_;
	size_t count_;
};

template <typename T>
class ManagedCudaObject
{
public:
	template <typename... Args>
	explicit ManagedCudaObject(Args&&... args) : object_(nullptr)
	{
		gevolution_cuda_check(cudaMallocManaged((void **) &object_, sizeof(T)), "CUDA managed-object allocation failed");
		new (object_) T(std::forward<Args>(args)...);
	}

	~ManagedCudaObject()
	{
		if (object_ != nullptr)
		{
			object_->~T();
			cudaFree(object_);
		}
	}

	ManagedCudaObject(const ManagedCudaObject &) = delete;
	ManagedCudaObject & operator=(const ManagedCudaObject &) = delete;

	T & get() { return *object_; }
	const T & get() const { return *object_; }
	T * operator->() { return object_; }
	const T * operator->() const { return object_; }

private:
	T * object_;
};

#endif
