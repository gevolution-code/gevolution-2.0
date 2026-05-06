//////////////////////////
// lightcone_device_workspace.hpp
//////////////////////////

#ifndef LIGHTCONE_DEVICE_WORKSPACE_HEADER
#define LIGHTCONE_DEVICE_WORKSPACE_HEADER

#include <cub/cub.cuh>
#include <iostream>
#include <stddef.h>
#include <stdint.h>
#include <stdexcept>

class LightconeDeviceWorkspace
{
public:
	LightconeDeviceWorkspace(size_t bytes, const char * context, const char * buffer_class)
		: base_(NULL), offset_(0), bytes_(bytes), private_alloc_(false)
		, context_(context), buffer_class_(buffer_class)
	{
		if (bytes_ == 0)
			return;

#ifdef FFT3D
		LATfield2::tempMemory.reserveDeviceWorkspaceBytes(bytes_, context);

		if (LATfield2::tempMemory.deviceWorkspace() != NULL && LATfield2::tempMemory.deviceWorkspaceBytes() >= bytes_)
		{
			base_ = static_cast<unsigned char *>(LATfield2::tempMemory.deviceWorkspace());
			return;
		}
#endif

		private_alloc_ = true;
		cudaMalloc((void **) &base_, bytes_);

		static bool warned = false;

		if (!warned)
		{
			std::cout << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " particle light-cone " << buffer_class << " exceeds LATfield2 shared device workspace (" << bytes_ << " bytes required); using private device allocation." << std::endl;
			warned = true;
		}
	}

	~LightconeDeviceWorkspace()
	{
		if (private_alloc_ && base_ != NULL)
			cudaFree(base_);
	}

	LightconeDeviceWorkspace(const LightconeDeviceWorkspace &) = delete;
	LightconeDeviceWorkspace & operator=(const LightconeDeviceWorkspace &) = delete;

	size_t mark() const
	{
		return offset_;
	}

	void reset(size_t offset = 0)
	{
		offset_ = offset;
	}

	template <typename T>
	T * slice(size_t count, const char * label = NULL)
	{
		return static_cast<T *>(slice_bytes(sizeof(T) * count, alignment<T>(), label));
	}

	void * slice_bytes(size_t bytes, size_t alignment = 256, const char * label = NULL)
	{
		if (bytes == 0)
			return NULL;

		size_t aligned_offset = align_up(offset_, alignment);

		if (aligned_offset + bytes > bytes_)
		{
			std::cerr << COLORTEXT_RED << " error" << COLORTEXT_RESET
			          << ": proc#" << parallel.rank()
			          << " Lightcone device workspace exhausted";

			if (label != NULL)
				std::cerr << " while allocating " << label;

			if (context_ != NULL)
				std::cerr << " (" << context_ << ")";

			std::cerr << ": request=" << bytes
			          << " bytes, alignment=" << alignment
			          << ", used=" << offset_
			          << " bytes, aligned_used=" << aligned_offset
			          << " bytes, capacity=" << bytes_
			          << " bytes, backing=" << (private_alloc_ ? "private cudaMalloc" : "LATfield2 shared workspace");

			if (buffer_class_ != NULL)
				std::cerr << ", class=" << buffer_class_;

			std::cerr << std::endl;

			throw std::runtime_error("Lightcone device workspace exhausted");
		}

		offset_ = aligned_offset;
		void * ptr = base_ + offset_;
		offset_ += bytes;
		return ptr;
	}

	static size_t align_up(size_t value, size_t alignment = 256)
	{
		return ((value + alignment - 1) / alignment) * alignment;
	}

	template <typename T>
	static size_t aligned_bytes(size_t count)
	{
		return align_up(sizeof(T) * count, alignment<T>());
	}

private:
	template <typename T>
	static size_t alignment()
	{
		return alignof(T) > 256 ? alignof(T) : 256;
	}

	unsigned char * base_;
	size_t offset_;
	size_t bytes_;
	bool private_alloc_;
	const char * context_;
	const char * buffer_class_;
};

inline void lightcone_device_radix_sort_host_ids(long * ids, size_t count, const char * context)
{
	if (count < 2)
		return;

	size_t sort_temp_bytes = 0;

	cub::DeviceRadixSort::SortKeys(NULL, sort_temp_bytes, (long *) NULL, (long *) NULL, count);

	size_t workspace_bytes = 0;

	workspace_bytes += LightconeDeviceWorkspace::aligned_bytes<long>(count);
	workspace_bytes += LightconeDeviceWorkspace::aligned_bytes<long>(count);
	workspace_bytes += LightconeDeviceWorkspace::align_up(sort_temp_bytes);

	LightconeDeviceWorkspace workspace(workspace_bytes, context, "ID backlog radix sort workspace");

	long * d_in = workspace.slice<long>(count, "radix-sort input IDs");
	long * d_out = workspace.slice<long>(count, "radix-sort output IDs");
	void * sort_temp = workspace.slice_bytes(sort_temp_bytes, 256, "CUB radix-sort temporary storage");

	cudaMemcpy(d_in, ids, sizeof(long) * count, cudaMemcpyHostToDevice);
	cub::DeviceRadixSort::SortKeys(sort_temp, sort_temp_bytes, d_in, d_out, count);
	cudaMemcpy(ids, d_out, sizeof(long) * count, cudaMemcpyDeviceToHost);
}

#endif
