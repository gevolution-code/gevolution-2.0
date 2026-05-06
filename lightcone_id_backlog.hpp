//////////////////////////
// lightcone_id_backlog.hpp
//////////////////////////

#ifndef LIGHTCONE_ID_BACKLOG_HEADER
#define LIGHTCONE_ID_BACKLOG_HEADER

#include <algorithm>
#include <stddef.h>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

class LightconeIDBacklog
{
public:
	LightconeIDBacklog()
		: finalized_(true)
	{
	}

	void clear()
	{
		ids_.clear();
		finalized_ = true;
	}

	void append(long id)
	{
		ids_.push_back(id);
		finalized_ = false;
	}

	void append(const long * ids, size_t count)
	{
		if (count == 0)
			return;

		ids_.insert(ids_.end(), ids, ids + count);
		finalized_ = false;
	}

	void append(const std::vector<long> & ids)
	{
		append(ids.data(), ids.size());
	}

#ifdef __CUDACC__
	void append_from_device(const long * ids, size_t count)
	{
		if (count == 0)
			return;

		size_t old_size = ids_.size();

		ids_.resize(old_size + count);
		cudaMemcpy(ids_.data() + old_size, ids, sizeof(long) * count, cudaMemcpyDeviceToHost);
		finalized_ = false;
	}
#endif

	void finalize()
	{
		if (finalized_)
			return;

		std::sort(ids_.begin(), ids_.end());
		finalize_presorted();
	}

	void finalize_presorted()
	{
		if (finalized_)
			return;

		ids_.erase(std::unique(ids_.begin(), ids_.end()), ids_.end());
		finalized_ = true;
	}

	bool contains(long id) const
	{
		return std::binary_search(ids_.begin(), ids_.end(), id);
	}

	size_t size() const
	{
		return ids_.size();
	}

	bool empty() const
	{
		return ids_.empty();
	}

	const long * data() const
	{
		return ids_.data();
	}

	long * data()
	{
		return ids_.data();
	}

	void resize(size_t count)
	{
		ids_.resize(count);
		finalized_ = false;
	}

	void mark_finalized()
	{
		finalized_ = true;
	}

	bool finalized() const
	{
		return finalized_;
	}

private:
	std::vector<long> ids_;
	bool finalized_;
};

#endif
