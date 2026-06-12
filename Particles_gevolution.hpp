//////////////////////////
// Particles_gevolution.hpp
//////////////////////////
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich & ETH Zürich)
//
// Last modified: June 2026
//
//////////////////////////

#ifndef PARTICLES_GEVOLUTION_HEADER
#define PARTICLES_GEVOLUTION_HEADER

#include <cstring>
#include "particles/LATfield2_perfParticles.hpp"
#include "cuda_aware_mpi.hpp"
#include "lightcone_id_backlog.hpp"
#include "device_workspace.hpp"
#include <algorithm>
#include <limits.h>
#include <omp.h>
#include <string.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <vector>

#ifndef PCLBUFFER
#define PCLBUFFER 1048576
#endif

#ifndef PARTICLE_LC_BALANCED_IO
#define PARTICLE_LC_BALANCED_IO 0
#endif

using namespace LATfield2;

struct LightconeParticleWriteChunk
{
	int row_start;
	int row_count;
	long count;
	unsigned long long int check_count;
};

inline long long lightcone_balanced_partition_begin(long long total, int rank, int size)
{
	return (total / size) * rank + ((total % size) * rank) / size;
}

inline void lightcone_mpi_isend_bytes(const void * buffer, size_t bytes, int dest, int tag, MPI_Comm comm, vector<MPI_Request> & requests)
{
	const char * ptr = (const char *) buffer;

	while (bytes > 0)
	{
		int count = (bytes > (size_t) INT_MAX) ? INT_MAX : (int) bytes;
		requests.push_back(MPI_Request());
		MPI_Isend((void *) ptr, count, MPI_BYTE, dest, tag, comm, &requests.back());
		ptr += count;
		bytes -= count;
	}
}

inline void lightcone_mpi_irecv_bytes(void * buffer, size_t bytes, int source, int tag, MPI_Comm comm, vector<MPI_Request> & requests)
{
	char * ptr = (char *) buffer;

	while (bytes > 0)
	{
		int count = (bytes > (size_t) INT_MAX) ? INT_MAX : (int) bytes;
		requests.push_back(MPI_Request());
		MPI_Irecv(ptr, count, MPI_BYTE, source, tag, comm, &requests.back());
		ptr += count;
		bytes -= count;
	}
}

inline void lightcone_mpi_file_write_at_all_bytes(MPI_File file, MPI_Offset offset, void * buffer, unsigned long long int bytes, MPI_Comm comm, MPI_Status * status)
{
	unsigned long long int max_bytes = 0;
	unsigned long long int done = 0;
	char * ptr = (char *) buffer;
	char dummy = 0;

	MPI_Allreduce(&bytes, &max_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);

	while (done < max_bytes)
	{
		unsigned long long int remaining = (done < bytes) ? (bytes - done) : 0;
		int count = (remaining > (unsigned long long int) INT_MAX) ? INT_MAX : (int) remaining;
		void * write_ptr = (count > 0 && ptr != NULL) ? (void *) (ptr + done) : (void *) &dummy;
		MPI_File_write_at_all(file, offset + (MPI_Offset) done, write_ptr, count, MPI_BYTE, status);
		done += (unsigned long long int) INT_MAX;
	}
}

struct express_header
{
	char magic[8];
	uint32_t version;
	uint32_t header_size;
	uint32_t endian;
	uint32_t sizeof_real;
	uint32_t sizeof_long;
	uint32_t sizeof_gadget2_header;
	int32_t rank;
	int32_t size;
	int32_t grid_size[2];
	int32_t lat_size[3];
	int32_t lat_size_local[3];
	int32_t coord_skip[2];
	uint64_t local_npart;
	uint64_t global_npart;
	uint64_t position_bytes;
	uint64_t momentum_bytes;
	uint64_t id_bytes;
};

static const char EXPRESS_MAGIC[8] = {'G', 'E', 'X', 'P', 'C', 'D', 'M', '\0'};
static const uint32_t EXPRESS_VERSION = 1;
static const uint32_t EXPRESS_ENDIAN = 0x01020304u;

inline string express_rank_filename(string filename, int rank)
{
	size_t dot = filename.find_last_of('.');
	if (dot != string::npos && dot + 1 < filename.size())
	{
		bool numeric_suffix = true;
		for (size_t i = dot + 1; i < filename.size(); i++)
		{
			if (filename[i] < '0' || filename[i] > '9')
			{
				numeric_suffix = false;
				break;
			}
		}
		if (numeric_suffix)
			filename.resize(dot);
	}

	return filename + "." + to_string(rank);
}

template <typename part, typename part_info>
class perfParticles_gevolution;

template <typename part, typename part_info>
__global__ void count_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, long * npart, int * npart_row);

template <typename part, typename part_info>
__global__ void count_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry lightcone, Real inner, Real outer, Real dtau_old, const double vertex[MAX_INTERSECTS][3], int vertexcount, long * npart, int * npart_row, int * npart_checkID_row);

template <typename part, typename part_info>
__global__ void buffer_tracer_IDs(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry lightcone, Real inner, Real outer, Real dtau_old, const double vertex[MAX_INTERSECTS][3], int vertexcount, long * IDs, long row_offset, unsigned long long int * buffer_count);

template <typename part, typename part_info>
__global__ void buffer_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, long row_offset, unsigned long long int * buffer_count);

template <typename part, typename part_info, int IDlog_scatter = 0>
__global__ void buffer_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry lightcone, Real dist, Real inner, Real outer, double dtau, double dtau_old, double a, double dadtau, double boxsize, const Real * domain, Field<Real> * phi, const double vertex[MAX_INTERSECTS][3], int vertexcount, float * posdata, float * veldata, long * IDs, unsigned char * loginfo, long row_offset, unsigned long long int * buffer_count1, unsigned long long int * buffer_count2);

template <typename part, typename part_info>
__global__ void add_particles(perfParticles_gevolution<part, part_info> * pcl, float * posdata, float * veldata, void * IDs, uint32_t count, unsigned long long int * buffer_idx);

template <typename part, typename part_info, typename part_dataType>
class Particles_gevolution: public Particles<part, part_info, part_dataType>
{
	public:
		void saveGadget2(string filename, gadget2_header & hdr, const int tracer_factor = 1, double dtau_pos = 0., double dtau_vel = 0., Field<Real> * phi = NULL);
		template <int IDlog_scatter = 0>
		void saveGadget2(string filename, gadget2_header & hdr, lightcone_geometry & lightcone, double dist, double dtau, double dtau_old, double dadtau, double vertex[MAX_INTERSECTS][3], const int vertexcount, LightconeIDBacklog & IDbacklog, vector<long> * IDprelog, Field<Real> * phi, const int tracer_factor = 1);
		void loadGadget2(string filename, gadget2_header & hdr);
};

template <typename part, typename part_info>
class perfParticles_gevolution: public perfParticles<part, part_info>
{
	public:
		void saveGadget2(string filename, gadget2_header & hdr, const int tracer_factor = 1, double dtau_pos = 0., double dtau_vel = 0., Field<Real> * phi = NULL);
		template <int IDlog_scatter = 0>
		void saveGadget2(string filename, gadget2_header & hdr, lightcone_geometry & lightcone, double dist, double dtau, double dtau_old, double dadtau, double vertex[MAX_INTERSECTS][3], const int vertexcount, LightconeIDBacklog & IDbacklog, vector<long> * IDprelog, Field<Real> * phi, const int tracer_factor = 1);
		void loadGadget2(string filename, gadget2_header & hdr);
		void loadGadget2_express(string filename, gadget2_header & hdr);
		void saveExpress(string filename, gadget2_header & hdr);
		void loadExpress(string filename, gadget2_header & hdr);
		uint64_t getTotalCapacity() const
		{
			return this->total_capacity_;
		}

		__host__ __device__ void bufferTracerParticle(int row, int idx, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, unsigned long long int buffer_idx, float * pos_offset = nullptr);

		template <typename part2, typename part_info2>
		friend __global__ void count_tracer_particles(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, long * npart, int * npart_row);

		template <typename part2, typename part_info2>
		friend __global__ void count_tracer_particles(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, lightcone_geometry lightcone, Real inner, Real outer, Real dtau_old, const double vertex[MAX_INTERSECTS][3], int vertexcount, long * npart, int * npart_row, int * npart_checkID_row);

		template <typename part2, typename part_info2>
		friend __global__ void buffer_tracer_IDs(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, lightcone_geometry lightcone, Real inner, Real outer, Real dtau_old, const double vertex[MAX_INTERSECTS][3], int vertexcount, long * IDs, long row_offset, unsigned long long int * buffer_count);

		template <typename part2, typename part_info2>
		friend __global__ void buffer_tracer_particles(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, long row_offset, unsigned long long int * buffer_count);

		template <typename part2, typename part_info2, int IDlog_scatter>
		friend __global__ void buffer_tracer_particles(perfParticles_gevolution<part2, part_info2> * pcl, int tracer_factor, lightcone_geometry lightcone, Real dist, Real inner, Real outer, double dtau, double dtau_old, double a, double dadtau, double boxsize, const Real * domain, Field<Real> * phi, const double vertex[MAX_INTERSECTS][3], int vertexcount, float * posdata, float * veldata, long * IDs, unsigned char * loginfo, long row_offset, unsigned long long int * buffer_count1, unsigned long long int * buffer_count2);

		template <typename part2, typename part_info2>
		friend __global__ void add_particles(perfParticles_gevolution<part2, part_info2> * pcl, float * posdata, float * veldata, void * IDs, uint32_t count, unsigned long long int * buffer_idx);
};

template <typename part, typename part_info, typename part_dataType>
void Particles_gevolution<part,part_info,part_dataType>::saveGadget2(string filename, gadget2_header & hdr, const int tracer_factor, double dtau_pos, double dtau_vel, Field<Real> * phi)
{
	float * posdata;
	float * veldata;
	void * IDs;
	MPI_File outfile;
	long count, npart;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Status status;
	uint32_t blocksize;
	uint32_t i;
	double rescale_vel = 1. / sqrt(hdr.time) / GADGET_VELOCITY_CONVERSION;
#ifdef EXACT_OUTPUT_REDSHIFTS
	Real phip = 0.;
	Real gradphi[3] = {0., 0., 0.};
	double ref_dist[3];
	LATfield2::Site xField;
#endif
	
	LATfield2::Site xPart(this->lat_part_);
	
	if (hdr.num_files != 1 && hdr.num_files != parallel.grid_size()[1])
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": number of Gadget2 files does not match the number of processes in dim-1!" << endl;
		return;
	}
	
	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);

#if GADGET_ID_BYTES == 8
	IDs = malloc(sizeof(int64_t) * PCLBUFFER);
#else
	IDs = malloc(sizeof(int32_t) * PCLBUFFER);
#endif
	
	npart = 0;
	for(xPart.first(); xPart.test(); xPart.next())
	{
		for (auto it=(this->field_part_)(xPart).parts.begin(); it != (this->field_part_)(xPart).parts.end(); ++it)
		{
			if ((*it).ID % tracer_factor == 0)
				npart++;
		}
	}

	if (hdr.num_files == 1)
	{	
		if (parallel.rank() == 0)
		{
			parallel.send<long>(npart, 1);
			parallel.receive<long>(count, parallel.size()-1);
			if (count != hdr.npart[1]) cout << " error: number of particles in saveGadget2 does not match request!" << endl;
			count = 0;
		}
		else
		{
			parallel.receive<long>(count, parallel.rank()-1);
			npart += count;
			parallel.send<long>(npart, (parallel.rank()+1)%parallel.size());
		}
	
		MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);
	}
	else
	{
		if (parallel.grid_rank()[0] == 0)
		{
			parallel.send_dim0<long>(npart, 1);
			parallel.receive_dim0<long>(count, parallel.grid_size()[0]-1);
			hdr.npart[1] = (uint32_t) count;
			count = 0;
		}
		else
		{
			parallel.receive_dim0<long>(count, parallel.grid_rank()[0]-1);
			npart += count;
			parallel.send_dim0<long>(npart, (parallel.grid_rank()[0]+1)%parallel.grid_size()[0]);
		}

		parallel.broadcast_dim0<uint32_t>(hdr.npart[1], 0);
		
		std::string filename_dot_number = filename + "." + std::to_string(parallel.grid_rank()[1]);

		MPI_File_open(parallel.dim0_comm()[parallel.grid_rank()[1]], filename_dot_number.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);
	}
	
	offset_pos = (MPI_Offset) hdr.npart[1];
	offset_pos *= (MPI_Offset) (6 * sizeof(float) + ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)));
	offset_pos += (MPI_Offset) (8 * sizeof(uint32_t) + sizeof(hdr));
	MPI_File_set_size(outfile, offset_pos);
	
	offset_pos = (MPI_Offset) (3 * sizeof(uint32_t) + sizeof(hdr)) + ((MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float)));
	offset_vel = offset_pos + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) hdr.npart[1]) * ((MPI_Offset) (3 * sizeof(float)));
	offset_ID = offset_vel + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) hdr.npart[1] - (MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float))) + ((MPI_Offset) count) * ((MPI_Offset) ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)));
	
	if ((hdr.num_files == 1 && parallel.rank() == 0) || (hdr.num_files > 1 && parallel.grid_rank()[0] == 0))
	{
		blocksize = sizeof(hdr);		
		MPI_File_write_at(outfile, 0, &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, sizeof(uint32_t), &hdr, sizeof(hdr), MPI_BYTE, &status);
		MPI_File_write_at(outfile, sizeof(hdr) + sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		blocksize = 3 * sizeof(float) * hdr.npart[1];
		MPI_File_write_at(outfile, sizeof(hdr) + 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, offset_vel - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, offset_vel - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, offset_ID - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		blocksize = ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)) * hdr.npart[1];
		MPI_File_write_at(outfile, offset_ID - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
		MPI_File_write_at(outfile, offset_ID + blocksize, &blocksize, 1, MPI_UNSIGNED, &status);
	}
	
	count = 0;
#ifdef EXACT_OUTPUT_REDSHIFTS
	if (phi != NULL)
		xField.initialize(phi->lattice());
	else
		xField.initialize(this->lat_part_);
	xField.first();
#endif
	for(xPart.first(); xPart.test(); xPart.next())
	{
		for (auto it=(this->field_part_)(xPart).parts.begin(); it != (this->field_part_)(xPart).parts.end(); ++it)
		{
			if ((*it).ID % tracer_factor == 0)
			{
#ifdef EXACT_OUTPUT_REDSHIFTS
				if (phi != NULL)
				{
					for (i = 0; i < 3; i++)
						ref_dist[i] = modf((*it).pos[i] / this->lat_resolution_, &phip);
							
					phip = (*phi)(xField) * (1.-ref_dist[0]) * (1.-ref_dist[1]) * (1.-ref_dist[2]);
					phip += (*phi)(xField+0) * ref_dist[0] * (1.-ref_dist[1]) * (1.-ref_dist[2]);
					phip += (*phi)(xField+1) * (1.-ref_dist[0]) * ref_dist[1] * (1.-ref_dist[2]);
					phip += (*phi)(xField+0+1) * ref_dist[0] * ref_dist[1] * (1.-ref_dist[2]);
					phip += (*phi)(xField+2) * (1.-ref_dist[0]) * (1.-ref_dist[1]) * ref_dist[2];
					phip += (*phi)(xField+0+2) * ref_dist[0] * (1.-ref_dist[1]) * ref_dist[2];
					phip += (*phi)(xField+1+2) * (1.-ref_dist[0]) * ref_dist[1] * ref_dist[2];
					phip += (*phi)(xField+0+1+2) * ref_dist[0] * ref_dist[1] * ref_dist[2];
					
					gradphi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * ((*phi)(xField+0) - (*phi)(xField));
					gradphi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * ((*phi)(xField+1) - (*phi)(xField));
					gradphi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * ((*phi)(xField+2) - (*phi)(xField));
					gradphi[0] += ref_dist[1] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+1));
					gradphi[1] += ref_dist[0] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+0));
					gradphi[2] += ref_dist[0] * (1.-ref_dist[1]) * ((*phi)(xField+2+0) - (*phi)(xField+0));
					gradphi[0] += (1.-ref_dist[1]) * ref_dist[2] * ((*phi)(xField+2+0) - (*phi)(xField+2));
					gradphi[1] += (1.-ref_dist[0]) * ref_dist[2] * ((*phi)(xField+2+1) - (*phi)(xField+2));
					gradphi[2] += (1.-ref_dist[0]) * ref_dist[1] * ((*phi)(xField+2+1) - (*phi)(xField+1));
					gradphi[0] += ref_dist[1] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+1));
					gradphi[1] += ref_dist[0] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+0));
					gradphi[2] += ref_dist[0] * ref_dist[1] * ((*phi)(xField+2+1+0) - (*phi)(xField+1+0));
				}
					
				ref_dist[0] = (*it).vel[0]*(*it).vel[0] + (*it).vel[1]*(*it).vel[1] + (*it).vel[2]*(*it).vel[2];
				ref_dist[1] = ref_dist[0] + hdr.time * hdr.time;
				ref_dist[2] = sqrt(ref_dist[1]);
				ref_dist[0] += ref_dist[1];
				ref_dist[1] = 1. + (4. - (ref_dist[0] / ref_dist[1])) * phip;
					
				for (i = 0; i < 3; i++)
					posdata[3*count+i] = modf(1. + (*it).pos[i] + dtau_pos * (*it).vel[i] * ref_dist[1] / ref_dist[2], &phip) * hdr.BoxSize;
					
				for (i = 0; i < 3; i++)
					veldata[3*count+i] = ((*it).vel[i] - dtau_vel * ref_dist[0] * gradphi[i] / this->lat_resolution_ / ref_dist[2]) * rescale_vel / hdr.time;
#else						
				for (i = 0; i < 3; i++)
					posdata[3*count+i] = (*it).pos[i] * hdr.BoxSize;
					
				for (i = 0; i < 3; i++)
					veldata[3*count+i] = (*it).vel[i] * rescale_vel / hdr.time;
#endif
					
#if GADGET_ID_BYTES == 8
				*((int64_t *) IDs + count) = (int64_t) (*it).ID;
#else	
				*((int32_t *) IDs + count) = (int32_t) (*it).ID;
#endif
					
				count++;
						
				if (count == PCLBUFFER)
				{
					MPI_File_write_at(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
					offset_pos += 3 * PCLBUFFER * sizeof(float);
					MPI_File_write_at(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
					offset_vel += 3 * PCLBUFFER * sizeof(float);
					count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
					MPI_File_write_at(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
					offset_ID += count;
					count = 0;
				}
			}
		}
#ifdef EXACT_OUTPUT_REDSHIFTS
		xField.next();
#endif
	}
	
	MPI_File_write_at_all(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
	MPI_File_write_at_all(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
	count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
	MPI_File_write_at_all(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
	
	MPI_File_close(&outfile);
	
	free(posdata);
	free(veldata);
	free(IDs);
}


// CUDA kernel to count particles to be written
template <typename part, typename part_info>
__global__ void count_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, long * npart, int * npart_row)
{
	int row = blockIdx.x;
	int thread_id = threadIdx.x;
	long local_count = 0;
	constexpr int W = 32; // warp size
	const int lane = thread_id & (W - 1);
	const int warp_id = thread_id / W;
	constexpr int nWarps = 4; // 128 / W
	unsigned mask = __activemask();

	__shared__ int smem[nWarps];

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
			local_count++;
	}

	#pragma unroll
	for (int ofs = 16; ofs > 0; ofs >>= 1)  // 16,8,4,2,1 = log2(32) steps
	{
		local_count += __shfl_down_sync(mask, local_count, ofs);
	}

	if (lane == 0)
	{
		smem[warp_id] = local_count;
	}
	__syncthreads();

	if (warp_id == 0)
	{
		local_count = (lane < nWarps) ? smem[lane] : 0;

		#pragma unroll
        for (int ofs = 2; ofs > 0; ofs >>= 1)  // 2,1 = log2(4) steps, blockDim.x = 128 => nWarps = 4
		{
			local_count += __shfl_down_sync(mask, local_count, ofs);
		}

		if (lane == 0 && local_count > 0)
		{
			cuda::atomic_ref<long, cuda::thread_scope_device> count_ref(*npart);
			count_ref.fetch_add(local_count);

			cuda::atomic_ref<int, cuda::thread_scope_device> row_count_ref(npart_row[row]);
			row_count_ref.fetch_add(local_count);
		}
	}
}

// CUDA kernel to write particles to buffers
template <typename part, typename part_info>
__global__ void buffer_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, long row_offset, unsigned long long int * buffer_count)
{
	int row = blockIdx.x + row_offset;
	int thread_id = threadIdx.x;

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
		{
			unsigned long long int buffer_idx = atomicAdd(buffer_count, 1);
			
			pcl->bufferTracerParticle(row, idx, dtau_pos, dtau_vel, a, boxsize, phi, posdata, veldata, IDs, buffer_idx);
		}
	}
}

template <typename part, typename part_info>
__host__ __device__ void perfParticles_gevolution<part,part_info>::bufferTracerParticle(int row, int idx, double dtau_pos, double dtau_vel, double a, double boxsize, Field<Real> * phi, float * posdata, float * veldata, long * IDs, unsigned long long int buffer_idx, float * pos_offset)
{
	Real rescale_vel = Real(1) / sqrt(static_cast<Real>(a)) / GADGET_VELOCITY_CONVERSION;

	#ifdef EXACT_OUTPUT_REDSHIFTS
	Real phip = Real(0);
	Real gradphi[3] = {Real(0), Real(0), Real(0)};
	int coord[3];
	Real frac[3];

	Real dx = this->boxSize_[0] / this->lat_size_[0];
	
	if (phi != NULL)
	{
		LATfield2::Site xField;
		constexpr Real one = Real(1);

		coord[0] = (int) floor(this->row_buffers_[row].p[3*idx]*this->lat_size_[0]) % this->lat_size_[0];
		coord[1] = (int) floor(this->row_buffers_[row].p[3*idx+1]*this->lat_size_[1]) % this->lat_size_[1];
		coord[2] = (int) floor(this->row_buffers_[row].p[3*idx+2]*this->lat_size_[2]) % this->lat_size_[2];

		xField = Site(phi->lattice(), phi->lattice().siteFirst() 
				+ coord[0]*phi->lattice().jump(0) 
				+ (coord[1] - this->coordSkip_[1])*phi->lattice().jump(1)
				+ (coord[2] - this->coordSkip_[0])*phi->lattice().jump(2));

		frac[0] = this->row_buffers_[row].p[3*idx]*this->lat_size_[0] - coord[0];
		frac[1] = this->row_buffers_[row].p[3*idx+1]*this->lat_size_[1] - coord[1];
		frac[2] = this->row_buffers_[row].p[3*idx+2]*this->lat_size_[2] - coord[2];
				
		phip = (*phi)(xField) * (one-frac[0]) * (one-frac[1]) * (one-frac[2]);
		phip += (*phi)(xField+0) * frac[0] * (one-frac[1]) * (one-frac[2]);
		phip += (*phi)(xField+1) * (one-frac[0]) * frac[1] * (one-frac[2]);
		phip += (*phi)(xField+0+1) * frac[0] * frac[1] * (one-frac[2]);
		phip += (*phi)(xField+2) * (one-frac[0]) * (one-frac[1]) * frac[2];
		phip += (*phi)(xField+0+2) * frac[0] * (one-frac[1]) * frac[2];
		phip += (*phi)(xField+1+2) * (one-frac[0]) * frac[1] * frac[2];
		phip += (*phi)(xField+0+1+2) * frac[0] * frac[1] * frac[2];
		
		gradphi[0] = (one-frac[1]) * (one-frac[2]) * ((*phi)(xField+0) - (*phi)(xField));
		gradphi[1] = (one-frac[0]) * (one-frac[2]) * ((*phi)(xField+1) - (*phi)(xField));
		gradphi[2] = (one-frac[0]) * (one-frac[1]) * ((*phi)(xField+2) - (*phi)(xField));
		gradphi[0] += frac[1] * (one-frac[2]) * ((*phi)(xField+1+0) - (*phi)(xField+1));
		gradphi[1] += frac[0] * (one-frac[2]) * ((*phi)(xField+1+0) - (*phi)(xField+0));
		gradphi[2] += frac[0] * (one-frac[1]) * ((*phi)(xField+2+0) - (*phi)(xField+0));
		gradphi[0] += (one-frac[1]) * frac[2] * ((*phi)(xField+2+0) - (*phi)(xField+2));
		gradphi[1] += (one-frac[0]) * frac[2] * ((*phi)(xField+2+1) - (*phi)(xField+2));
		gradphi[2] += (one-frac[0]) * frac[1] * ((*phi)(xField+2+1) - (*phi)(xField+1));
		gradphi[0] += frac[1] * frac[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+1));
		gradphi[1] += frac[0] * frac[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+0));
		gradphi[2] += frac[0] * frac[1] * ((*phi)(xField+2+1+0) - (*phi)(xField+1+0));
	}
		
	frac[0] = this->row_buffers_[row].q[3*idx]*this->row_buffers_[row].q[3*idx] + this->row_buffers_[row].q[3*idx+1]*this->row_buffers_[row].q[3*idx+1] + this->row_buffers_[row].q[3*idx+2]*this->row_buffers_[row].q[3*idx+2];
	frac[1] = frac[0] + static_cast<Real>(a) * static_cast<Real>(a);
	frac[2] = sqrt(frac[1]);
	frac[0] += frac[1];
	frac[1] = Real(1) + (Real(4) - (frac[0] / frac[1])) * phip;

	if (pos_offset != nullptr)
	{
		for (int i = 0; i < 3; i++)
			posdata[3*buffer_idx+i] = (static_cast<float>(this->row_buffers_[row].p[3*idx+i] + static_cast<Real>(dtau_pos) * this->row_buffers_[row].q[3*idx+i] * frac[1] / frac[2]) + pos_offset[i]) * static_cast<float>(boxsize);
	}
	else
	{
		for (int i = 0; i < 3; i++)
#ifdef SINGLE
			posdata[3*buffer_idx+i] = modff(float(1) + static_cast<float>(this->row_buffers_[row].p[3*idx+i] + static_cast<Real>(dtau_pos) * this->row_buffers_[row].q[3*idx+i] * frac[1] / frac[2]), static_cast<float *>(&phip)) * static_cast<float>(boxsize);
#else
			posdata[3*buffer_idx+i] = static_cast<float>(modf(double(1) + static_cast<double>(this->row_buffers_[row].p[3*idx+i] + static_cast<Real>(dtau_pos) * this->row_buffers_[row].q[3*idx+i] * frac[1] / frac[2]), static_cast<double *>(&phip)) * static_cast<double>(boxsize));
#endif
	}
		
	for (int i = 0; i < 3; i++)
		veldata[3*buffer_idx+i] = static_cast<float>((this->row_buffers_[row].q[3*idx+i] - static_cast<Real>(dtau_vel) * frac[0] * gradphi[i] / dx / frac[2]) * rescale_vel) / static_cast<float>(a);
#else
	if (pos_offset != nullptr)
	{
		for (int i = 0; i < 3; i++)
			posdata[3*buffer_idx+i] = (static_cast<float>(this->row_buffers_[row].p[3*idx+i]) + pos_offset[i]) * static_cast<float>(boxsize);
	}
	else
	{						
		for (int i = 0; i < 3; i++)
			posdata[3*buffer_idx+i] = static_cast<float>(this->row_buffers_[row].p[3*idx+i]) * static_cast<float>(boxsize);
	}
					
	for (int i = 0; i < 3; i++)
		veldata[3*buffer_idx+i] = static_cast<float>(this->row_buffers_[row].q[3*idx+i] * rescale_vel) / static_cast<float>(a);
#endif

	IDs[buffer_idx] = this->row_buffers_[row].other[idx];
}


template <typename part, typename part_info>
void perfParticles_gevolution<part,part_info>::saveGadget2(string filename, gadget2_header & hdr, const int tracer_factor, double dtau_pos, double dtau_vel, Field<Real> * phi)
{
	float * posdata;
	float * veldata;
	long * IDs;
	long count, npart;
	int row_start = 0, row_count;
	uint32_t blocksize;
	unsigned long long int buffer_count;
	int * npart_row;
	long * d_npart;
	int * d_npart_row;
	unsigned long long int * d_buffer_count;
	float * d_posdata;
	float * d_veldata;
	long * d_IDs;

	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	IDs = (long *) malloc(sizeof(int64_t) * PCLBUFFER);

	npart_row = (int *) malloc(sizeof(int) * this->num_row_buffers_);

	cudaMalloc(&d_npart, sizeof(long));
	cudaMalloc(&d_npart_row, sizeof(int) * this->num_row_buffers_);
	cudaMalloc(&d_buffer_count, sizeof(unsigned long long int));

	if (posdata == NULL || veldata == NULL || IDs == NULL)
	{
		throw std::runtime_error("Error allocating memory for particle buffers");
	}

	nvtxRangePushA("count particles to be written");
	
	npart = 0;
#pragma omp parallel for
	for (int row = 0; row < this->num_row_buffers_; row++)
	{
		npart_row[row] = 0;
	}

	cudaMemcpy(d_npart, &npart, sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_npart_row, npart_row, sizeof(int) * this->num_row_buffers_, cudaMemcpyHostToDevice);
	
	// count particles
	count_tracer_particles<part, part_info><<<this->num_row_buffers_, 128>>>(this, tracer_factor, d_npart, d_npart_row);

	auto success = cudaDeviceSynchronize();

	if (success != cudaSuccess)
	{
		throw std::runtime_error("CUDA error in count_tracer_particles");
	}

	cudaMemcpy(&npart, d_npart, sizeof(long), cudaMemcpyDeviceToHost);
	cudaMemcpy(npart_row, d_npart_row, sizeof(int) * this->num_row_buffers_, cudaMemcpyDeviceToHost);

	nvtxRangePop();

	long max_chunk_count = 0;
	for (int row = 0; row < this->num_row_buffers_; )
	{
		long chunk_count = 0;
		int chunk_rows = 0;
		do
		{
			chunk_count += npart_row[row + chunk_rows];
			chunk_rows++;
		} while (chunk_count < PCLBUFFER && row + chunk_rows < this->num_row_buffers_);
		max_chunk_count = std::max(max_chunk_count, chunk_count);
		row += chunk_rows;
	}

	size_t output_workspace_bytes = DeviceWorkspace::aligned_bytes<float>(3 * max_chunk_count)
	                              + DeviceWorkspace::aligned_bytes<float>(3 * max_chunk_count)
	                              + DeviceWorkspace::aligned_bytes<long>(max_chunk_count);
	DeviceWorkspace output_workspace(output_workspace_bytes, "Gadget-2 snapshot output", "particle staging buffers");
	d_posdata = output_workspace.slice<float>(3 * max_chunk_count, "position staging buffer");
	d_veldata = output_workspace.slice<float>(3 * max_chunk_count, "velocity staging buffer");
	d_IDs = output_workspace.slice<long>(max_chunk_count, "ID staging buffer");

	if (hdr.num_files == 1)
	{	
		if (parallel.rank() == 0)
		{
			parallel.send<long>(npart, 1);
			parallel.receive<long>(count, parallel.size()-1);
			if (count != hdr.npart[1]) cout << " error: number of particles in saveGadget2 does not match request!" << endl;
			count = 0;
		}
		else
		{
			parallel.receive<long>(count, parallel.rank()-1);
			npart += count;
			parallel.send<long>(npart, (parallel.rank()+1)%parallel.size());
		}

		MPI_File outfile;
		MPI_Offset offset_pos, offset_vel, offset_ID;
		MPI_Status status;
	
		MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);

		offset_pos = (MPI_Offset) hdr.npart[1];
		offset_pos *= (MPI_Offset) (6 * sizeof(float) + sizeof(int64_t));
		offset_pos += (MPI_Offset) (8 * sizeof(uint32_t) + sizeof(hdr));
		MPI_File_set_size(outfile, offset_pos);
		
		offset_pos = (MPI_Offset) (3 * sizeof(uint32_t) + sizeof(hdr)) + ((MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float)));
		offset_vel = offset_pos + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) hdr.npart[1]) * ((MPI_Offset) (3 * sizeof(float)));
		offset_ID = offset_vel + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) hdr.npart[1] - (MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float))) + ((MPI_Offset) count) * ((MPI_Offset) sizeof(int64_t));

		if (parallel.rank() == 0)
		{
			blocksize = sizeof(hdr);		
			MPI_File_write_at(outfile, 0, &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, sizeof(uint32_t), &hdr, sizeof(hdr), MPI_BYTE, &status);
			MPI_File_write_at(outfile, sizeof(hdr) + sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = 3 * sizeof(float) * hdr.npart[1];
			MPI_File_write_at(outfile, sizeof(hdr) + 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)) * hdr.npart[1];
			MPI_File_write_at(outfile, offset_ID - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID + blocksize, &blocksize, 1, MPI_UNSIGNED, &status);
		}

		while (row_start < this->num_row_buffers_)
		{
			count = 0;
			row_count = 0;

			do
			{
				count += npart_row[row_start + row_count];
				row_count++;
			} while (count < PCLBUFFER && row_start + row_count < this->num_row_buffers_);

			if (count > PCLBUFFER)
			{
				float * new_posdata = (float *) realloc(posdata, 3 * sizeof(float) * count);
				float * new_veldata = (float *) realloc(veldata, 3 * sizeof(float) * count);
				long * new_IDs = (long *) realloc(IDs, sizeof(int64_t) * count);

				if (new_posdata == NULL || new_veldata == NULL || new_IDs == NULL)
				{
					throw std::runtime_error("Error reallocating memory for particle buffers");
				}

				posdata = new_posdata;
				veldata = new_veldata;
				IDs = new_IDs;
			}
			
			if (count > 0)
			{
				nvtxRangePushA("buffer particles");
				//buffer_count = 0;
				cudaMemset(d_buffer_count, 0, sizeof(unsigned long long int));

				buffer_tracer_particles<<<row_count, 128>>>(this, tracer_factor, dtau_pos, dtau_vel, hdr.time, hdr.BoxSize, phi, d_posdata, d_veldata, d_IDs, row_start, d_buffer_count);

				success = cudaDeviceSynchronize();

				if (success != cudaSuccess)
				{
					throw std::runtime_error("CUDA error in buffer_tracer_particles");
				}

				cudaMemcpy(&buffer_count, d_buffer_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
				cudaMemcpy(posdata, d_posdata, 3 * buffer_count * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(veldata, d_veldata, 3 * buffer_count * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(IDs, d_IDs, buffer_count * sizeof(long), cudaMemcpyDeviceToHost);
				nvtxRangePop();

				nvtxRangePushA("write particles to disk");
				MPI_File_write_at(outfile, offset_pos, posdata, 3 * buffer_count, MPI_FLOAT, &status);
				offset_pos += 3 * buffer_count * sizeof(float);
				MPI_File_write_at(outfile, offset_vel, veldata, 3 * buffer_count, MPI_FLOAT, &status);
				offset_vel += 3 * buffer_count * sizeof(float);
				buffer_count *= sizeof(int64_t);
				MPI_File_write_at(outfile, offset_ID, IDs, buffer_count, MPI_BYTE, &status);
				offset_ID += buffer_count;
				nvtxRangePop();
			}

			row_start += row_count;
		}

		MPI_File_close(&outfile);
	}
	else // writing multiple files independently - no MPI overhead
	{
		if (hdr.num_files != parallel.size())
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": number of Gadget2 files does not match the number of tasks!" << endl;
			return;
		}

		// check total number of particles makes sense
		if (parallel.rank() == 0)
		{
			parallel.send<long>(npart, 1);
			parallel.receive<long>(count, parallel.size()-1);
			if (count != (long) hdr.npartTotal[1] + ((long) hdr.npartTotalHW[1] << 32)) cout << " error: number of particles in saveGadget2 does not match request!" << endl;
		}
		else
		{
			parallel.receive<long>(count, parallel.rank()-1);
			count += npart;
			parallel.send<long>(count, (parallel.rank()+1)%parallel.size());
		}

		hdr.npart[1] = (uint32_t) npart;

		std::string filename_dot_number = filename + "." + std::to_string(parallel.rank());

		FILE * outfile = fopen(filename_dot_number.c_str(), "wb");

		if (outfile == NULL)
		{
			throw std::runtime_error("Error opening output file");
		}

		uint64_t offset_pos = (uint64_t) (3 * sizeof(uint32_t) + sizeof(hdr));
		uint64_t offset_vel = offset_pos + (uint64_t) (2 * sizeof(uint32_t)) + ((uint64_t) hdr.npart[1]) * ((uint64_t) (3 * sizeof(float)));
		uint64_t offset_ID = offset_vel + (uint64_t) (2 * sizeof(uint32_t)) + ((uint64_t) hdr.npart[1]) * ((uint64_t) (3 * sizeof(float)));

		blocksize = sizeof(hdr);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fwrite(&hdr, sizeof(hdr), 1, outfile);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		blocksize = 3 * sizeof(float) * hdr.npart[1];
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fseek(outfile, offset_vel - 2*sizeof(uint32_t), SEEK_SET);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fseek(outfile, offset_ID - 2*sizeof(uint32_t), SEEK_SET);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		blocksize = sizeof(int64_t) * hdr.npart[1];
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);
		fseek(outfile, offset_ID + blocksize, SEEK_SET);
		fwrite(&blocksize, sizeof(uint32_t), 1, outfile);

		while (row_start < this->num_row_buffers_)
		{
			nvtxRangePushA("buffer particles");
			count = 0;
			row_count = 0;

			do
			{
				count += npart_row[row_start + row_count];
				row_count++;
			} while (count < PCLBUFFER && row_start + row_count < this->num_row_buffers_);

			if (count > PCLBUFFER)
			{
				float * new_posdata = (float *) realloc(posdata, 3 * sizeof(float) * count);
				float * new_veldata = (float *) realloc(veldata, 3 * sizeof(float) * count);
				long * new_IDs = (long *) realloc(IDs, sizeof(int64_t) * count);

				if (new_posdata == NULL || new_veldata == NULL || new_IDs == NULL)
				{
					throw std::runtime_error("Error reallocating memory for particle buffers");
				}

				posdata = new_posdata;
				veldata = new_veldata;
				IDs = new_IDs;
			}
			
			//buffer_count = 0;
			cudaMemset(d_buffer_count, 0, sizeof(unsigned long long int));

			buffer_tracer_particles<<<row_count, 128>>>(this, tracer_factor, dtau_pos, dtau_vel, hdr.time, hdr.BoxSize, phi, d_posdata, d_veldata, d_IDs, row_start, d_buffer_count);

			success = cudaDeviceSynchronize();

			if (success != cudaSuccess)
			{
				throw std::runtime_error("CUDA error in buffer_tracer_particles");
			}
			cudaMemcpy(&buffer_count, d_buffer_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
			cudaMemcpy(posdata, d_posdata, 3 * buffer_count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(veldata, d_veldata, 3 * buffer_count * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(IDs, d_IDs, buffer_count * sizeof(long), cudaMemcpyDeviceToHost);
			nvtxRangePop();

			nvtxRangePushA("write particles to disk");
			fseek(outfile, offset_pos, SEEK_SET);
			fwrite(posdata, 3 * buffer_count, sizeof(float), outfile);
			offset_pos += 3 * buffer_count * sizeof(float);
			fseek(outfile, offset_vel, SEEK_SET);
			fwrite(veldata, 3 * buffer_count, sizeof(float), outfile);
			offset_vel += 3 * buffer_count * sizeof(float);
			buffer_count *= sizeof(int64_t);
			fseek(outfile, offset_ID, SEEK_SET);
			fwrite(IDs, buffer_count, 1, outfile);
			offset_ID += buffer_count;
			nvtxRangePop();

			row_start += row_count;
		}

		fclose(outfile);
	}

	free(posdata);
	free(veldata);
	free(IDs);
	free(npart_row);

	cudaFree(d_npart);
	cudaFree(d_npart_row);
	cudaFree(d_buffer_count);
}


template <typename part, typename part_info, typename part_dataType>
template <int IDlog_scatter>
void Particles_gevolution<part,part_info,part_dataType>::saveGadget2(string filename, gadget2_header & hdr, lightcone_geometry & lightcone, double dist, double dtau, double dtau_old, double dadtau, double vertex[MAX_INTERSECTS][3], const int vertexcount, LightconeIDBacklog & IDbacklog, vector<long> * IDprelog, Field<Real> * phi, const int tracer_factor)
{
	float * posdata;
	float * veldata;
	void * IDs;
	MPI_File outfile;
	long count, npart;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Status status;
	uint32_t blocksize;
	uint32_t i;
	double rescale_vel = 1. / GADGET_VELOCITY_CONVERSION;
	double inner = dist - 0.5 * dtau;
	double outer = dist + (0.5 + LIGHTCONE_IDCHECK_ZONE) * dtau_old;
	double d, v2, e2, vlos;
	double ref_dist[3];
	Real gradphi[3];
	
	LATfield2::Site xPart(this->lat_part_);
	LATfield2::Site xField(phi->lattice());

	double domain[4];

	domain[0] = this->lat_part_.coordSkip()[1];
	domain[1] = domain[0] + this->lat_part_.sizeLocal(1);
	domain[2] = this->lat_part_.coordSkip()[0];
	domain[3] = domain[2] + this->lat_part_.sizeLocal(2);

	for (int j = 0; j < 4; j++)
		domain[j] *= this->lat_resolution_;
	
	if (hdr.num_files != 1)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": writing multiple Gadget2 files not currently supported!" << endl;
		return;
	}
	
	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);

#if GADGET_ID_BYTES == 8
	IDs = malloc(sizeof(int64_t) * PCLBUFFER);
#else
	IDs = malloc(sizeof(int32_t) * PCLBUFFER);
#endif
	
	npart = 0;
	if (vertexcount > 0)
	{
		for(xPart.first(), xField.first(); xPart.test(); xPart.next(), xField.next())
		{
			for (auto it=(this->field_part_)(xPart).parts.begin(); it != (this->field_part_)(xPart).parts.end(); ++it)
			{
				if ((*it).ID % tracer_factor == 0)
				{
					for (i = 0; i < (uint32_t) vertexcount; i++)
					{
						d = sqrt(((*it).pos[0]-vertex[i][0])*((*it).pos[0]-vertex[i][0]) + ((*it).pos[1]-vertex[i][1])*((*it).pos[1]-vertex[i][1]) + ((*it).pos[2]-vertex[i][2])*((*it).pos[2]-vertex[i][2]));

						if (d < inner || d >= outer) continue;

						if (lightcone.opening == -1. || (((*it).pos[0]-vertex[i][0])*lightcone.direction[0] + ((*it).pos[1]-vertex[i][1])*lightcone.direction[1] + ((*it).pos[2]-vertex[i][2])*lightcone.direction[2]) / d > lightcone.opening)
						{
							if (outer - d > 2. * LIGHTCONE_IDCHECK_ZONE * dtau_old || !IDbacklog.contains((*it).ID))
							{
								if (d - inner < 2. * LIGHTCONE_IDCHECK_ZONE * dtau)
								{
									if (IDlog_scatter)
									{
										if ((*it).pos[1] - domain[0] < LIGHTCONE_IDCHECK_ZONE * dtau) // left edge
										{
											if ((*it).pos[2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower left corner
												IDprelog[0].push_back((*it).ID);
											else if (domain[3] - (*it).pos[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper left corner
												IDprelog[6].push_back((*it).ID);
											else
												IDprelog[3].push_back((*it).ID);
										}
										else if (domain[1] - (*it).pos[1] < LIGHTCONE_IDCHECK_ZONE * dtau) // right edge
										{
											if ((*it).pos[2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower right corner
												IDprelog[2].push_back((*it).ID);
											else if (domain[3] - (*it).pos[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper right corner
												IDprelog[8].push_back((*it).ID);
											else
												IDprelog[5].push_back((*it).ID);
										}
										else
										{
											if ((*it).pos[2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower edge
												IDprelog[1].push_back((*it).ID);
											else if (domain[3] - (*it).pos[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper edge
												IDprelog[7].push_back((*it).ID);
											else
												IDprelog[4].push_back((*it).ID);
										}
									}
									else
										(*IDprelog).push_back((*it).ID);
								}

								for (int j = 0; j < 3; j++)
									ref_dist[j] = modf((*it).pos[j] / this->lat_resolution_, &v2);
									
								v2 = (*it).vel[0] * (*it).vel[0] + (*it).vel[1] * (*it).vel[1] + (*it).vel[2] * (*it).vel[2];
								e2 = v2 + hdr.time * (hdr.time + (dist - d - 0.5 * dtau_old) * dadtau);
								vlos = ((*it).vel[0]*((*it).pos[0]-vertex[i][0]) + (*it).vel[1]*((*it).pos[1]-vertex[i][1]) + (*it).vel[2]*((*it).pos[2]-vertex[i][2])) / d;
	
								gradphi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * ((*phi)(xField+0) - (*phi)(xField));
								gradphi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * ((*phi)(xField+1) - (*phi)(xField));
								gradphi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * ((*phi)(xField+2) - (*phi)(xField));
								gradphi[0] += ref_dist[1] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+1));
								gradphi[1] += ref_dist[0] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+0));
								gradphi[2] += ref_dist[0] * (1.-ref_dist[1]) * ((*phi)(xField+2+0) - (*phi)(xField+0));
								gradphi[0] += (1.-ref_dist[1]) * ref_dist[2] * ((*phi)(xField+2+0) - (*phi)(xField+2));
								gradphi[1] += (1.-ref_dist[0]) * ref_dist[2] * ((*phi)(xField+2+1) - (*phi)(xField+2));
								gradphi[2] += (1.-ref_dist[0]) * ref_dist[1] * ((*phi)(xField+2+1) - (*phi)(xField+1));
								gradphi[0] += ref_dist[1] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+1));
								gradphi[1] += ref_dist[0] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+0));
								gradphi[2] += ref_dist[0] * ref_dist[1] * ((*phi)(xField+2+1+0) - (*phi)(xField+1+0));

								gradphi[0] *= (v2 + e2) / e2 / this->lat_resolution_;
								gradphi[1] *= (v2 + e2) / e2 / this->lat_resolution_;
								gradphi[2] *= (v2 + e2) / e2 / this->lat_resolution_;
						
								e2 = sqrt(e2);
									
								if (d < dist)
								{
									vlos -= dtau * sqrt(v2 + hdr.time * hdr.time) * (gradphi[0]*((*it).pos[0]-vertex[i][0]) + gradphi[1]*((*it).pos[1]-vertex[i][1]) + gradphi[2]*((*it).pos[2]-vertex[i][2])) / d;
									vlos /= sqrt(v2 + hdr.time * (hdr.time + dtau * dadtau));
								}
								else
									vlos /= sqrt(v2 + hdr.time * (hdr.time - dtau_old * dadtau));

								for (uint32_t j = 0; j < 3; j++)
									veldata[3*(npart%PCLBUFFER)+j] = ((*it).vel[j] - (((dist - d) / (1. + vlos)) + 0.5 * dtau_old) * e2 * gradphi[j]) * rescale_vel / (hdr.time + ((dist - d) / (1. + vlos)) * dadtau);
										
								if (d >= dist)
								{
									e2 = sqrt(v2 + hdr.time * (hdr.time - dtau_old * dadtau));
										
									for (uint32_t j = 0; j < 3; j++)
										posdata[3*(npart%PCLBUFFER)+j] = ((*it).pos[j] - vertex[i][j] + lightcone.vertex[j] + ((dist - d) / (1. + vlos)) * (*it).vel[j] / e2) * hdr.BoxSize;
								}
								else
								{
									e2 = sqrt(v2 + hdr.time * (hdr.time + dtau * dadtau));
									v2 = sqrt(v2 + hdr.time * hdr.time);
										
									for (uint32_t j = 0; j < 3; j++)
										posdata[3*(npart%PCLBUFFER)+j] = ((*it).pos[j] - vertex[i][j] + lightcone.vertex[j] + ((dist - d) / (1. + vlos)) * ((*it).vel[j] - dtau * v2 * gradphi[j]) / e2) * hdr.BoxSize;
								}
					
#if GADGET_ID_BYTES == 8
								*((int64_t *) IDs + (npart%PCLBUFFER)) = (int64_t) (*it).ID;
#else	
								*((int32_t *) IDs + (npart%PCLBUFFER)) = (int32_t) (*it).ID;
#endif

								npart++;
							}

							break;
						}
					}
				}
			}
		}
	}
	
	if (parallel.rank() == 0)
	{
		parallel.send<long>(npart, 1);
		parallel.receive<long>(count, parallel.size()-1);
		hdr.npart[1] = (uint32_t) (count % (1ll << 32));
		hdr.npartTotal[1] = (uint32_t) (count % (1ll << 32));
		hdr.npartTotalHW[1] = (uint32_t) (count / (1ll << 32));
		count = 0;
	}
	else
	{
		parallel.receive<long>(count, parallel.rank()-1);
		count += npart;
		parallel.send<long>(count, (parallel.rank()+1)%parallel.size());
		count -= npart;
	}

	parallel.broadcast<uint32_t>(hdr.npartTotal[1], 0);
	parallel.broadcast<uint32_t>(hdr.npartTotalHW[1], 0);

	if (hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32) > 0)
	{
		MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);
	
		offset_pos = (MPI_Offset) ((int64_t) hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32));
		offset_pos *= (MPI_Offset) (6 * sizeof(float) + ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)));
		offset_pos += (MPI_Offset) (8 * sizeof(uint32_t) + sizeof(hdr));
		MPI_File_set_size(outfile, offset_pos);
	
		offset_pos = (MPI_Offset) (3 * sizeof(uint32_t) + sizeof(hdr)) + ((MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float)));
		offset_vel = offset_pos + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) ((int64_t) hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32))) * ((MPI_Offset) (3 * sizeof(float)));
		offset_ID = offset_vel + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) ((int64_t) hdr.npartTotal[1] + ((int64_t) hdr.npartTotalHW[1] << 32)) - (MPI_Offset) count) * ((MPI_Offset) (3 * sizeof(float))) + ((MPI_Offset) count) * ((MPI_Offset) ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)));
	
		if (parallel.rank() == 0)
		{
			blocksize = sizeof(hdr);		
			MPI_File_write_at(outfile, 0, &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, sizeof(uint32_t), &hdr, sizeof(hdr), MPI_BYTE, &status);
			MPI_File_write_at(outfile, sizeof(hdr) + sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = 3 * sizeof(float) * hdr.npart[1];
			MPI_File_write_at(outfile, sizeof(hdr) + 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)) * hdr.npart[1];
			MPI_File_write_at(outfile, offset_ID - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID + blocksize, &blocksize, 1, MPI_UNSIGNED, &status);
		}
		
		count = (npart < PCLBUFFER) ? npart : PCLBUFFER;
		npart -= count;
		MPI_File_write_at_all(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
		offset_pos += 3 * count * sizeof(float);
		MPI_File_write_at_all(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
		offset_vel += 3 * count * sizeof(float);
		count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
		MPI_File_write_at_all(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
		offset_ID += count;
		count = 0;

		if (npart > 0)
		{
			for(xPart.first(); xPart.test() && npart > 0; xPart.next())
			{
				for (auto it=(this->field_part_)(xPart).parts.begin(); it != (this->field_part_)(xPart).parts.end(); ++it)
				{
					if ((*it).ID % tracer_factor == 0)
					{
						for (i = 0; i < (uint32_t) vertexcount; i++)
						{
							d = sqrt(((*it).pos[0]-vertex[i][0])*((*it).pos[0]-vertex[i][0]) + ((*it).pos[1]-vertex[i][1])*((*it).pos[1]-vertex[i][1]) + ((*it).pos[2]-vertex[i][2])*((*it).pos[2]-vertex[i][2]));

							if (d < inner || d >= outer) continue;

							if (lightcone.opening == -1. || (((*it).pos[0]-vertex[i][0])*lightcone.direction[0] + ((*it).pos[1]-vertex[i][1])*lightcone.direction[1] + ((*it).pos[2]-vertex[i][2])*lightcone.direction[2]) / d > lightcone.opening)
							{
								if (outer - d > 2. * LIGHTCONE_IDCHECK_ZONE * dtau_old || !IDbacklog.contains((*it).ID))
								{
									for (int j = 0; j < 3; j++)
										ref_dist[j] = modf((*it).pos[j] / this->lat_resolution_, &v2);
									
									v2 = (*it).vel[0] * (*it).vel[0] + (*it).vel[1] * (*it).vel[1] + (*it).vel[2] * (*it).vel[2];
									e2 = v2 + hdr.time * (hdr.time + (dist - d - 0.5 * dtau_old) * dadtau);
									vlos = ((*it).vel[0]*((*it).pos[0]-vertex[i][0]) + (*it).vel[1]*((*it).pos[1]-vertex[i][1]) + (*it).vel[2]*((*it).pos[2]-vertex[i][2])) / d;
	
									gradphi[0] = (1.-ref_dist[1]) * (1.-ref_dist[2]) * ((*phi)(xField+0) - (*phi)(xField));
									gradphi[1] = (1.-ref_dist[0]) * (1.-ref_dist[2]) * ((*phi)(xField+1) - (*phi)(xField));
									gradphi[2] = (1.-ref_dist[0]) * (1.-ref_dist[1]) * ((*phi)(xField+2) - (*phi)(xField));
									gradphi[0] += ref_dist[1] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+1));
									gradphi[1] += ref_dist[0] * (1.-ref_dist[2]) * ((*phi)(xField+1+0) - (*phi)(xField+0));
									gradphi[2] += ref_dist[0] * (1.-ref_dist[1]) * ((*phi)(xField+2+0) - (*phi)(xField+0));
									gradphi[0] += (1.-ref_dist[1]) * ref_dist[2] * ((*phi)(xField+2+0) - (*phi)(xField+2));
									gradphi[1] += (1.-ref_dist[0]) * ref_dist[2] * ((*phi)(xField+2+1) - (*phi)(xField+2));
									gradphi[2] += (1.-ref_dist[0]) * ref_dist[1] * ((*phi)(xField+2+1) - (*phi)(xField+1));
									gradphi[0] += ref_dist[1] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+1));
									gradphi[1] += ref_dist[0] * ref_dist[2] * ((*phi)(xField+2+1+0) - (*phi)(xField+2+0));
									gradphi[2] += ref_dist[0] * ref_dist[1] * ((*phi)(xField+2+1+0) - (*phi)(xField+1+0));

									gradphi[0] *= (v2 + e2) / e2 / this->lat_resolution_;
									gradphi[1] *= (v2 + e2) / e2 / this->lat_resolution_;
									gradphi[2] *= (v2 + e2) / e2 / this->lat_resolution_;
						
									e2 = sqrt(e2);
										
									if (d < dist)
									{
										vlos -= dtau * sqrt(v2 + hdr.time * hdr.time) * (gradphi[0]*((*it).pos[0]-vertex[i][0]) + gradphi[1]*((*it).pos[1]-vertex[i][1]) + gradphi[2]*((*it).pos[2]-vertex[i][2])) / d;
										vlos /= sqrt(v2 + hdr.time * (hdr.time + dtau * dadtau));
									}
									else
										vlos /= sqrt(v2 + hdr.time * (hdr.time - dtau_old * dadtau));

									for (uint32_t j = 0; j < 3; j++)
										veldata[3*(npart%PCLBUFFER)+j] = ((*it).vel[j] - (((dist - d) / (1. + vlos)) + 0.5 * dtau_old) * e2 * gradphi[j]) * rescale_vel / (hdr.time + ((dist - d) / (1. + vlos)) * dadtau);
											
									if (d >= dist)
									{
										e2 = sqrt(v2 + hdr.time * (hdr.time - dtau_old * dadtau));
										
										for (uint32_t j = 0; j < 3; j++)
											posdata[3*(npart%PCLBUFFER)+j] = ((*it).pos[j] - vertex[i][j] + lightcone.vertex[j] + ((dist - d) / (1. + vlos)) * (*it).vel[j] / e2) * hdr.BoxSize;
									}
									else
									{
										e2 = sqrt(v2 + hdr.time * (hdr.time + dtau * dadtau));
										v2 = sqrt(v2 + hdr.time * hdr.time);
										
										for (uint32_t j = 0; j < 3; j++)
											posdata[3*(npart%PCLBUFFER)+j] = ((*it).pos[j] - vertex[i][j] + lightcone.vertex[j] + ((dist - d) / (1. + vlos)) * ((*it).vel[j] - dtau * v2 * gradphi[j]) / e2) * hdr.BoxSize;
									}
						
#if GADGET_ID_BYTES == 8
									*((int64_t *) IDs + count) = (int64_t) (*it).ID;
#else	
									*((int32_t *) IDs + count) = (int32_t) (*it).ID;
#endif
	
									npart--;
									count++;
								}
								break;
							}
						}
							
						if (count == PCLBUFFER)
						{
							MPI_File_write_at(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
							offset_pos += 3 * PCLBUFFER * sizeof(float);
							MPI_File_write_at(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
							offset_vel += 3 * PCLBUFFER * sizeof(float);
							count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
							MPI_File_write_at(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
							offset_ID += count;
							count = 0;
						}

						if (npart <= 0) break;
					}
				}
			}

			if (count > 0)
			{
					MPI_File_write_at(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
					MPI_File_write_at(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
					count *= (GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t);
					MPI_File_write_at(outfile, offset_ID, IDs, count, MPI_BYTE, &status);
			}
		}	
	
		MPI_File_close(&outfile);
	}
	
	free(posdata);
	free(veldata);
	free(IDs);
}


// CUDA kernel to count particles to be written
template <typename part, typename part_info>
__global__ void count_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry lightcone, Real inner, Real outer, Real dtau_old, const double vertex[MAX_INTERSECTS][3], int vertexcount, long * npart, int * npart_row, int * npart_checkID_row)
{
	int row = blockIdx.x;
	int thread_id = threadIdx.x;
	long local_count = 0;
	long local_count_checkID = 0;
	constexpr int W = 32; // warp size
	const int lane = thread_id & (W - 1);
	const int warp_id = thread_id / W;
	constexpr int nWarps = 4; // 128 / W
	unsigned mask = __activemask();

	__shared__ int smem[2 * nWarps];

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
		{
			for (int v = 0; v < vertexcount; v++)
			{
				Real d = sqrt((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*(pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0])) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*(pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1])) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*(pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2])));

				if ((d >= inner && d < outer) && (lightcone.opening == -1. || ((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*static_cast<Real>(lightcone.direction[0]) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*static_cast<Real>(lightcone.direction[1]) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*static_cast<Real>(lightcone.direction[2])) / d > static_cast<Real>(lightcone.opening)))
				{
					if (outer - d <= Real(2) * LIGHTCONE_IDCHECK_ZONE * dtau_old)
					{
						local_count_checkID++;
					}
					
					local_count++;
					break;
				}
			}
		}
	}

	#pragma unroll
	for (int ofs = 16; ofs > 0; ofs >>= 1)  // 16,8,4,2,1 = log2(32) steps
	{
		local_count += __shfl_down_sync(mask, local_count, ofs);
		local_count_checkID += __shfl_down_sync(mask, local_count_checkID, ofs);
	}

	if (lane == 0)
	{
		smem[2 * warp_id] = local_count;
		smem[2 * warp_id + 1] = local_count_checkID;
	}
	__syncthreads();

	if (warp_id == 0)
	{
		local_count = (lane < nWarps) ? smem[2 * lane] : 0;
		local_count_checkID = (lane < nWarps) ? smem[2 * lane + 1] : 0;

		#pragma unroll
        for (int ofs = 2; ofs > 0; ofs >>= 1)  // 2,1 = log2(4) steps, blockDim.x = 128 => nWarps = 4
		{
			local_count += __shfl_down_sync(mask, local_count, ofs);
			local_count_checkID += __shfl_down_sync(mask, local_count_checkID, ofs);
		}

		if (lane == 0 && local_count > 0)
		{
			cuda::atomic_ref<long, cuda::thread_scope_device> count_ref(*npart);
			count_ref.fetch_add(local_count);

			cuda::atomic_ref<int, cuda::thread_scope_device> row_count_ref(npart_row[row]);
			row_count_ref.fetch_add(local_count);

			cuda::atomic_ref<int, cuda::thread_scope_device> row_count_checkID_ref(npart_checkID_row[row]);
			row_count_checkID_ref.fetch_add(local_count_checkID);
		}
	}
}


// CUDA kernel to write particle IDs to buffers
template <typename part, typename part_info>
__global__ void buffer_tracer_IDs(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry lightcone, Real inner, Real outer, Real dtau_old, const double vertex[MAX_INTERSECTS][3], int vertexcount, long * IDs, long row_offset, unsigned long long int * buffer_count)
{
	int row = blockIdx.x + row_offset;
	int thread_id = threadIdx.x;

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
		{
			for (int v = 0; v < vertexcount; v++)
			{
				Real d = sqrt((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*(pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0])) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*(pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1])) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*(pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2])));

				if ((d >= inner && d < outer) && (lightcone.opening == -1. || ((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*static_cast<Real>(lightcone.direction[0]) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*static_cast<Real>(lightcone.direction[1]) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*static_cast<Real>(lightcone.direction[2])) / d > static_cast<Real>(lightcone.opening)))
				{
					if (outer - d <= Real(2) * LIGHTCONE_IDCHECK_ZONE * dtau_old)
					{
						unsigned long long int buffer_idx = atomicAdd(buffer_count, 1);
						IDs[buffer_idx] = pcl->row_buffers_[row].other[idx];
					}

					break;
				}
			}
		}
	}
}


// CUDA kernel to write particles to buffers
template <typename part, typename part_info, int IDlog_scatter>
__global__ void buffer_tracer_particles(perfParticles_gevolution<part, part_info> * pcl, int tracer_factor, lightcone_geometry lightcone, Real dist, Real inner, Real outer, double dtau, double dtau_old, double a, double dadtau, double boxsize, const Real * domain, Field<Real> * phi, const double vertex[MAX_INTERSECTS][3], int vertexcount, float * posdata, float * veldata, long * IDs, unsigned char * loginfo, long row_offset, unsigned long long int * buffer_count1, unsigned long long int * buffer_count2)
{
	int row = blockIdx.x + row_offset;
	int thread_id = threadIdx.x;
	unsigned long long int buffer_idx;

	for (int idx = thread_id; idx < pcl->row_buffers_[row].count; idx += 128)
	{
		if (pcl->row_buffers_[row].other[idx] % tracer_factor == 0)
		{
			for (int v = 0; v < vertexcount; v++)
			{
				Real d = sqrt((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*(pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0])) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*(pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1])) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*(pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2])));

				if ((d >= inner && d < outer) && (lightcone.opening == -1. || ((pcl->row_buffers_[row].p[3*idx]-static_cast<Real>(vertex[v][0]))*static_cast<Real>(lightcone.direction[0]) + (pcl->row_buffers_[row].p[3*idx+1]-static_cast<Real>(vertex[v][1]))*static_cast<Real>(lightcone.direction[1]) + (pcl->row_buffers_[row].p[3*idx+2]-static_cast<Real>(vertex[v][2]))*static_cast<Real>(lightcone.direction[2])) / d > static_cast<Real>(lightcone.opening)))
				{
					if (outer - d <= Real(2) * LIGHTCONE_IDCHECK_ZONE * dtau_old)
					{
						buffer_idx = atomicAdd(buffer_count1, 1);
					}
					else
					{
						buffer_idx = atomicAdd(buffer_count2, 1);
					}

					if (d - inner < Real(2) * LIGHTCONE_IDCHECK_ZONE * dtau)
					{
						if (IDlog_scatter)
						{
							if (pcl->row_buffers_[row].p[3*idx+1] - domain[0] < LIGHTCONE_IDCHECK_ZONE * dtau) // left edge
							{
								if (pcl->row_buffers_[row].p[3*idx+2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower left corner
									loginfo[buffer_idx] = 0;
								else if (domain[3] - pcl->row_buffers_[row].p[3*idx+2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper left corner
									loginfo[buffer_idx] = 6;
								else
									loginfo[buffer_idx] = 3;
							}
							else if (domain[1] - pcl->row_buffers_[row].p[3*idx+1] < LIGHTCONE_IDCHECK_ZONE * dtau) // right edge
							{
								if (pcl->row_buffers_[row].p[3*idx+2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower right corner
									loginfo[buffer_idx] = 2;
								else if (domain[3] - pcl->row_buffers_[row].p[3*idx+2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper right corner
									loginfo[buffer_idx] = 8;
								else
									loginfo[buffer_idx] = 5;
							}
							else
							{
								if (pcl->row_buffers_[row].p[3*idx+2] - domain[2] < LIGHTCONE_IDCHECK_ZONE * dtau) // lower edge
									loginfo[buffer_idx] = 1;
								else if (domain[3] - pcl->row_buffers_[row].p[3*idx+2] < LIGHTCONE_IDCHECK_ZONE * dtau) // upper edge
									loginfo[buffer_idx] = 7;
								else
									loginfo[buffer_idx] = 4;
							}
						}
						else
						{
							loginfo[buffer_idx] = 0;
						}
					}
					else
					{
						loginfo[buffer_idx] = 255;
					}

					Real vlos = sqrt(pcl->row_buffers_[row].q[3*idx]*pcl->row_buffers_[row].q[3*idx] + pcl->row_buffers_[row].q[3*idx+1]*pcl->row_buffers_[row].q[3*idx+1] + pcl->row_buffers_[row].q[3*idx+2]*pcl->row_buffers_[row].q[3*idx+2] + static_cast<Real>(a * (a + dtau_old * dadtau)));
						
					vlos = (pcl->row_buffers_[row].q[3*idx]*(pcl->row_buffers_[row].p[3*idx]-vertex[v][0]) + pcl->row_buffers_[row].q[3*idx+1]*(pcl->row_buffers_[row].p[3*idx+1]-vertex[v][1]) + pcl->row_buffers_[row].q[3*idx+2]*(pcl->row_buffers_[row].p[3*idx+2]-vertex[v][2])) / d / vlos;

					double dtau_pos = (dist - d) / (1. + vlos);
					float pos_offset[3];

					pos_offset[0] = lightcone.vertex[0] - vertex[v][0];
					pos_offset[1] = lightcone.vertex[1] - vertex[v][1];
					pos_offset[2] = lightcone.vertex[2] - vertex[v][2];

					pcl->bufferTracerParticle(row, idx, dtau_pos, dtau_pos + 0.5 * dtau_old, a + dtau_pos * dadtau, boxsize, phi, posdata, veldata, IDs, buffer_idx, pos_offset);

					break;
				}
			}
		}
	}
}

__device__ inline bool lightcone_backlog_contains_device(const long * backlog, long backlog_count, long id)
{
	long lo = 0;
	long hi = backlog_count;

	while (lo < hi)
	{
		long mid = lo + (hi - lo) / 2;
		long value = backlog[mid];

		if (value < id)
			lo = mid + 1;
		else
			hi = mid;
	}

	return lo < backlog_count && backlog[lo] == id;
}

__global__ void count_lightcone_duplicate_ids(const long * IDs, unsigned long long int count, const long * backlog, long backlog_count, unsigned long long int * duplicates)
{
	unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long int stride = blockDim.x * gridDim.x;
	unsigned long long int local_count = 0;

	for (; idx < count; idx += stride)
	{
		if (lightcone_backlog_contains_device(backlog, backlog_count, IDs[idx]))
			local_count++;
	}

	if (local_count > 0)
		atomicAdd(duplicates, local_count);
}

__global__ void mark_lightcone_kept_particles(const long * IDs, unsigned long long int count, unsigned long long int check_count, const long * backlog, long backlog_count, unsigned char * keep_flags)
{
	unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long int stride = blockDim.x * gridDim.x;

	for (; idx < count; idx += stride)
	{
		keep_flags[idx] = (idx >= check_count || !lightcone_backlog_contains_device(backlog, backlog_count, IDs[idx])) ? 1 : 0;
	}
}

__global__ void gather_lightcone_kept_particles(const float * pos_in, const float * vel_in, const long * ids_in, const unsigned char * log_in, float * pos_out, float * vel_out, long * ids_out, unsigned char * log_out, const unsigned long long int * selected, unsigned long long int count)
{
	unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long int stride = blockDim.x * gridDim.x;

	for (; idx < count; idx += stride)
	{
		unsigned long long int src = selected[idx];

		for (int j = 0; j < 3; j++)
		{
			pos_out[3*idx+j] = pos_in[3*src+j];
			vel_out[3*idx+j] = vel_in[3*src+j];
		}

		ids_out[idx] = ids_in[src];
		log_out[idx] = log_in[src];
	}
}

__global__ void count_lightcone_prelog_bins(const unsigned char * loginfo, unsigned long long int count, int log_bins, unsigned long long int * bin_counts)
{
	unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long int stride = blockDim.x * gridDim.x;

	for (; idx < count; idx += stride)
	{
		if (loginfo[idx] < 255)
		{
			int bin = (log_bins > 1) ? loginfo[idx] : 0;
			atomicAdd(bin_counts + bin, 1);
		}
	}
}

__global__ void fill_lightcone_prelog_bins(const long * IDs, const unsigned char * loginfo, unsigned long long int count, int log_bins, const unsigned long long int * bin_offsets, unsigned long long int * bin_cursors, long * prelog_ids)
{
	unsigned long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long int stride = blockDim.x * gridDim.x;

	for (; idx < count; idx += stride)
	{
		if (loginfo[idx] < 255)
		{
			int bin = (log_bins > 1) ? loginfo[idx] : 0;
			unsigned long long int dst = bin_offsets[bin] + atomicAdd(bin_cursors + bin, 1);
			prelog_ids[dst] = IDs[idx];
		}
	}
}

template <typename part, typename part_info>
template <int IDlog_scatter>
void perfParticles_gevolution<part,part_info>::saveGadget2(string filename, gadget2_header & hdr, lightcone_geometry & lightcone, double dist, double dtau, double dtau_old, double dadtau, double vertex[MAX_INTERSECTS][3], const int vertexcount, LightconeIDBacklog & IDbacklog, vector<long> * IDprelog, Field<Real> * phi, const int tracer_factor)
{
	float * posdata;
	float * veldata;
	long * IDs;
	float * d_posdata = NULL;
	float * d_veldata = NULL;
	float * d_posdata_compact = NULL;
	float * d_veldata_compact = NULL;
	long * d_IDs = NULL;
	long * d_IDs_compact = NULL;
	long * d_IDbacklog = NULL;
	long * d_prelog_ids = NULL;
	unsigned char * d_loginfo = NULL;
	unsigned char * d_loginfo_compact = NULL;
	unsigned char * d_keep_flags = NULL;
	unsigned long long int * d_indices = NULL;
	unsigned long long int * d_selected = NULL;
	unsigned long long int * d_selected_count = NULL;
	unsigned long long int * d_prelog_counts = NULL;
	unsigned long long int * d_prelog_offsets = NULL;
	unsigned long long int * d_prelog_cursors = NULL;
	long count, npart, reject;
	long backlog_count = IDbacklog.size();
	int row_start = 0, row_count;
	MPI_File outfile;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Offset offset_pos_base, offset_vel_base, offset_ID_base;
	MPI_Status status;
	uint32_t blocksize;
	unsigned long long int buffer_count1, buffer_count2;
	int * npart_row;
	int * npart_checkID_row;
	Real domain[4];
	Real inner = dist - 0.5 * dtau;
	Real outer = dist + (0.5 + LIGHTCONE_IDCHECK_ZONE) * dtau_old;
	long * d_npart;
	unsigned long long int * d_buffer_count1;
	unsigned long long int * d_buffer_count2;
	int * d_npart_row;
	int * d_npart_checkID_row;
	double (* d_vertex)[3] = NULL;
	Real * d_domain = NULL;
	vector<LightconeParticleWriteChunk> particle_write_chunks;
	long long local_particle_begin = 0;
	long long local_particle_cursor = 0;
	long long total_particles = 0;
	long long writer_begin = 0;
	long long writer_end = 0;
	long long writer_count = 0;
	int global_write_chunk_count = 0;
#if PARTICLE_LC_BALANCED_IO
	bool balanced_io_active = true;
#else
	bool balanced_io_active = false;
#endif
	const bool cuda_aware_mpi_active = gevolution_cuda_aware_mpi_active();
	cudaError_t success;

	if (hdr.num_files != 1)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": writing multiple Gadget2 files not currently supported!" << endl;
		return;
	}

	npart_row = (int *) malloc(sizeof(int) * this->num_row_buffers_);
	npart_checkID_row = (int *) malloc(sizeof(int) * this->num_row_buffers_);

	domain[0] = this->coordSkip_[1] * this->boxSize_[0] / this->lat_size_[0];
	domain[1] = domain[0] + this->lat_size_local_[1] * this->boxSize_[0] / this->lat_size_[0];
	domain[2] = this->coordSkip_[0] * this->boxSize_[0] / this->lat_size_[0];
	domain[3] = domain[2] + this->lat_size_local_[2] * this->boxSize_[0] / this->lat_size_[0];

	IDs = NULL;

	nvtxRangePushA("count particles to be written");

	npart = 0;
#pragma omp parallel for
	for (int row = 0; row < this->num_row_buffers_; row++)
	{
		npart_row[row] = 0;
		npart_checkID_row[row] = 0;
	}

	{
		size_t count_workspace_bytes = 0;
		count_workspace_bytes += DeviceWorkspace::aligned_bytes<long>(1);
		count_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(1);
		count_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(1);
		count_workspace_bytes += DeviceWorkspace::aligned_bytes<int>(this->num_row_buffers_);
		count_workspace_bytes += DeviceWorkspace::aligned_bytes<int>(this->num_row_buffers_);
		count_workspace_bytes += DeviceWorkspace::aligned_bytes<double>(MAX_INTERSECTS * 3);

		DeviceWorkspace count_workspace(count_workspace_bytes, "particle light-cone counting", "counting workspace");

		d_npart = count_workspace.slice<long>(1, "particle count scalar");
		d_buffer_count1 = count_workspace.slice<unsigned long long int>(1, "check-zone buffer counter");
		d_buffer_count2 = count_workspace.slice<unsigned long long int>(1, "particle buffer counter");
		d_npart_row = count_workspace.slice<int>(this->num_row_buffers_, "per-row particle counts");
		d_npart_checkID_row = count_workspace.slice<int>(this->num_row_buffers_, "per-row check-zone particle counts");
		d_vertex = (double (*)[3]) count_workspace.slice<double>(MAX_INTERSECTS * 3, "light-cone intersection vertices");

		cudaMemcpy(d_npart, &npart, sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(d_npart_row, npart_row, sizeof(int) * this->num_row_buffers_, cudaMemcpyHostToDevice);
		cudaMemcpy(d_npart_checkID_row, npart_checkID_row, sizeof(int) * this->num_row_buffers_, cudaMemcpyHostToDevice);
		cudaMemcpy(d_vertex, vertex, sizeof(double) * MAX_INTERSECTS * 3, cudaMemcpyHostToDevice);

		// count particles
		count_tracer_particles<part, part_info><<<this->num_row_buffers_, 128>>>(this, tracer_factor, lightcone, inner, outer, dtau_old, d_vertex, vertexcount, d_npart, d_npart_row, d_npart_checkID_row);

		success = cudaDeviceSynchronize();

		if (success != cudaSuccess)
		{
			throw std::runtime_error("CUDA error in count_tracer_particles");
		}

		cudaMemcpy(&npart, d_npart, sizeof(long), cudaMemcpyDeviceToHost);
		cudaMemcpy(npart_row, d_npart_row, sizeof(int) * this->num_row_buffers_, cudaMemcpyDeviceToHost);
		cudaMemcpy(npart_checkID_row, d_npart_checkID_row, sizeof(int) * this->num_row_buffers_, cudaMemcpyDeviceToHost);
	}

	long max_check_count = 0;
	long max_buffer_count = 0;

	for (int row = 0; row < this->num_row_buffers_; )
	{
		long chunk_count = 0;
		int chunk_rows = 0;

		do
		{
			chunk_count += npart_checkID_row[row + chunk_rows];
			chunk_rows++;
		} while (chunk_count < PCLBUFFER && row + chunk_rows < this->num_row_buffers_);

		if (chunk_count > max_check_count)
			max_check_count = chunk_count;

		row += chunk_rows;
	}

	for (int row = 0; row < this->num_row_buffers_; )
	{
		long chunk_count = 0;
		int chunk_rows = 0;

		do
		{
			chunk_count += npart_row[row + chunk_rows];
			chunk_rows++;
		} while (chunk_count < PCLBUFFER && row + chunk_rows < this->num_row_buffers_);

		if (chunk_count > max_buffer_count)
			max_buffer_count = chunk_count;

		row += chunk_rows;
	}

	size_t select_temp_bytes = 0;

	if (max_buffer_count > 0)
	{
		cub::DeviceSelect::Flagged((void *) NULL, select_temp_bytes, (unsigned long long int *) NULL, (unsigned char *) NULL, (unsigned long long int *) NULL, (unsigned long long int *) NULL, max_buffer_count);
	}

	size_t persistent_workspace_bytes = DeviceWorkspace::aligned_bytes<long>(backlog_count)
	                                  + DeviceWorkspace::aligned_bytes<double>(MAX_INTERSECTS * 3)
	                                  + DeviceWorkspace::aligned_bytes<Real>(4);
	size_t check_phase_workspace_bytes = 0;
	size_t buffer_phase_workspace_bytes = 0;

	check_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(1);
	check_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<long>(max_check_count);
	check_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(1);

	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(1);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(1);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<float>(3 * max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<float>(3 * max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<long>(max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned char>(max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned char>(max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(1);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<float>(3 * max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<float>(3 * max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<long>(max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned char>(max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(9);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(9);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<unsigned long long int>(9);
	buffer_phase_workspace_bytes += DeviceWorkspace::aligned_bytes<long>(max_buffer_count);
	buffer_phase_workspace_bytes += DeviceWorkspace::align_up(select_temp_bytes);

#ifdef DEBUG_DEVICE_WORKSPACE
	if (parallel.rank() == 0)
	{
		cout << " particle light-cone workspace request: persistent=" << persistent_workspace_bytes
		     << " bytes, check_phase=" << check_phase_workspace_bytes
		     << " bytes, buffer_phase=" << buffer_phase_workspace_bytes
		     << " bytes, max_check_count=" << max_check_count
		     << ", max_buffer_count=" << max_buffer_count
		     << ", cub_select_temp=" << select_temp_bytes
#ifdef FFT3D
		     << ", LATfield2_workspace_available=" << LATfield2::tempMemory.deviceWorkspaceBytes()
#endif
		     << endl;
	}
#endif

	DeviceWorkspace function_workspace(persistent_workspace_bytes + std::max(check_phase_workspace_bytes, buffer_phase_workspace_bytes), "particle light-cone buffers", "temporary buffer workspace");

	if (backlog_count > 0)
	{
		d_IDbacklog = function_workspace.slice<long>(backlog_count, "device ID backlog");
		cudaMemcpy(d_IDbacklog, IDbacklog.data(), sizeof(long) * backlog_count, cudaMemcpyHostToDevice);
	}
	d_vertex = (double (*)[3]) function_workspace.slice<double>(MAX_INTERSECTS * 3, "light-cone intersection vertices");
	d_domain = function_workspace.slice<Real>(4, "light-cone local domain");
	cudaMemcpy(d_vertex, vertex, sizeof(double) * MAX_INTERSECTS * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_domain, domain, sizeof(Real) * 4, cudaMemcpyHostToDevice);

	size_t persistent_workspace_mark = function_workspace.mark();

	// first loop: collect IDs to be checked against IDbacklog
	while (row_start < this->num_row_buffers_)
	{
		count = 0;
		row_count = 0;

		do
		{
			count += npart_checkID_row[row_start + row_count];
			row_count++;
		} while (count < PCLBUFFER && row_start + row_count < this->num_row_buffers_);

		if (count > 0)
		{
			function_workspace.reset(persistent_workspace_mark);

			d_buffer_count1 = function_workspace.slice<unsigned long long int>(1, "check-zone ID counter");
			d_IDs = function_workspace.slice<long>(count, "check-zone IDs");

			//buffer_count1 = 0;
			cudaMemset(d_buffer_count1, 0, sizeof(unsigned long long int));

			buffer_tracer_IDs<part, part_info><<<row_count, 128>>>(this, tracer_factor, lightcone, inner, outer, dtau_old, d_vertex, vertexcount, d_IDs, row_start, d_buffer_count1);

			success = cudaDeviceSynchronize();

			if (success != cudaSuccess)
			{
				throw std::runtime_error("CUDA error in buffer_tracer_IDs");
			}

			cudaMemcpy(&buffer_count1, d_buffer_count1, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

			// check IDs against IDbacklog
			reject = 0;

			if (buffer_count1 > 0 && backlog_count > 0)
			{
				unsigned long long int d_reject_host = 0;
				unsigned long long int * d_reject;
				int blocks = (buffer_count1 + 255) / 256;

				if (blocks > 65535)
					blocks = 65535;

				d_reject = function_workspace.slice<unsigned long long int>(1, "duplicate rejection counter");
				cudaMemcpy(d_reject, &d_reject_host, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
				count_lightcone_duplicate_ids<<<blocks, 256>>>(d_IDs, buffer_count1, d_IDbacklog, backlog_count, d_reject);
				success = cudaDeviceSynchronize();

				if (success != cudaSuccess)
				{
					throw std::runtime_error("CUDA error in count_lightcone_duplicate_ids");
				}

				cudaMemcpy(&d_reject_host, d_reject, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
				reject = (long) d_reject_host;
			}

			npart -= reject;
			d_IDs = NULL;
		}

		row_start += row_count;
	}

	// communicate number of particles to be written
	if (parallel.rank() == 0)
	{
		parallel.send<long>(npart, 1);
		parallel.receive<long>(count, parallel.size()-1);
		hdr.npart[1] = (uint32_t) (count % (1ll << 32));
		hdr.npartTotal[1] = (uint32_t) (count % (1ll << 32));
		hdr.npartTotalHW[1] = (uint32_t) (count / (1ll << 32));
		count = 0;
	}
	else
	{
		parallel.receive<long>(count, parallel.rank()-1);
		count += npart;
		parallel.send<long>(count, (parallel.rank()+1)%parallel.size());
		count -= npart;
	}

	parallel.broadcast<uint32_t>(hdr.npartTotal[1], 0);
	parallel.broadcast<uint32_t>(hdr.npartTotalHW[1], 0);
	local_particle_begin = count;
	local_particle_cursor = local_particle_begin;
	total_particles = (long long) hdr.npartTotal[1] + ((long long) hdr.npartTotalHW[1] << 32);

	nvtxRangePop();

	if (total_particles > 0)
	{
		MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &outfile);
	
		offset_pos = (MPI_Offset) total_particles;
		offset_pos *= (MPI_Offset) (6 * sizeof(float) + sizeof(int64_t));
		offset_pos += (MPI_Offset) (8 * sizeof(uint32_t) + sizeof(hdr));
		MPI_File_set_size(outfile, offset_pos);
	
		offset_pos_base = (MPI_Offset) (3 * sizeof(uint32_t) + sizeof(hdr));
		offset_vel_base = offset_pos_base + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) total_particles) * ((MPI_Offset) (3 * sizeof(float)));
		offset_ID_base = offset_vel_base + (MPI_Offset) (2 * sizeof(uint32_t)) + ((MPI_Offset) total_particles) * ((MPI_Offset) (3 * sizeof(float)));
		offset_pos = offset_pos_base + ((MPI_Offset) local_particle_begin) * ((MPI_Offset) (3 * sizeof(float)));
		offset_vel = offset_vel_base + ((MPI_Offset) local_particle_begin) * ((MPI_Offset) (3 * sizeof(float)));
		offset_ID = offset_ID_base + ((MPI_Offset) local_particle_begin) * ((MPI_Offset) sizeof(int64_t));
	
		if (parallel.rank() == 0)
		{
			blocksize = sizeof(hdr);		
			MPI_File_write_at(outfile, 0, &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, sizeof(uint32_t), &hdr, sizeof(hdr), MPI_BYTE, &status);
			MPI_File_write_at(outfile, sizeof(hdr) + sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = 3 * sizeof(float) * hdr.npart[1];
			MPI_File_write_at(outfile, sizeof(hdr) + 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_vel - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID - 2*sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			blocksize = ((GADGET_ID_BYTES == 8) ? sizeof(int64_t) : sizeof(int32_t)) * hdr.npart[1];
			MPI_File_write_at(outfile, offset_ID - sizeof(uint32_t), &blocksize, 1, MPI_UNSIGNED, &status);
			MPI_File_write_at(outfile, offset_ID + blocksize, &blocksize, 1, MPI_UNSIGNED, &status);
		}

		for (int row = 0; row < this->num_row_buffers_; )
		{
			LightconeParticleWriteChunk chunk;
			chunk.row_start = row;
			chunk.row_count = 0;
			chunk.count = 0;
			chunk.check_count = 0;

			do
			{
				chunk.count += npart_row[row + chunk.row_count];
				chunk.check_count += npart_checkID_row[row + chunk.row_count];
				chunk.row_count++;
			} while (chunk.count < PCLBUFFER && row + chunk.row_count < this->num_row_buffers_);

			particle_write_chunks.push_back(chunk);
			row += chunk.row_count;
		}

		if (balanced_io_active)
		{
			int local_write_chunk_count = (int) particle_write_chunks.size();
			MPI_Allreduce(&local_write_chunk_count, &global_write_chunk_count, 1, MPI_INT, MPI_MAX, parallel.lat_world_comm());

			writer_begin = lightcone_balanced_partition_begin(total_particles, parallel.rank(), parallel.size());
			writer_end = lightcone_balanced_partition_begin(total_particles, parallel.rank()+1, parallel.size());
			writer_count = writer_end - writer_begin;

			if (writer_count > 0)
			{
				posdata = (float *) malloc((size_t) writer_count * 3 * sizeof(float));
				veldata = (float *) malloc((size_t) writer_count * 3 * sizeof(float));
				IDs = (long *) malloc((size_t) writer_count * sizeof(long));
			}
			else
			{
				posdata = NULL;
				veldata = NULL;
				IDs = NULL;
			}

			int local_allocation_failed = (writer_count > 0 && (posdata == NULL || veldata == NULL || IDs == NULL)) ? 1 : 0;
			int global_allocation_failed = 0;
			MPI_Allreduce(&local_allocation_failed, &global_allocation_failed, 1, MPI_INT, MPI_MAX, parallel.lat_world_comm());

			if (global_allocation_failed)
			{
				if (local_allocation_failed)
				{
					cout << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " unable to allocate balanced particle light-cone writer buffers for " << writer_count << " particles; falling back to direct MPI-IO." << endl;
				}
				if (posdata != NULL) free(posdata);
				if (veldata != NULL) free(veldata);
				if (IDs != NULL) free(IDs);
				posdata = NULL;
				veldata = NULL;
				IDs = NULL;
				balanced_io_active = false;
			}
		}

		if (!balanced_io_active)
		{
			posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
			veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
			IDs = (long *) malloc(sizeof(long) * PCLBUFFER);
		}

		if (posdata == NULL || veldata == NULL || IDs == NULL)
		{
			if (!balanced_io_active || writer_count > 0)
				throw std::runtime_error("Error allocating memory for particle buffers");
		}

		// second loop: buffer and write particles
		int write_chunk_count = balanced_io_active ? global_write_chunk_count : (int) particle_write_chunks.size();

		for (int write_chunk = 0; write_chunk < write_chunk_count; write_chunk++)
		{
			nvtxRangePushA("buffer particles");
			if (write_chunk < (int) particle_write_chunks.size())
			{
				row_start = particle_write_chunks[write_chunk].row_start;
				row_count = particle_write_chunks[write_chunk].row_count;
				count = particle_write_chunks[write_chunk].count;
				buffer_count2 = particle_write_chunks[write_chunk].check_count;
			}
			else
			{
				row_start = this->num_row_buffers_;
				row_count = 0;
				count = 0;
				buffer_count2 = 0;
			}

			if (!balanced_io_active && count > PCLBUFFER)
			{
				float * new_posdata = (float *) realloc(posdata, 3 * sizeof(float) * count);
				float * new_veldata = (float *) realloc(veldata, 3 * sizeof(float) * count);
				long * new_IDs = (long *) realloc(IDs, sizeof(long) * count);

				if (new_posdata == NULL || new_veldata == NULL || new_IDs == NULL)
				{
					throw std::runtime_error("Error reallocating memory for particle buffers");
				}

				posdata = new_posdata;
				veldata = new_veldata;
				IDs = new_IDs;
			}

			if (count > 0)
			{
				function_workspace.reset(persistent_workspace_mark);

				d_buffer_count1 = function_workspace.slice<unsigned long long int>(1, "particle overlap counter");
				d_buffer_count2 = function_workspace.slice<unsigned long long int>(1, "particle output counter");
				d_posdata = function_workspace.slice<float>(3 * count, "particle position write buffer");
				d_veldata = function_workspace.slice<float>(3 * count, "particle velocity write buffer");
				d_IDs = function_workspace.slice<long>(count, "particle ID write buffer");
				d_loginfo = function_workspace.slice<unsigned char>(count, "particle ID-log routing tags");
				d_keep_flags = function_workspace.slice<unsigned char>(count, "particle duplicate keep flags");
				d_indices = function_workspace.slice<unsigned long long int>(count, "particle compaction source indices");
				d_selected = function_workspace.slice<unsigned long long int>(count, "particle compaction selected indices");
				d_selected_count = function_workspace.slice<unsigned long long int>(1, "particle compaction selected count");
				d_posdata_compact = function_workspace.slice<float>(3 * count, "compacted particle position buffer");
				d_veldata_compact = function_workspace.slice<float>(3 * count, "compacted particle velocity buffer");
				d_IDs_compact = function_workspace.slice<long>(count, "compacted particle ID buffer");
				d_loginfo_compact = function_workspace.slice<unsigned char>(count, "compacted ID-log routing tags");
				d_prelog_counts = function_workspace.slice<unsigned long long int>(9, "ID-prelog bin counts");
				d_prelog_offsets = function_workspace.slice<unsigned long long int>(9, "ID-prelog bin offsets");
				d_prelog_cursors = function_workspace.slice<unsigned long long int>(9, "ID-prelog bin cursors");
				d_prelog_ids = function_workspace.slice<long>(count, "ID-prelog IDs");
				void * select_temp = function_workspace.slice_bytes(select_temp_bytes, 256, "CUB duplicate compaction temporary storage");

				//buffer_count1 = 0;
				cudaMemset(d_buffer_count1, 0, sizeof(unsigned long long int));
				cudaMemcpy(d_buffer_count2, &buffer_count2, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

				buffer_tracer_particles<part, part_info, IDlog_scatter><<<row_count, 128>>>(this, tracer_factor, lightcone, (Real) dist, inner, outer, dtau, dtau_old, (double) hdr.time, dadtau, hdr.BoxSize, d_domain, phi, d_vertex, vertexcount, d_posdata, d_veldata, d_IDs, d_loginfo, row_start, d_buffer_count1, d_buffer_count2);

				success = cudaDeviceSynchronize();

				if (success != cudaSuccess)
				{
					throw std::runtime_error("CUDA error in buffer_tracer_particles");
				}

				cudaMemcpy(&buffer_count1, d_buffer_count1, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

				if (buffer_count1 > 0 && backlog_count > 0)
				{
					int blocks = (count + 255) / 256;

					if (blocks > 65535)
						blocks = 65535;

					mark_lightcone_kept_particles<<<blocks, 256>>>(d_IDs, count, buffer_count1, d_IDbacklog, backlog_count, d_keep_flags);
					success = cudaDeviceSynchronize();

					if (success != cudaSuccess)
					{
						throw std::runtime_error("CUDA error in mark_lightcone_kept_particles");
					}

					thrust::sequence(thrust::device, d_indices, d_indices + count);

					cub::DeviceSelect::Flagged(select_temp, select_temp_bytes, d_indices, d_keep_flags, d_selected, d_selected_count, count);
					cudaDeviceSynchronize();

					unsigned long long int kept_count = 0;
					cudaMemcpy(&kept_count, d_selected_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

					if (kept_count < (unsigned long long int) count)
					{
						if (kept_count > 0)
						{
							blocks = (kept_count + 255) / 256;

							if (blocks > 65535)
								blocks = 65535;

							gather_lightcone_kept_particles<<<blocks, 256>>>(d_posdata, d_veldata, d_IDs, d_loginfo, d_posdata_compact, d_veldata_compact, d_IDs_compact, d_loginfo_compact, d_selected, kept_count);
							success = cudaDeviceSynchronize();

							if (success != cudaSuccess)
							{
								throw std::runtime_error("CUDA error in gather_lightcone_kept_particles");
							}
						}

						d_posdata = d_posdata_compact;
						d_veldata = d_veldata_compact;
						d_IDs = d_IDs_compact;
						d_loginfo = d_loginfo_compact;
						count = (long) kept_count;
					}
				}
			}
			nvtxRangePop();

			nvtxRangePushA("write particles to disk");
			if (count > 0)
			{
				// fill the IDprelogs
				int log_bins = IDlog_scatter ? 9 : 1;
				unsigned long long int bin_counts[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
				unsigned long long int bin_offsets[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
				unsigned long long int total_prelog = 0;
				int blocks = (count + 255) / 256;

				if (blocks > 65535)
					blocks = 65535;

				cudaMemset(d_prelog_counts, 0, sizeof(unsigned long long int) * 9);
				count_lightcone_prelog_bins<<<blocks, 256>>>(d_loginfo, count, log_bins, d_prelog_counts);
				success = cudaDeviceSynchronize();

				if (success != cudaSuccess)
				{
					throw std::runtime_error("CUDA error in count_lightcone_prelog_bins");
				}

				cudaMemcpy(bin_counts, d_prelog_counts, sizeof(unsigned long long int) * 9, cudaMemcpyDeviceToHost);

				for (int bin = 0; bin < log_bins; bin++)
				{
					bin_offsets[bin] = total_prelog;
					total_prelog += bin_counts[bin];
				}

				if (total_prelog > 0)
				{
					cudaMemcpy(d_prelog_offsets, bin_offsets, sizeof(unsigned long long int) * 9, cudaMemcpyHostToDevice);
					cudaMemset(d_prelog_cursors, 0, sizeof(unsigned long long int) * 9);
					fill_lightcone_prelog_bins<<<blocks, 256>>>(d_IDs, d_loginfo, count, log_bins, d_prelog_offsets, d_prelog_cursors, d_prelog_ids);
					success = cudaDeviceSynchronize();

					if (success != cudaSuccess)
					{
						throw std::runtime_error("CUDA error in fill_lightcone_prelog_bins");
					}

					for (int bin = 0; bin < log_bins; bin++)
					{
						if (bin_counts[bin] > 0)
						{
							size_t old_size = IDprelog[bin].size();
							IDprelog[bin].resize(old_size + bin_counts[bin]);
							cudaMemcpy(IDprelog[bin].data() + old_size, d_prelog_ids + bin_offsets[bin], sizeof(long) * bin_counts[bin], cudaMemcpyDeviceToHost);
						}
					}
				}

				if (!balanced_io_active)
				{
					cudaMemcpy(posdata, d_posdata, 3 * sizeof(float) * count, cudaMemcpyDeviceToHost);
					cudaMemcpy(veldata, d_veldata, 3 * sizeof(float) * count, cudaMemcpyDeviceToHost);
					cudaMemcpy(IDs, d_IDs, sizeof(long) * count, cudaMemcpyDeviceToHost);

					MPI_File_write_at(outfile, offset_pos, posdata, 3 * count, MPI_FLOAT, &status);
					offset_pos += 3 * count * sizeof(float);
					MPI_File_write_at(outfile, offset_vel, veldata, 3 * count, MPI_FLOAT, &status);
					offset_vel += 3 * count * sizeof(float);
					long ID_bytes = count * sizeof(int64_t);
					MPI_File_write_at(outfile, offset_ID, IDs, ID_bytes, MPI_BYTE, &status);
					offset_ID += ID_bytes;
				}
			}

			if (balanced_io_active)
			{
				const int tag_pos = 12011;
				const int tag_vel = 12012;
				const int tag_id = 12013;
				vector<long long> send_counts(parallel.size(), 0);
				vector<long long> recv_counts(parallel.size(), 0);
				vector<long long> send_starts(parallel.size(), -1);
				vector<long long> recv_starts(parallel.size(), -1);
				vector<MPI_Request> requests;
				vector<float> host_pos_send;
				vector<float> host_vel_send;
				vector<long> host_ID_send;
				const float * mpi_pos_send = d_posdata;
				const float * mpi_vel_send = d_veldata;
				const long * mpi_ID_send = d_IDs;
				long long chunk_begin = local_particle_cursor;
				long long chunk_end = chunk_begin + count;

				if (!cuda_aware_mpi_active && count > 0)
				{
					host_pos_send.resize((size_t) count * 3);
					host_vel_send.resize((size_t) count * 3);
					host_ID_send.resize((size_t) count);
					cudaMemcpy(host_pos_send.data(), d_posdata, (size_t) count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(host_vel_send.data(), d_veldata, (size_t) count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(host_ID_send.data(), d_IDs, (size_t) count * sizeof(long), cudaMemcpyDeviceToHost);
					mpi_pos_send = host_pos_send.data();
					mpi_vel_send = host_vel_send.data();
					mpi_ID_send = host_ID_send.data();
				}

				if (count > 0)
				{
					for (int dest = 0; dest < parallel.size(); dest++)
					{
						long long dest_begin = lightcone_balanced_partition_begin(total_particles, dest, parallel.size());
						long long dest_end = lightcone_balanced_partition_begin(total_particles, dest+1, parallel.size());
						long long segment_begin = std::max(chunk_begin, dest_begin);
						long long segment_end = std::min(chunk_end, dest_end);

						if (segment_begin < segment_end)
						{
							send_starts[dest] = segment_begin;
							send_counts[dest] = segment_end - segment_begin;
						}
					}
				}

				MPI_Alltoall(send_counts.data(), 1, MPI_LONG_LONG, recv_counts.data(), 1, MPI_LONG_LONG, parallel.lat_world_comm());
				MPI_Alltoall(send_starts.data(), 1, MPI_LONG_LONG, recv_starts.data(), 1, MPI_LONG_LONG, parallel.lat_world_comm());

				for (int source = 0; source < parallel.size(); source++)
				{
					if (source == parallel.rank() || recv_counts[source] <= 0)
						continue;

					long long recv_offset = recv_starts[source] - writer_begin;
					lightcone_mpi_irecv_bytes(posdata + 3 * recv_offset, (size_t) recv_counts[source] * 3 * sizeof(float), source, tag_pos, parallel.lat_world_comm(), requests);
					lightcone_mpi_irecv_bytes(veldata + 3 * recv_offset, (size_t) recv_counts[source] * 3 * sizeof(float), source, tag_vel, parallel.lat_world_comm(), requests);
					lightcone_mpi_irecv_bytes(IDs + recv_offset, (size_t) recv_counts[source] * sizeof(long), source, tag_id, parallel.lat_world_comm(), requests);
				}

				if (send_counts[parallel.rank()] > 0)
				{
					long long source_offset = send_starts[parallel.rank()] - chunk_begin;
					long long writer_offset = send_starts[parallel.rank()] - writer_begin;

					cudaMemcpy(posdata + 3 * writer_offset, d_posdata + 3 * source_offset, (size_t) send_counts[parallel.rank()] * 3 * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(veldata + 3 * writer_offset, d_veldata + 3 * source_offset, (size_t) send_counts[parallel.rank()] * 3 * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(IDs + writer_offset, d_IDs + source_offset, (size_t) send_counts[parallel.rank()] * sizeof(long), cudaMemcpyDeviceToHost);
				}

				for (int dest = 0; dest < parallel.size(); dest++)
				{
					if (dest == parallel.rank() || send_counts[dest] <= 0)
						continue;

					long long source_offset = send_starts[dest] - chunk_begin;
					lightcone_mpi_isend_bytes(mpi_pos_send + 3 * source_offset, (size_t) send_counts[dest] * 3 * sizeof(float), dest, tag_pos, parallel.lat_world_comm(), requests);
					lightcone_mpi_isend_bytes(mpi_vel_send + 3 * source_offset, (size_t) send_counts[dest] * 3 * sizeof(float), dest, tag_vel, parallel.lat_world_comm(), requests);
					lightcone_mpi_isend_bytes(mpi_ID_send + source_offset, (size_t) send_counts[dest] * sizeof(long), dest, tag_id, parallel.lat_world_comm(), requests);
				}

				if (!requests.empty())
					MPI_Waitall((int) requests.size(), requests.data(), MPI_STATUSES_IGNORE);
			}
			nvtxRangePop();

			d_posdata = NULL;
			d_veldata = NULL;
			d_IDs = NULL;
			d_loginfo = NULL;
			d_posdata_compact = NULL;
			d_veldata_compact = NULL;
			d_IDs_compact = NULL;
			d_loginfo_compact = NULL;
			d_keep_flags = NULL;
			d_indices = NULL;
			d_selected = NULL;
			d_selected_count = NULL;
			d_prelog_counts = NULL;
			d_prelog_offsets = NULL;
			d_prelog_cursors = NULL;
			d_prelog_ids = NULL;

			local_particle_cursor += count;
		}

		if (balanced_io_active)
		{
			nvtxRangePushA("write balanced particles to disk");
			MPI_Barrier(parallel.lat_world_comm());
			lightcone_mpi_file_write_at_all_bytes(outfile, offset_pos_base + ((MPI_Offset) writer_begin) * ((MPI_Offset) (3 * sizeof(float))), posdata, (unsigned long long int) writer_count * 3 * sizeof(float), parallel.lat_world_comm(), &status);
			lightcone_mpi_file_write_at_all_bytes(outfile, offset_vel_base + ((MPI_Offset) writer_begin) * ((MPI_Offset) (3 * sizeof(float))), veldata, (unsigned long long int) writer_count * 3 * sizeof(float), parallel.lat_world_comm(), &status);
			lightcone_mpi_file_write_at_all_bytes(outfile, offset_ID_base + ((MPI_Offset) writer_begin) * ((MPI_Offset) sizeof(int64_t)), IDs, (unsigned long long int) writer_count * sizeof(int64_t), parallel.lat_world_comm(), &status);
			nvtxRangePop();
		}

		MPI_File_close(&outfile);

		free(posdata);
		free(veldata);
	}

	free(IDs);
	free(npart_row);
	free(npart_checkID_row);
}


template <typename part, typename part_info, typename part_dataType>
void Particles_gevolution<part,part_info,part_dataType>::loadGadget2(string filename, gadget2_header & hdr)
{
	float * posdata;
	float * veldata;
	void * IDs;
	part pcl;
	MPI_File infile;
	uint32_t count, npart = 0;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Status status;
	uint32_t blocksize;
	double rescale_vel = 1. / GADGET_VELOCITY_CONVERSION;
	
	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);

#if GADGET_ID_BYTES == 8
	IDs = malloc(sizeof(int64_t) * PCLBUFFER);
#else
	IDs = malloc(sizeof(int32_t) * PCLBUFFER);
#endif

	MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);

	MPI_File_read_all(infile, &blocksize, 1, MPI_UNSIGNED, &status);

	if (blocksize != sizeof(hdr))
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": file type not recognized when reading Gadget2 file!" << endl;
		MPI_File_close(&infile);
		return;
	}

	MPI_File_read_all(infile, &hdr, sizeof(hdr), MPI_BYTE, &status);

	rescale_vel /= sqrt(hdr.time);
	offset_pos = (MPI_Offset) sizeof(hdr) + (MPI_Offset) (3 * sizeof(uint32_t));
	offset_vel = offset_pos + ((MPI_Offset) hdr.npart[1]) * ((MPI_Offset) (3 * sizeof(float))) + (MPI_Offset) (2 * sizeof(uint32_t));
	offset_ID = offset_vel + offset_vel - offset_pos;

	MPI_File_seek(infile, offset_pos, MPI_SEEK_SET);
	while (npart < hdr.npart[1])
	{
		count = (hdr.npart[1] - npart > PCLBUFFER) ? PCLBUFFER : (hdr.npart[1] - npart);

		MPI_File_read_all(infile, posdata, 3 * count, MPI_FLOAT, &status);
		offset_pos += (MPI_Offset) (3 * count * sizeof(float));
		MPI_File_seek(infile, offset_vel, MPI_SEEK_SET);
		MPI_File_read_all(infile, veldata, 3 * count, MPI_FLOAT, &status);
		offset_vel += (MPI_Offset) (3 * count * sizeof(float));
		MPI_File_seek(infile, offset_ID, MPI_SEEK_SET);
#if GADGET_ID_BYTES == 8
		MPI_File_read_all(infile, IDs, count * sizeof(int64_t), MPI_BYTE, &status);
		offset_ID += (MPI_Offset) (count * sizeof(int64_t));
#else
		MPI_File_read_all(infile, IDs, count * sizeof(int32_t), MPI_BYTE, &status);
		offset_ID += (MPI_Offset) (count * sizeof(int32_t));
#endif
		MPI_File_seek(infile, offset_pos, MPI_SEEK_SET);

		for (int i = 0; i < 3 * count; i++)
		{
			posdata[i] /= hdr.BoxSize;
			while (posdata[i] < 0) posdata[i] += 1;
			while (posdata[i] >= 1.) posdata[i] -= 1.;
			veldata[i] *= hdr.time / rescale_vel;
		}

		for (int i = 0; i < count; i++)
		{
#if GADGET_ID_BYTES == 8
			pcl.ID = *((int64_t *) IDs + i);
#else
			pcl.ID = *((int32_t *) IDs + i);
#endif
			pcl.pos[0] = posdata[3*i];
			pcl.pos[1] = posdata[3*i+1];
			pcl.pos[2] = posdata[3*i+2];
			pcl.vel[0] = veldata[3*i];
			pcl.vel[1] = veldata[3*i+1];
			pcl.vel[2] = veldata[3*i+2];
			this->addParticle_global(pcl);
		}
		
		npart += count;
	}

	MPI_File_close(&infile);
	
	free(posdata);
	free(veldata);
	free(IDs);
}


// CUDA kernel to add particles
template <typename part, typename part_info>
__global__ void add_particles(perfParticles_gevolution<part, part_info> * pcl, float * posdata, float * veldata, void * IDs, uint32_t count, unsigned long long int * buffer_idx)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
	{
		int coord[3];

#ifdef SINGLE
		pcl->getPartCoordLocal(posdata+3*i, coord);
#else
		double pos[3];
		pos[0] = posdata[3*i];
		pos[1] = posdata[3*i+1];
		pos[2] = posdata[3*i+2];
		pcl->getPartCoordLocal(pos, coord);
#endif

		if (coord[0] >= 0 && coord[0] < pcl->lat_size_local_[0] && coord[1] >= 0 && coord[1] < pcl->lat_size_local_[1] && coord[2] >= 0 && coord[2] < pcl->lat_size_local_[2])
		{
			unsigned long long int idx = atomicAdd(buffer_idx, 1);

			pcl->p[3*idx] = posdata[3*i];
			pcl->p[3*idx+1] = posdata[3*i+1];
			pcl->p[3*idx+2] = posdata[3*i+2];
			pcl->q[3*idx] = veldata[3*i];
			pcl->q[3*idx+1] = veldata[3*i+1];
			pcl->q[3*idx+2] = veldata[3*i+2];
#if GADGET_ID_BYTES == 8
			pcl->other[idx] = *((int64_t *) IDs + i);
#else
			pcl->other[idx] = *((int32_t *) IDs + i);
#endif
		}
	}
}

template <typename part, typename part_info>
void perfParticles_gevolution<part,part_info>::loadGadget2(string filename, gadget2_header & hdr)
{
	float * posdata;
	float * veldata;
	void * IDs;
	MPI_File infile;
	uint32_t count, npart = 0;
	MPI_Offset offset_pos, offset_vel, offset_ID;
	MPI_Status status;
	uint32_t blocksize;
	double rescale_vel = 1. / GADGET_VELOCITY_CONVERSION;
	uint64_t estimated_capacity = 0;
	
	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);

#if GADGET_ID_BYTES == 8
	IDs = malloc(sizeof(int64_t) * PCLBUFFER);
#else
	IDs = malloc(sizeof(int32_t) * PCLBUFFER);
#endif

	size_t reader_workspace_bytes = particle_io_staging_workspace_bytes(PCLBUFFER);
	DeviceWorkspace reader_workspace(reader_workspace_bytes, "Gadget-2 particle input", "particle staging buffers");
	float * d_posdata = reader_workspace.slice<float>(3 * PCLBUFFER, "position staging buffer");
	float * d_veldata = reader_workspace.slice<float>(3 * PCLBUFFER, "velocity staging buffer");
	void * d_IDs = reader_workspace.slice_bytes(sizeof(int64_t) * PCLBUFFER, 256, "ID staging buffer");
	unsigned long long int * d_buffer_idx = reader_workspace.slice<unsigned long long int>(1, "particle insertion counter");

	MPI_File_open(parallel.lat_world_comm(), filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);

	// check if file could be opened
	if (infile == MPI_FILE_NULL)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not open file " << filename << "!" << endl;
		return;
	}

	auto read_success = MPI_File_read_all(infile, &blocksize, 1, MPI_UNSIGNED, &status);

	if (blocksize != sizeof(hdr) || read_success != MPI_SUCCESS)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": file type not recognized when reading Gadget2 file!" << endl;
		MPI_File_close(&infile);
		return;
	}

	read_success = MPI_File_read_all(infile, &hdr, sizeof(hdr), MPI_BYTE, &status);

	if (read_success != MPI_SUCCESS)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read header from file " << filename << "!" << endl;
		MPI_File_close(&infile);
		return;
	}

	estimated_capacity = (((uint64_t) hdr.npartTotal[1] + ((uint64_t) hdr.npartTotalHW[1] << 32)) / parallel.size()) + this->extra_capacity_;

	rescale_vel /= sqrt(hdr.time);
	offset_pos = (MPI_Offset) sizeof(hdr) + (MPI_Offset) (3 * sizeof(uint32_t));
	offset_vel = offset_pos + ((MPI_Offset) hdr.npart[1]) * ((MPI_Offset) (3 * sizeof(float))) + (MPI_Offset) (2 * sizeof(uint32_t));
	offset_ID = offset_vel + offset_vel - offset_pos;

	MPI_File_seek(infile, offset_pos, MPI_SEEK_SET);

	while (npart < hdr.npart[1])
	{
		count = (hdr.npart[1] - npart > PCLBUFFER) ? PCLBUFFER : (hdr.npart[1] - npart);

		read_success = MPI_File_read_all(infile, posdata, 3 * count, MPI_FLOAT, &status);

		if (read_success != MPI_SUCCESS)
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read particle positions from file " << filename << "!" << endl;
			MPI_File_close(&infile);
			return;
		}

		offset_pos += (MPI_Offset) (3 * count * sizeof(float));
		MPI_File_seek(infile, offset_vel, MPI_SEEK_SET);
		read_success = MPI_File_read_all(infile, veldata, 3 * count, MPI_FLOAT, &status);

		if (read_success != MPI_SUCCESS)
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read particle velocities from file " << filename << "!" << endl;
			MPI_File_close(&infile);
			return;
		}

		offset_vel += (MPI_Offset) (3 * count * sizeof(float));
		MPI_File_seek(infile, offset_ID, MPI_SEEK_SET);
#if GADGET_ID_BYTES == 8
		read_success = MPI_File_read_all(infile, IDs, count * sizeof(int64_t), MPI_BYTE, &status);
		offset_ID += (MPI_Offset) (count * sizeof(int64_t));
#else
		read_success = MPI_File_read_all(infile, IDs, count * sizeof(int32_t), MPI_BYTE, &status);
		offset_ID += (MPI_Offset) (count * sizeof(int32_t));
#endif
		MPI_File_seek(infile, offset_pos, MPI_SEEK_SET);

		if (read_success != MPI_SUCCESS)
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read particle IDs from file " << filename << "!" << endl;
			MPI_File_close(&infile);
			return;
		}

#pragma omp parallel for
		for (int i = 0; i < 3 * count; i++)
		{
			posdata[i] /= hdr.BoxSize;
			while (posdata[i] < 0) posdata[i] += 1;
			while (posdata[i] >= 1) posdata[i] -= 1;
			veldata[i] *= hdr.time / rescale_vel;
		}

		int local_count = 0;

#pragma omp parallel for reduction(+:local_count)
		for (int i = 0; i < count; i++)
		{
			int coord[3];
#ifdef SINGLE
			this->getPartCoordLocal(posdata+3*i, coord);
#else
			double pos[3];
			pos[0] = posdata[3*i];
			pos[1] = posdata[3*i+1];
			pos[2] = posdata[3*i+2];
			this->getPartCoordLocal(pos, coord);
#endif
			if (coord[0] >= 0 && coord[0] < this->lat_size_local_[0] && coord[1] >= 0 && coord[1] < this->lat_size_local_[1] && coord[2] >= 0 && coord[2] < this->lat_size_local_[2])
			{
				local_count++;
			}
		}

		if (this->num_particles_ + local_count > this->total_capacity_)
		{
			if (this->num_particles_ + local_count + this->extra_capacity_ > estimated_capacity)
			{
				estimated_capacity = this->num_particles_ + local_count + this->extra_capacity_;
			}
			this->resizeGlobalBuffers(estimated_capacity);
			//cerr << " rank#" << parallel.rank() << ": resized particle buffers to capacity " << this->total_capacity_ << endl;
		}

		if (local_count > 0)
		{
			unsigned long long int buffer_idx = this->num_particles_;
			cudaMemcpy(d_buffer_idx, &buffer_idx, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_posdata, posdata, 3 * count * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_veldata, veldata, 3 * count * sizeof(float), cudaMemcpyHostToDevice);
#if GADGET_ID_BYTES == 8
			cudaMemcpy(d_IDs, IDs, count * sizeof(int64_t), cudaMemcpyHostToDevice);
#else
			cudaMemcpy(d_IDs, IDs, count * sizeof(int32_t), cudaMemcpyHostToDevice);
#endif
			
			add_particles<part, part_info><<<count/128+1, 128>>>(this, d_posdata, d_veldata, d_IDs, count, d_buffer_idx);

			auto success = cudaDeviceSynchronize();

			if (success != cudaSuccess)
			{
				COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": CUDA error when loading particles!" << endl;
				MPI_File_close(&infile);
				throw std::runtime_error("CUDA error when loading particles!");
			}

			this->num_particles_ += local_count;
		}
		
		npart += count;
	}

	MPI_File_close(&infile);

	//cerr << " rank#" << parallel.rank() << ": loaded a total of " << this->num_particles_ << " particles." << endl;
	this->updateRowBuffers();
	
	free(posdata);
	free(veldata);
	free(IDs);
}

template <typename part, typename part_info>
void perfParticles_gevolution<part,part_info>::saveExpress(string filename, gadget2_header & hdr)
{
	if (hdr.num_files != parallel.size())
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": number of express files must match the number of tasks!" << endl;
		throw std::runtime_error("Invalid express file count");
	}

	this->updateRowBuffers();

	long local_npart = (long) this->num_particles_;
	long global_npart = local_npart;
	parallel.sum(global_npart);

	hdr.num_files = parallel.size();
	hdr.npart[1] = (uint32_t) (local_npart & 0xffffffffu);
	hdr.npartTotal[1] = (uint32_t) (global_npart & 0xffffffffu);
	hdr.npartTotalHW[1] = (uint32_t) (((uint64_t) global_npart) >> 32);

	express_header ehdr;
	memset(&ehdr, 0, sizeof(ehdr));
	memcpy(ehdr.magic, EXPRESS_MAGIC, sizeof(ehdr.magic));
	ehdr.version = EXPRESS_VERSION;
	ehdr.header_size = sizeof(ehdr);
	ehdr.endian = EXPRESS_ENDIAN;
	ehdr.sizeof_real = sizeof(Real);
	ehdr.sizeof_long = sizeof(long);
	ehdr.sizeof_gadget2_header = sizeof(gadget2_header);
	ehdr.rank = parallel.rank();
	ehdr.size = parallel.size();
	ehdr.grid_size[0] = parallel.grid_size()[0];
	ehdr.grid_size[1] = parallel.grid_size()[1];
	for (int i = 0; i < 3; i++)
	{
		ehdr.lat_size[i] = this->lat_size_[i];
		ehdr.lat_size_local[i] = this->lat_size_local_[i];
	}
	ehdr.coord_skip[0] = this->coordSkip_[0];
	ehdr.coord_skip[1] = this->coordSkip_[1];
	ehdr.local_npart = (uint64_t) local_npart;
	ehdr.global_npart = (uint64_t) global_npart;
	ehdr.position_bytes = 3 * ehdr.local_npart * sizeof(Real);
	ehdr.momentum_bytes = 3 * ehdr.local_npart * sizeof(Real);
	ehdr.id_bytes = ehdr.local_npart * sizeof(long);

	string rank_filename = express_rank_filename(filename, parallel.rank());
	FILE * outfile = fopen(rank_filename.c_str(), "wb");
	if (outfile == NULL)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not open express file " << rank_filename << " for writing!" << endl;
		throw std::runtime_error("Could not open express output file");
	}

	if (fwrite(&ehdr, sizeof(ehdr), 1, outfile) != 1)
	{
		fclose(outfile);
		throw std::runtime_error("Could not write express header");
	}

	Real * real_buffer = (Real *) malloc(3 * sizeof(Real) * PCLBUFFER);
	long * id_buffer = (long *) malloc(sizeof(long) * PCLBUFFER);
	if (real_buffer == NULL || id_buffer == NULL)
	{
		fclose(outfile);
		free(real_buffer);
		free(id_buffer);
		throw std::runtime_error("Could not allocate express staging buffers");
	}

	uint64_t written = 0;
	while (written < ehdr.local_npart)
	{
		uint64_t count = (ehdr.local_npart - written > PCLBUFFER) ? PCLBUFFER : (ehdr.local_npart - written);
		cudaMemcpy(real_buffer, this->p + 3 * written, 3 * count * sizeof(Real), cudaMemcpyDeviceToHost);
		if (fwrite(real_buffer, sizeof(Real), 3 * count, outfile) != 3 * count)
		{
			fclose(outfile);
			free(real_buffer);
			free(id_buffer);
			throw std::runtime_error("Could not write express positions");
		}
		written += count;
	}

	written = 0;
	while (written < ehdr.local_npart)
	{
		uint64_t count = (ehdr.local_npart - written > PCLBUFFER) ? PCLBUFFER : (ehdr.local_npart - written);
		cudaMemcpy(real_buffer, this->q + 3 * written, 3 * count * sizeof(Real), cudaMemcpyDeviceToHost);
		if (fwrite(real_buffer, sizeof(Real), 3 * count, outfile) != 3 * count)
		{
			fclose(outfile);
			free(real_buffer);
			free(id_buffer);
			throw std::runtime_error("Could not write express momenta");
		}
		written += count;
	}

	written = 0;
	while (written < ehdr.local_npart)
	{
		uint64_t count = (ehdr.local_npart - written > PCLBUFFER) ? PCLBUFFER : (ehdr.local_npart - written);
		cudaMemcpy(id_buffer, this->other + written, count * sizeof(long), cudaMemcpyDeviceToHost);
		if (fwrite(id_buffer, sizeof(long), count, outfile) != count)
		{
			fclose(outfile);
			free(real_buffer);
			free(id_buffer);
			throw std::runtime_error("Could not write express IDs");
		}
		written += count;
	}

	if (fwrite(&hdr, sizeof(hdr), 1, outfile) != 1)
	{
		fclose(outfile);
		free(real_buffer);
		free(id_buffer);
		throw std::runtime_error("Could not write express Gadget2 metadata");
	}

	fclose(outfile);
	free(real_buffer);
	free(id_buffer);
}

template <typename part, typename part_info>
void perfParticles_gevolution<part,part_info>::loadExpress(string filename, gadget2_header & hdr)
{
	string rank_filename = express_rank_filename(filename, parallel.rank());
	FILE * infile = fopen(rank_filename.c_str(), "rb");
	if (infile == NULL)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not open express file " << rank_filename << "!" << endl;
		throw std::runtime_error("Could not open express input file");
	}

	express_header ehdr;
	if (fread(&ehdr, sizeof(ehdr), 1, infile) != 1)
	{
		fclose(infile);
		throw std::runtime_error("Could not read express header");
	}

	bool layout_ok = true;
	layout_ok = layout_ok && memcmp(ehdr.magic, EXPRESS_MAGIC, sizeof(ehdr.magic)) == 0;
	layout_ok = layout_ok && ehdr.version == EXPRESS_VERSION;
	layout_ok = layout_ok && ehdr.header_size == sizeof(ehdr);
	layout_ok = layout_ok && ehdr.endian == EXPRESS_ENDIAN;
	layout_ok = layout_ok && ehdr.sizeof_real == sizeof(Real);
	layout_ok = layout_ok && ehdr.sizeof_long == sizeof(long);
	layout_ok = layout_ok && ehdr.sizeof_gadget2_header == sizeof(gadget2_header);
	layout_ok = layout_ok && ehdr.rank == parallel.rank();
	layout_ok = layout_ok && ehdr.size == parallel.size();
	layout_ok = layout_ok && ehdr.grid_size[0] == parallel.grid_size()[0];
	layout_ok = layout_ok && ehdr.grid_size[1] == parallel.grid_size()[1];
	for (int i = 0; i < 3; i++)
	{
		layout_ok = layout_ok && ehdr.lat_size[i] == this->lat_size_[i];
		layout_ok = layout_ok && ehdr.lat_size_local[i] == this->lat_size_local_[i];
	}
	layout_ok = layout_ok && ehdr.coord_skip[0] == this->coordSkip_[0];
	layout_ok = layout_ok && ehdr.coord_skip[1] == this->coordSkip_[1];
	layout_ok = layout_ok && ehdr.position_bytes == 3 * ehdr.local_npart * sizeof(Real);
	layout_ok = layout_ok && ehdr.momentum_bytes == 3 * ehdr.local_npart * sizeof(Real);
	layout_ok = layout_ok && ehdr.id_bytes == ehdr.local_npart * sizeof(long);

	if (!layout_ok)
	{
		fclose(infile);
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": express file " << rank_filename << " is incompatible with this run layout or ABI!" << endl;
		throw std::runtime_error("Incompatible express input file");
	}

	if (this->total_capacity_ < ehdr.local_npart)
		this->resizeGlobalBuffers(ehdr.local_npart + this->extra_capacity_);

	Real * real_buffer = (Real *) malloc(3 * sizeof(Real) * PCLBUFFER);
	long * id_buffer = (long *) malloc(sizeof(long) * PCLBUFFER);
	if (real_buffer == NULL || id_buffer == NULL)
	{
		fclose(infile);
		free(real_buffer);
		free(id_buffer);
		throw std::runtime_error("Could not allocate express staging buffers");
	}

	uint64_t read = 0;
	while (read < ehdr.local_npart)
	{
		uint64_t count = (ehdr.local_npart - read > PCLBUFFER) ? PCLBUFFER : (ehdr.local_npart - read);
		if (fread(real_buffer, sizeof(Real), 3 * count, infile) != 3 * count)
		{
			fclose(infile);
			free(real_buffer);
			free(id_buffer);
			throw std::runtime_error("Could not read express positions");
		}
		cudaMemcpy(this->p + 3 * read, real_buffer, 3 * count * sizeof(Real), cudaMemcpyHostToDevice);
		read += count;
	}

	read = 0;
	while (read < ehdr.local_npart)
	{
		uint64_t count = (ehdr.local_npart - read > PCLBUFFER) ? PCLBUFFER : (ehdr.local_npart - read);
		if (fread(real_buffer, sizeof(Real), 3 * count, infile) != 3 * count)
		{
			fclose(infile);
			free(real_buffer);
			free(id_buffer);
			throw std::runtime_error("Could not read express momenta");
		}
		cudaMemcpy(this->q + 3 * read, real_buffer, 3 * count * sizeof(Real), cudaMemcpyHostToDevice);
		read += count;
	}

	read = 0;
	while (read < ehdr.local_npart)
	{
		uint64_t count = (ehdr.local_npart - read > PCLBUFFER) ? PCLBUFFER : (ehdr.local_npart - read);
		if (fread(id_buffer, sizeof(long), count, infile) != count)
		{
			fclose(infile);
			free(real_buffer);
			free(id_buffer);
			throw std::runtime_error("Could not read express IDs");
		}
		cudaMemcpy(this->other + read, id_buffer, count * sizeof(long), cudaMemcpyHostToDevice);
		read += count;
	}

	if (fread(&hdr, sizeof(hdr), 1, infile) != 1)
	{
		fclose(infile);
		free(real_buffer);
		free(id_buffer);
		throw std::runtime_error("Could not read express Gadget2 metadata");
	}

	uint64_t hdr_global_npart = (uint64_t) hdr.npartTotal[1] + ((uint64_t) hdr.npartTotalHW[1] << 32);
	if (hdr_global_npart != ehdr.global_npart || hdr.num_files != ehdr.size)
	{
		fclose(infile);
		free(real_buffer);
		free(id_buffer);
		throw std::runtime_error("Express Gadget2 metadata does not match express header");
	}

	fclose(infile);
	free(real_buffer);
	free(id_buffer);

	this->num_particles_ = ehdr.local_npart;
	if (this->num_particles_ > 0)
		this->updateRowBuffers();
}


// This particle load function assumes that each rank reads exactly one file, no MPI communication needed
template <typename part, typename part_info>
void perfParticles_gevolution<part,part_info>::loadGadget2_express(string filename, gadget2_header & hdr)
{
	float * posdata;
	float * veldata;
	void * IDs;
	FILE * infile;
	uint32_t count, npart = 0;
	uint64_t offset_pos, offset_vel, offset_ID;
	uint32_t blocksize;
	double rescale_vel = 1. / GADGET_VELOCITY_CONVERSION;
	
	posdata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);
	veldata = (float *) malloc(3 * sizeof(float) * PCLBUFFER);

#if GADGET_ID_BYTES == 8
	IDs = malloc(sizeof(int64_t) * PCLBUFFER);
#else
	IDs = malloc(sizeof(int32_t) * PCLBUFFER);
#endif

	size_t reader_workspace_bytes = particle_io_staging_workspace_bytes(PCLBUFFER);
	DeviceWorkspace reader_workspace(reader_workspace_bytes, "legacy express Gadget-2 particle input", "particle staging buffers");
	float * d_posdata = reader_workspace.slice<float>(3 * PCLBUFFER, "position staging buffer");
	float * d_veldata = reader_workspace.slice<float>(3 * PCLBUFFER, "velocity staging buffer");
	void * d_IDs = reader_workspace.slice_bytes(sizeof(int64_t) * PCLBUFFER, 256, "ID staging buffer");
	unsigned long long int * d_buffer_idx = reader_workspace.slice<unsigned long long int>(1, "particle insertion counter");

	filename = filename.substr(0, filename.find_last_of('.')+1) + to_string(parallel.rank());

	infile = fopen(filename.c_str(), "rb");
	
	// check if file could be opened
	if (infile == NULL)
	{
		cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not open file " << filename << "!" << endl;
		return;
	}

	auto read_success = fread(&blocksize, 1, sizeof(blocksize), infile);

	if (blocksize != sizeof(hdr) || read_success != sizeof(blocksize))
	{
		cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": file type not recognized when reading Gadget2 file!" << endl;
		fclose(infile);
		return;
	}

	read_success = fread(&hdr, 1, sizeof(hdr), infile);

	if (read_success != sizeof(hdr))
	{
		cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read header from file " << filename << "!" << endl;
		fclose(infile);
		return;
	}

	rescale_vel /= sqrt(hdr.time);
	offset_pos = (uint64_t) sizeof(hdr) + (uint64_t) (3 * sizeof(uint32_t));
	offset_vel = offset_pos + ((uint64_t) hdr.npart[1]) * ((uint64_t) (3 * sizeof(float))) + (uint64_t) (2 * sizeof(uint32_t));
	offset_ID = offset_vel + offset_vel - offset_pos;

	cout << " rank#" << parallel.rank() << ": loading " << hdr.npart[1] << " particles from file " << filename << endl;

	fseek(infile, offset_pos, SEEK_SET);
	while (npart < hdr.npart[1])
	{
		count = (hdr.npart[1] - npart > PCLBUFFER) ? PCLBUFFER : (hdr.npart[1] - npart);

		read_success = fread(posdata, sizeof(float), 3 * count, infile);

		if (read_success != 3 * count)
		{
			cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read particle positions from file " << filename << "!" << endl;
			fclose(infile);
			return;
		}

		offset_pos += (uint64_t) (3 * count * sizeof(float));
		fseek(infile, offset_vel, SEEK_SET);
		read_success = fread(veldata, sizeof(float), 3 * count, infile);

		if (read_success != 3 * count)
		{
			cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read particle velocities from file " << filename << "!" << endl;
			fclose(infile);
			return;
		}

		offset_vel += (uint64_t) (3 * count * sizeof(float));
		fseek(infile, offset_ID, SEEK_SET);
#if GADGET_ID_BYTES == 8
		read_success = fread(IDs, sizeof(int64_t), count, infile);
		offset_ID += (uint64_t) (count * sizeof(int64_t));
#else
		read_success = fread(IDs, sizeof(int32_t), count, infile);
		offset_ID += (uint64_t) (count * sizeof(int32_t));
#endif
		fseek(infile, offset_pos, SEEK_SET);

		if (read_success != count)
		{
			cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": could not read particle IDs from file " << filename << "!" << endl;
			fclose(infile);
			return;
		}

#pragma omp parallel for
		for (int i = 0; i < 3 * count; i++)
		{
			posdata[i] /= hdr.BoxSize;
			while (posdata[i] < 0) posdata[i] += 1;
			while (posdata[i] >= 1) posdata[i] -= 1;
			veldata[i] *= hdr.time / rescale_vel;
		}

		if (this->num_particles_ + count > this->total_capacity_)
		{
			this->resizeGlobalBuffers(this->num_particles_ + count + this->extra_capacity_);
		}

		unsigned long long int buffer_idx = this->num_particles_;
		cudaMemcpy(d_buffer_idx, &buffer_idx, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_posdata, posdata, 3 * count * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_veldata, veldata, 3 * count * sizeof(float), cudaMemcpyHostToDevice);
#if GADGET_ID_BYTES == 8
		cudaMemcpy(d_IDs, IDs, count * sizeof(int64_t), cudaMemcpyHostToDevice);
#else
		cudaMemcpy(d_IDs, IDs, count * sizeof(int32_t), cudaMemcpyHostToDevice);
#endif

		add_particles<part, part_info><<<count/128+1, 128>>>(this, d_posdata, d_veldata, d_IDs, count, d_buffer_idx);

		auto success = cudaDeviceSynchronize();

		if (success != cudaSuccess)
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": CUDA error when loading particles!" << endl;
			fclose(infile);
			throw std::runtime_error("CUDA error when loading particles!");
		}

		this->num_particles_ += count;

		npart += count;
	}

	fclose(infile);

	this->updateRowBuffers();
	
	free(posdata);
	free(veldata);
	free(IDs);
}

#endif
