//////////////////////////
// cuda_aware_mpi.hpp
//////////////////////////
//
// Runtime policy for communication paths that may pass CUDA device pointers
// directly to MPI.
//
//////////////////////////

#ifndef CUDA_AWARE_MPI_HEADER
#define CUDA_AWARE_MPI_HEADER

#include <cstdlib>
#include <cstring>

inline bool gevolution_env_var_enabled(const char * name)
{
	const char * value = std::getenv(name);
	if (value == NULL) return false;
	return !(std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0);
}

inline bool gevolution_cuda_aware_mpi_active()
{
	const bool cuda_aware_mpi_hint =
		gevolution_env_var_enabled("LATFIELD2_ENABLE_CUDA_AWARE_MPI") ||
		gevolution_env_var_enabled("MPICH_GPU_SUPPORT_ENABLED") ||
		gevolution_env_var_enabled("MV2_USE_CUDA") ||
		gevolution_env_var_enabled("PSM2_CUDA") ||
		gevolution_env_var_enabled("OMPI_MCA_opal_cuda_support") ||
		gevolution_env_var_enabled("OMPI_MCA_mpi_cuda_support");

	return !gevolution_env_var_enabled("LATFIELD2_DISABLE_CUDA_AWARE_MPI") &&
		cuda_aware_mpi_hint;
}

#endif
