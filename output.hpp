//////////////////////////
// output.hpp
//////////////////////////
//
// Output of snapshots, light cones and spectra
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich & ETH Zürich)
//
// Last modified: April 2026
//
//////////////////////////

#ifndef OUTPUT_HEADER
#define OUTPUT_HEADER

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <stdint.h>
#include <nvtx3/nvToolsExt.h>

using namespace std;


//////////////////////////
// writeSnapshots
//////////////////////////
// Description:
//   output of snapshots
//
// Arguments:
//   sim            simulation metadata structure
//   cosmo          cosmological parameter structure
//   fourpiG        4 pi G (in code units)
//   a              scale factor
//   snapcount      snapshot index
//   h5filename     base name for HDF5 output file
//   pcls_cdm       pointer to particle handler for CDM
//   pcls_b         pointer to particle handler for baryons
//   pcls_ncdm      array of particle handlers for
//                  non-cold DM (may be set to NULL)
//   phi            pointer to allocated field
//   chi            pointer to allocated field
//   Bi             pointer to allocated field
//   source         pointer to allocated field
//   Sij            pointer to allocated field
//   scalarFT       pointer to allocated field
//   BiFT           pointer to allocated field
//   SijFT          pointer to allocated field
//   plan_phi       pointer to FFT planner
//   plan_chi       pointer to FFT planner
//   plan_Bi        pointer to FFT planner
//   plan_source    pointer to FFT planner
//   plan_Sij       pointer to FFT planner
//   Bi_check       pointer to allocated field
//   BiFT_check     pointer to allocated field
//   plan_Bi_check  pointer to FFT planner
//   vi             pointer to allocated field
//
// Returns:
//
//////////////////////////

void writeSnapshots(metadata & sim, cosmology & cosmo, const double fourpiG, const double a, const double dtau_old, const int done_hij, const int snapcount, string h5filename, perfParticles_gevolution<part_simple,part_simple_info> * pcls_cdm, perfParticles_gevolution<part_simple,part_simple_info> * pcls_b, Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> * pcls_ncdm, Field<Real> * phi, Field<Real> * chi, Field<Real> * Bi, Field<Real> * source, Field<Real> * Sij, Field<Cplx> * scalarFT, Field<Cplx> * BiFT, Field<Cplx> * SijFT, PlanFFT<Cplx> * plan_phi, PlanFFT<Cplx> * plan_chi, PlanFFT<Cplx> * plan_Bi, PlanFFT<Cplx> * plan_source, PlanFFT<Cplx> * plan_Sij
#ifdef CHECK_B
, Field<Real> * Bi_check, Field<Cplx> * BiFT_check, PlanFFT<Cplx> * plan_Bi_check
#endif
#ifdef VELOCITY
, Field<Real> * vi
#endif
)
{
	char filename[2*PARAM_MAX_LENGTH+64];
	char buffer[64];
	gadget2_header hdr;
	Site x(phi->lattice());
	Real divB, curlB, divh, traceh, normh;
	double dtau_pos = 0.;

	sprintf(filename, "%03d", snapcount);

#ifdef EXTERNAL_IO
	while (ioserver.openOstream()== OSTREAM_FAIL);

	if (sim.out_snapshot & MASK_PCLS)
	{
		/*pcls_cdm->saveHDF5_server_open(h5filename + filename + "_cdm");
		if (sim.baryon_flag)
			pcls_b->saveHDF5_server_open(h5filename + filename + "_b");*/
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			sprintf(buffer, "_ncdm%d", i);
			pcls_ncdm[i].saveHDF5_server_open(h5filename + filename + buffer);
		}
	}

	if (sim.out_snapshot & MASK_T00)
		source->saveHDF5_server_open(h5filename + filename + "_T00");

#ifdef VELOCITY
	if (sim.out_snapshot & MASK_VEL)
		vi->saveHDF5_server_open(h5filename + filename + "_v");
#endif

	if (sim.out_snapshot & MASK_B)
		Bi->saveHDF5_server_open(h5filename + filename + "_B");

	if (sim.out_snapshot & MASK_PHI)
		phi->saveHDF5_server_open(h5filename + filename + "_phi");

	if (sim.out_snapshot & MASK_CHI)
		chi->saveHDF5_server_open(h5filename + filename + "_chi");

	if (sim.out_snapshot & MASK_HIJ)
		Sij->saveHDF5_server_open(h5filename + filename + "_hij");

#ifdef CHECK_B
	if (sim.out_snapshot & MASK_B)
		Bi_check->saveHDF5_server_open(h5filename + filename + "_B_check");
#endif
#endif

	if (sim.out_snapshot & MASK_RBARE || sim.out_snapshot & MASK_POT)
	{
		//projection_init(source);
		nvtxRangePushA("particle projection");
		thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
		scalarProjectionCIC_project(pcls_cdm, source);
		if (sim.baryon_flag)
			scalarProjectionCIC_project(pcls_b, source);
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			scalarProjectionCIC_project(pcls_ncdm+i, source);
		}
		scalarProjectionCIC_comm(source);
		nvtxRangePop();
	}

	if (sim.out_snapshot & MASK_RBARE)
	{
		if (sim.downgrade_factor > 1)
			source->saveHDF5_coarseGrain3D(h5filename + filename + "_rhoN.h5", sim.downgrade_factor);
		else
			source->saveHDF5(h5filename + filename + "_rhoN.h5");
	}

	if (sim.out_snapshot & MASK_POT)
	{
		plan_source->execute(FFT_FORWARD);
		solveModifiedPoissonFT(*scalarFT, *scalarFT, fourpiG / a);
		plan_source->execute(FFT_BACKWARD);
		if (sim.downgrade_factor > 1)
			source->saveHDF5_coarseGrain3D(h5filename + filename + "_psiN.h5", sim.downgrade_factor);
		else
			source->saveHDF5(h5filename + filename + "_psiN.h5");
	}

	if (sim.out_snapshot & MASK_T00)
	{
		nvtxRangePushA("T00 output");
		//projection_init(source);
		thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
		if (sim.gr_flag > 0)
		{
			projection_T00_project(pcls_cdm, source, a, phi);
			if (sim.baryon_flag)
				projection_T00_project(pcls_b, source, a, phi);
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
				projection_T00_project(pcls_ncdm+i, source, a, phi);
			}
		}
		else
		{
			scalarProjectionCIC_project(pcls_cdm, source);
			if (sim.baryon_flag)
				scalarProjectionCIC_project(pcls_b, source);
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
				scalarProjectionCIC_project(pcls_ncdm+i, source);
			}
		}
		projection_T00_comm(source);
#ifdef EXTERNAL_IO
		source->saveHDF5_server_write(NUMBER_OF_IO_FILES);
#else
		if (sim.downgrade_factor > 1)
			source->saveHDF5_coarseGrain3D(h5filename + filename + "_T00.h5", sim.downgrade_factor);
		else
			source->saveHDF5(h5filename + filename + "_T00.h5");
#endif
		nvtxRangePop();
	}

#ifdef VELOCITY
	if (sim.out_snapshot & MASK_VEL)
	{
		nvtxRangePushA("v output");
#ifdef EXTERNAL_IO
		vi->saveHDF5_server_write(NUMBER_OF_IO_FILES);
#else
		if (sim.downgrade_factor > 1)
			vi->saveHDF5_coarseGrain3D(h5filename + filename + "_v.h5", sim.downgrade_factor);
		else
			vi->saveHDF5(h5filename + filename + "_v.h5");
#endif
		nvtxRangePop();
	}
#endif

	if (sim.out_snapshot & MASK_B)
	{
		nvtxRangePushA("B output");
		if (sim.gr_flag == 0)
		{
			plan_Bi->execute(FFT_BACKWARD);
		}

		double params = 1. / (a * a * sim.numpts);
		double * d_params;
		cudaMalloc((void **) &d_params, sizeof(double));
		cudaMemcpy(d_params, &params, sizeof(double), cudaMemcpyDefault);

		lattice_for_each<<<dim3(Bi->lattice().sizeLocal(1), Bi->lattice().sizeLocal(2)), 128>>>(lattice_multiply_functor<3>(), sim.numpts, &Bi, 1, d_params, nullptr, nullptr);

		cudaDeviceSynchronize();
		cudaFree(d_params);

		Bi->updateHalo();

		computeVectorDiagnostics(*Bi, divB, curlB);
		COUT << " B diagnostics: max |divB| = " << divB << ", max |curlB| = " << curlB << endl;

#ifdef EXTERNAL_IO
		Bi->saveHDF5_server_write(NUMBER_OF_IO_FILES);
#else
		if (sim.downgrade_factor > 1)
			Bi->saveHDF5_coarseGrain3D(h5filename + filename + "_B.h5", sim.downgrade_factor);
		else
			Bi->saveHDF5(h5filename + filename + "_B.h5");
#endif

		if (sim.gr_flag > 0)
		{
			plan_Bi->execute(FFT_BACKWARD);
			Bi->updateHalo();
		}
		nvtxRangePop();
	}

	if (sim.out_snapshot & MASK_PHI)
	{
		nvtxRangePushA("phi output");
#ifdef EXTERNAL_IO
		phi->saveHDF5_server_write(NUMBER_OF_IO_FILES);
#else
		if (sim.downgrade_factor > 1)
			phi->saveHDF5_coarseGrain3D(h5filename + filename + "_phi.h5", sim.downgrade_factor);
		else
			phi->saveHDF5(h5filename + filename + "_phi.h5");
#endif
		nvtxRangePop();
	}

	if (sim.out_snapshot & MASK_CHI)
	{
		nvtxRangePushA("chi output");
#ifdef EXTERNAL_IO
		chi->saveHDF5_server_write(NUMBER_OF_IO_FILES);
#else
		if (sim.downgrade_factor > 1)
			chi->saveHDF5_coarseGrain3D(h5filename + filename + "_chi.h5", sim.downgrade_factor);
		else
			chi->saveHDF5(h5filename + filename + "_chi.h5");
#endif
		nvtxRangePop();
	}

	if (sim.out_snapshot & MASK_HIJ)
	{
		nvtxRangePushA("hij output");
		if (done_hij == 0)
		{
			projectFTtensor(*SijFT, *SijFT);
			plan_Sij->execute(FFT_BACKWARD);
			Sij->updateHalo();
		}

		computeTensorDiagnostics(*Sij, divh, traceh, normh);
		COUT << " GW diagnostics: max |divh| = " << divh << ", max |traceh| = " << traceh << ", max |h| = " << normh << endl;

#ifdef EXTERNAL_IO
		Sij->saveHDF5_server_write(NUMBER_OF_IO_FILES);
#else
		if (sim.downgrade_factor > 1)
			Sij->saveHDF5_coarseGrain3D(h5filename + filename + "_hij.h5", sim.downgrade_factor);
		else
			Sij->saveHDF5(h5filename + filename + "_hij.h5");
#endif
		nvtxRangePop();
	}

	if (sim.out_snapshot & MASK_TIJ)
	{
		nvtxRangePushA("Tij output");
		//projection_init(Sij);
		thrust::fill_n(thrust::device, Sij->data(), 6*Sij->lattice().sitesLocalGross(), Real(0));
		projection_Tij_project(pcls_cdm, Sij, a, phi);
		if (sim.baryon_flag)
			projection_Tij_project(pcls_b, Sij, a, phi);
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			projection_Tij_project(pcls_ncdm+i, Sij, a, phi);
		}
		projection_Tij_comm(Sij);

		if (sim.downgrade_factor > 1)
			Sij->saveHDF5_coarseGrain3D(h5filename + filename + "_Tij.h5", sim.downgrade_factor);
		else
			Sij->saveHDF5(h5filename + filename + "_Tij.h5");
		nvtxRangePop();
	}

	if (sim.out_snapshot & MASK_P)
	{
		nvtxRangePushA("p (momentum density) output");
		//projection_init(Bi);
		thrust::fill_n(thrust::device, Bi->data(), 3*Bi->lattice().sitesLocalGross(), Real(0));
		projection_T0i_project(pcls_cdm, Bi, phi);
		if (sim.baryon_flag)
			projection_T0i_project(pcls_b, Bi, phi);
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			projection_T0i_project(pcls_ncdm+i, Bi, phi);
		}
		projection_T0i_comm(Bi);
		if (sim.downgrade_factor > 1)
			Bi->saveHDF5_coarseGrain3D(h5filename + filename + "_p.h5", sim.downgrade_factor);
		else
			Bi->saveHDF5(h5filename + filename + "_p.h5");
		if (sim.gr_flag > 0)
		{
			plan_Bi->execute(FFT_BACKWARD);
			Bi->updateHalo();
		}
		nvtxRangePop();
	}

#ifdef CHECK_B
	if (sim.out_snapshot & MASK_B)
	{
		nvtxRangePushA("B check output");
		if (sim.vector_flag == VECTOR_PARABOLIC)
		{
			//projection_init(Bi_check);
			thrust::fill_n(thrust::device, Bi_check->data(), 3*Bi_check->lattice().sitesLocalGross(), Real(0));
			projection_T0i_project(pcls_cdm, Bi_check, phi);
			if (sim.baryon_flag)
				projection_T0i_project(pcls_b, Bi_check, phi);
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
				projection_T0i_project(pcls_ncdm+i, Bi_check, phi);
			}
			projection_T0i_comm(Bi_check);
			plan_Bi_check->execute(FFT_FORWARD);
			projectFTvector(*BiFT_check, *BiFT_check, fourpiG / (double) sim.numpts / (double) sim.numpts);
		}
		plan_Bi_check->execute(FFT_BACKWARD);

		double params = 1. / (a * a * sim.numpts);
		double * d_params;
		cudaMalloc((void **) &d_params, sizeof(double));
		cudaMemcpy(d_params, &params, sizeof(double), cudaMemcpyDefault);

		lattice_for_each<<<dim3(Bi_check->lattice().sizeLocal(1), Bi_check->lattice().sizeLocal(2)), 128>>>(lattice_multiply_functor<3>(), sim.numpts, &Bi_check, 1, d_params, nullptr, nullptr);

		cudaDeviceSynchronize();
		cudaFree(d_params);

#ifdef EXTERNAL_IO
		Bi_check->saveHDF5_server_write(NUMBER_OF_IO_FILES);
#else
		if (sim.downgrade_factor > 1)
			Bi_check->saveHDF5_coarseGrain3D(h5filename + filename + "_B_check.h5", sim.downgrade_factor);
		else
			Bi_check->saveHDF5(h5filename + filename + "_B_check.h5");
#endif
		nvtxRangePop();
	}
#endif

	if (sim.out_snapshot & MASK_GADGET)
	{
		nvtxRangePushA("Gadget2 output");
		if (sim.out_snapshot & MASK_MULTI)
			hdr.num_files = parallel.size();
		else
			hdr.num_files = 1;
		hdr.Omega0 = cosmo.Omega_m;
		hdr.OmegaLambda = cosmo.Omega_Lambda;
		hdr.HubbleParam = cosmo.h;
		hdr.BoxSize = sim.boxsize / GADGET_LENGTH_CONVERSION;
		hdr.flag_sfr = 0;
		hdr.flag_cooling = 0;
		hdr.flag_feedback = 0;
		hdr.flag_age = 0;
		hdr.flag_metals = 0;
		for (int i = 0; i < 256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8 - 2 * 4 - 6 * 4; i++)
			hdr.fill[i] = 0;
		for (int i = 0; i < 6; i++)
		{
			hdr.npart[i] = 0;
			hdr.npartTotal[i] = 0;
			hdr.npartTotalHW[i] = 0;
			hdr.mass[i] = 0.;
		}

#ifdef EXACT_OUTPUT_REDSHIFTS
		hdr.time = 1. / (sim.z_snapshot[snapcount] + 1.);
		hdr.redshift = sim.z_snapshot[snapcount];
		dtau_pos = (hdr.time - a) / a / Hconf(a, fourpiG, cosmo);
#else
		hdr.time = a;
		hdr.redshift = (1./a) - 1.;
#endif

		if (sim.tracer_factor[0] > 0)
		{
			hdr.npart[1] = (uint32_t) (((sim.numpcl[0] % sim.tracer_factor[0]) ? (1 + (sim.numpcl[0] / sim.tracer_factor[0])) : (sim.numpcl[0] / sim.tracer_factor[0])) % (1ll << 32));
			hdr.npartTotal[1] = hdr.npart[1];
			hdr.npartTotalHW[1] = (uint32_t) (((sim.numpcl[0] % sim.tracer_factor[0]) ? (1 + (sim.numpcl[0] / sim.tracer_factor[0])) : (sim.numpcl[0] / sim.tracer_factor[0])) / (1ll << 32));
			if (sim.baryon_flag)
				hdr.mass[1] = (double) sim.tracer_factor[0] * C_RHO_CRIT * cosmo.Omega_cdm * sim.boxsize * sim.boxsize * sim.boxsize / sim.numpcl[0] / GADGET_MASS_CONVERSION;
			else
				hdr.mass[1] = (double) sim.tracer_factor[0] * C_RHO_CRIT * (cosmo.Omega_cdm + cosmo.Omega_b) * sim.boxsize * sim.boxsize * sim.boxsize / sim.numpcl[0] / GADGET_MASS_CONVERSION;

			if (hdr.npartTotalHW[1] != 0 && hdr.num_files == 1)
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": number of particles (" << (sim.numpcl[0] / sim.tracer_factor[0]) << ") in Gadget2 file exceeds limit (4294967295). Try using multi-Gadget2 output format." << endl;
			}
			else
				pcls_cdm->saveGadget2(h5filename + filename + "_cdm", hdr, sim.tracer_factor[0], dtau_pos, dtau_pos + 0.5 * dtau_old, phi);
		}

		if (sim.baryon_flag && sim.tracer_factor[1] > 0)
		{
			hdr.npart[1] = (uint32_t) (((sim.numpcl[1] % sim.tracer_factor[1]) ? (1 + (sim.numpcl[1] / sim.tracer_factor[1])) : (sim.numpcl[1] / sim.tracer_factor[1])) % (1ll << 32));
			hdr.npartTotal[1] = hdr.npart[1];
			hdr.npartTotalHW[1] = (uint32_t) (((sim.numpcl[1] % sim.tracer_factor[1]) ? (1 + (sim.numpcl[1] / sim.tracer_factor[1])) : (sim.numpcl[1] / sim.tracer_factor[1])) / (1ll << 32));
			hdr.mass[1] = (double) sim.tracer_factor[1] * C_RHO_CRIT * cosmo.Omega_b * sim.boxsize * sim.boxsize * sim.boxsize / sim.numpcl[1] / GADGET_MASS_CONVERSION;
			if (hdr.npartTotalHW[1] != 0 && hdr.num_files == 1)
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": number of particles (" << (sim.numpcl[1] / sim.tracer_factor[1]) << ") in Gadget2 file exceeds limit (4294967295). Try using multi-Gadget2 output format." << endl;
			}
			else
				pcls_b->saveGadget2(h5filename + filename + "_b", hdr, sim.tracer_factor[1], dtau_pos, dtau_pos + 0.5 * dtau_old, phi);
		}

		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.out_snapshot & MASK_MULTI)
				hdr.num_files = parallel.grid_size()[1];

			if (sim.numpcl[1+sim.baryon_flag+i] == 0 || sim.tracer_factor[i+1+sim.baryon_flag] == 0) continue;
			sprintf(buffer, "_ncdm%d", i);
			hdr.npart[1] = (uint32_t) (((sim.numpcl[i+1+sim.baryon_flag] % sim.tracer_factor[i+1+sim.baryon_flag]) ? (1 + (sim.numpcl[i+1+sim.baryon_flag] / sim.tracer_factor[i+1+sim.baryon_flag])) : (sim.numpcl[i+1+sim.baryon_flag] / sim.tracer_factor[i+1+sim.baryon_flag])) % (1ll << 32));
			hdr.npartTotal[1] = hdr.npart[1];
			hdr.npartTotalHW[1] = (uint32_t) (((sim.numpcl[i+1+sim.baryon_flag] % sim.tracer_factor[i+1+sim.baryon_flag]) ? (1 + (sim.numpcl[i+1+sim.baryon_flag] / sim.tracer_factor[i+1+sim.baryon_flag])) : (sim.numpcl[i+1+sim.baryon_flag] / sim.tracer_factor[i+1+sim.baryon_flag])) / (1ll << 32));
			hdr.mass[1] = (double) sim.tracer_factor[i+1+sim.baryon_flag] * C_RHO_CRIT * cosmo.Omega_ncdm[i] * sim.boxsize * sim.boxsize * sim.boxsize / sim.numpcl[i+1+sim.baryon_flag] / GADGET_MASS_CONVERSION;
			if (hdr.npartTotalHW[1] != 0 && hdr.num_files == 1)
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": number of particles (" << (sim.numpcl[i+1+sim.baryon_flag] / sim.tracer_factor[i+1+sim.baryon_flag]) << ") in Gadget2 file exceeds limit (4294967295). Try using multi-Gadget2 output format." << endl;
			}
			else
				pcls_ncdm[i].saveGadget2(h5filename + filename + buffer, hdr, sim.tracer_factor[i+1+sim.baryon_flag], dtau_pos, dtau_pos + 0.5 * dtau_old, phi);
		}
		nvtxRangePop();
	}

	if (sim.out_snapshot & MASK_PCLS)
	{
		nvtxRangePushA("particle HDF5 output");
#ifdef EXTERNAL_IO
		/*pcls_cdm->saveHDF5_server_write();
		if (sim.baryon_flag)
			pcls_b->saveHDF5_server_write();*/
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			pcls_ncdm[i].saveHDF5_server_write();
		}
#else
		/*pcls_cdm->saveHDF5(h5filename + filename + "_cdm", 1);
		if (sim.baryon_flag)
			pcls_b->saveHDF5(h5filename + filename + "_b", 1);*/
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			sprintf(buffer, "_ncdm%d", i);
			pcls_ncdm[i].saveHDF5(h5filename + filename + buffer, 1);
		}
#endif
		nvtxRangePop();
	}

#ifdef EXTERNAL_IO
	ioserver.closeOstream();
#endif
}

#ifdef HAVE_HEALPIX
#ifndef HEALPIX_SHELL_CHUNK
#define HEALPIX_SHELL_CHUNK 16
#endif

// CUDA kernel for projection of metric to Healpix maps
__global__ void project_metric_to_healpix_batch(Real * pixbuf_phi, Real * pixbuf_chi, Real * pixbuf_B1, Real * pixbuf_B2, Real * pixbuf_B3, Real * pixbuf_h11, Real * pixbuf_h12, Real * pixbuf_h13, Real * pixbuf_h22, Real * pixbuf_h23, int64_t nside, int64_t pix, Real a2, double dist, double vertex[3], double R[3][3], int numpts, Field<Real> ** fields, int outputs, int batchsize, int64_t * packmap)
{
	int64_t q = blockIdx.x * blockDim.x + threadIdx.x;

	if (q >= batchsize)
		return;

	double w[3];

	pix2vec_nest64_gpu(nside, pix + q, w);

	if (packmap != nullptr)
	{
		q = packmap[q];

		if (q < 0)
			return;
	}

	double pos[3];
	double temp;
	int base_pos[3];

	pos[0] = (dist * (R[0][0] * w[0] + R[0][1] * w[1] + R[0][2] * w[2]) + vertex[0]) * numpts;
	pos[1] = (dist * (R[1][0] * w[0] + R[1][1] * w[1] + R[1][2] * w[2]) + vertex[1]) * numpts;
	pos[2] = (dist * (R[2][0] * w[0] + R[2][1] * w[1] + R[2][2] * w[2]) + vertex[2]) * numpts;

	if (pos[0] >= 0)
	{
		w[0] = modf(pos[0], &temp);
		base_pos[0] = static_cast<int>(temp) % numpts;
	}
	else
	{
		w[0] = 1. + modf(pos[0], &temp);
		base_pos[0] = numpts - 1 - (static_cast<int>(-temp) % numpts);
	}
	if (pos[1] >= 0)
	{
		w[1] = modf(pos[1], &temp);
		base_pos[1] = (static_cast<int>(temp) % numpts) - fields[0]->lattice().coordSkip()[1];
	}
	else
	{
		w[1] = 1. + modf(pos[1], &temp);
		base_pos[1] = numpts - 1 - (static_cast<int>(-temp) % numpts) - fields[0]->lattice().coordSkip()[1];
	}
	if (pos[2] >= 0)
	{
		w[2] = modf(pos[2], &temp);
		base_pos[2] = (static_cast<int>(temp) % numpts) - fields[0]->lattice().coordSkip()[0];
	}
	else
	{
		w[2] = 1. + modf(pos[2], &temp);
		base_pos[2] = numpts - 1 - (static_cast<int>(-temp) % numpts) - fields[0]->lattice().coordSkip()[0];
	}

	if (base_pos[1] >= 0 && base_pos[1] < fields[0]->lattice().sizeLocal(1) && base_pos[2] >= 0 && base_pos[2] < fields[0]->lattice().sizeLocal(2))
	{
		Site xsim = Site(fields[0]->lattice(), fields[0]->lattice().siteFirst() + base_pos[0]*fields[0]->lattice().jump(0) + base_pos[1]*fields[0]->lattice().jump(1) + base_pos[2]*fields[0]->lattice().jump(2));

		if (outputs & MASK_PHI)
		{
			*(pixbuf_phi+q) = (1.-w[0]) * (1.-w[1]) * ((1.-w[2]) * (*fields[0])(xsim) + w[2] * (*fields[0])(xsim+2));
			*(pixbuf_phi+q) += w[0] * (1.-w[1]) * ((1.-w[2]) * (*fields[0])(xsim+0) + w[2] * (*fields[0])(xsim+0+2));
			*(pixbuf_phi+q) += w[0] * w[1] * ((1.-w[2]) * (*fields[0])(xsim+0+1) + w[2] * (*fields[0])(xsim+0+1+2));
			*(pixbuf_phi+q) += (1.-w[0]) * w[1] * ((1.-w[2]) * (*fields[0])(xsim+1) + w[2] * (*fields[0])(xsim+1+2));
		}
		if (outputs & MASK_CHI)
		{
			*(pixbuf_chi+q) = (1.-w[0]) * (1.-w[1]) * ((1.-w[2]) * (*fields[1])(xsim) + w[2] * (*fields[1])(xsim+2));
			*(pixbuf_chi+q) += w[0] * (1.-w[1]) * ((1.-w[2]) * (*fields[1])(xsim+0) + w[2] * (*fields[1])(xsim+0+2));
			*(pixbuf_chi+q) += w[0] * w[1] * ((1.-w[2]) * (*fields[1])(xsim+0+1) + w[2] * (*fields[1])(xsim+0+1+2));
			*(pixbuf_chi+q) += (1.-w[0]) * w[1] * ((1.-w[2]) * (*fields[1])(xsim+1) + w[2] * (*fields[1])(xsim+1+2));
		}
		if (outputs & MASK_B)
		{
#ifdef LIGHTCONE_INTERPOLATE
			*(pixbuf_B1+q) = (1.-w[2]) * (1.-w[1]) * ((1.-w[0]) * (*fields[2])(xsim-0,0) + w[0] * (*fields[2])(xsim+0,0) + (*fields[2])(xsim,0));
			*(pixbuf_B1+q) += w[2] * (1.-w[1]) * ((1.-w[0]) * (*fields[2])(xsim-0+2,0) + w[0] * (*fields[2])(xsim+0+2,0) + (*fields[2])(xsim+2,0));
			*(pixbuf_B1+q) += w[2] * w[1] * ((1.-w[0]) * (*fields[2])(xsim-0+1+2,0) + w[0] * (*fields[2])(xsim+0+1+2,0) + (*fields[2])(xsim+1+2,0));
			*(pixbuf_B1+q) += (1.-w[2]) * w[1] * ((1.-w[0]) * (*fields[2])(xsim-0+1,0) + w[0] * (*fields[2])(xsim+0+1,0) + (*fields[2])(xsim+1,0));

			*(pixbuf_B2+q) = (1.-w[2]) * (1.-w[0]) * ((1.-w[1]) * (*fields[2])(xsim-1,1) + w[1] * (*fields[2])(xsim+1,1) + (*fields[2])(xsim,1));
			*(pixbuf_B2+q) += w[2] * (1.-w[0]) * ((1.-w[1]) * (*fields[2])(xsim-1+2,1) + w[1] * (*fields[2])(xsim+1+2,1) + (*fields[2])(xsim+2,1));
			*(pixbuf_B2+q) += w[2] * w[0] * ((1.-w[1]) * (*fields[2])(xsim+0-1+2,1) + w[1] * (*fields[2])(xsim+0+1+2,1) + (*fields[2])(xsim+0+2,1));
			*(pixbuf_B2+q) += (1.-w[2]) * w[0] * ((1.-w[1]) * (*fields[2])(xsim+0-1,1) + w[1] * (*fields[2])(xsim+0+1,1) + (*fields[2])(xsim+0,1));

			*(pixbuf_B3+q) = (1.-w[0]) * (1.-w[1]) * ((1.-w[2]) * (*fields[2])(xsim-2,2) + w[2] * (*fields[2])(xsim+2,2) + (*fields[2])(xsim,2));
			*(pixbuf_B3+q) += w[0] * (1.-w[1]) * ((1.-w[2]) * (*fields[2])(xsim+0-2,2) + w[2] * (*fields[2])(xsim+0+2,2) + (*fields[2])(xsim+0,2));
			*(pixbuf_B3+q) += w[0] * w[1] * ((1.-w[2]) * (*fields[2])(xsim+0+1-2,2) + w[2] * (*fields[2])(xsim+0+1+2,2) + (*fields[2])(xsim+0+1,2));
			*(pixbuf_B3+q) += (1.-w[0]) * w[1] * ((1.-w[2]) * (*fields[2])(xsim+1-2,2) + w[2] * (*fields[2])(xsim+1+2,2) + (*fields[2])(xsim+1,2));

			*(pixbuf_B1+q) /= Real(2) * a2 * numpts;
			*(pixbuf_B2+q) /= Real(2) * a2 * numpts;
			*(pixbuf_B3+q) /= Real(2) * a2 * numpts;
#else
			if (w[0] > 0.5)
			{
				*(pixbuf_B1+q) = (1.5-w[0]) * (1.-w[1]) * ((1.-w[2]) * (*fields[2])(xsim,0) + w[2] * (*fields[2])(xsim+2,0));
				*(pixbuf_B1+q) += (1.5-w[0]) * w[1] * ((1.-w[2]) * (*fields[2])(xsim+1,0) + w[2] * (*fields[2])(xsim+1+2,0));
				*(pixbuf_B1+q) += (w[0]-0.5) * w[1] * ((1.-w[2]) * (*fields[2])(xsim+0+1,0) + w[2] * (*fields[2])(xsim+0+1+2,0));
				*(pixbuf_B1+q) += (w[0]-0.5) * (1.-w[1]) * ((1.-w[2]) * (*fields[2])(xsim+0,0) + w[2] * (*fields[2])(xsim+0+2,0));
			}
			else
			{
				*(pixbuf_B1+q) = (0.5+w[0]) * (1.-w[1]) * ((1.-w[2]) * (*fields[2])(xsim,0) + w[2] * (*fields[2])(xsim+2,0));
				*(pixbuf_B1+q) += (0.5+w[0]) * w[1] * ((1.-w[2]) * (*fields[2])(xsim+1,0) + w[2] * (*fields[2])(xsim+1+2,0));
				*(pixbuf_B1+q) += (0.5-w[0]) * w[1] * ((1.-w[2]) * (*fields[2])(xsim-0+1,0) + w[2] * (*fields[2])(xsim-0+1+2,0));
				*(pixbuf_B1+q) += (0.5-w[0]) * (1.-w[1]) * ((1.-w[2]) * (*fields[2])(xsim-0,0) + w[2] * (*fields[2])(xsim-0+2,0));
			}
			if (w[1] > 0.5)
			{
				*(pixbuf_B2+q) = (1.-w[0]) * (1.5-w[1]) * ((1.-w[2]) * (*fields[2])(xsim,1) + w[2] * (*fields[2])(xsim+2,1));
				*(pixbuf_B2+q) += (1.-w[0]) * (w[1]-0.5) * ((1.-w[2]) * (*fields[2])(xsim+1,1) + w[2] * (*fields[2])(xsim+1+2,1));
				*(pixbuf_B2+q) += w[0] * (w[1]-0.5) * ((1.-w[2]) * (*fields[2])(xsim+0+1,1) + w[2] * (*fields[2])(xsim+0+1+2,1));
				*(pixbuf_B2+q) += w[0] * (1.5-w[1]) * ((1.-w[2]) * (*fields[2])(xsim+0,1) + w[2] * (*fields[2])(xsim+0+2,1));
			}
			else
			{
				*(pixbuf_B2+q) = (1.-w[0]) * (0.5+w[1]) * ((1.-w[2]) * (*fields[2])(xsim,1) + w[2] * (*fields[2])(xsim+2,1));
				*(pixbuf_B2+q) += (1.-w[0]) * (0.5-w[1]) * ((1.-w[2]) * (*fields[2])(xsim-1,1) + w[2] * (*fields[2])(xsim-1+2,1));
				*(pixbuf_B2+q) += w[0] * (0.5-w[1]) * ((1.-w[2]) * (*fields[2])(xsim+0-1,1) + w[2] * (*fields[2])(xsim+0-1+2,1));
				*(pixbuf_B2+q) += w[0] * (0.5+w[1]) * ((1.-w[2]) * (*fields[2])(xsim+0,1) + w[2] * (*fields[2])(xsim+0+2,1));
			}
			if (w[2] > 0.5)
			{
				*(pixbuf_B3+q) = (1.-w[0]) * (1.-w[1]) * ((1.5-w[2]) * (*fields[2])(xsim,2) + (w[2]-0.5) * (*fields[2])(xsim+2,2));
				*(pixbuf_B3+q) += (1.-w[0]) * w[1] * ((1.5-w[2]) * (*fields[2])(xsim+1,2) + (w[2]-0.5) * (*fields[2])(xsim+1+2,2));
				*(pixbuf_B3+q) += w[0] * w[1] * ((1.5-w[2]) * (*fields[2])(xsim+0+1,2) + (w[2]-0.5) * (*fields[2])(xsim+0+1+2,2));
				*(pixbuf_B3+q) += w[0] * (1.-w[1]) * ((1.5-w[2]) * (*fields[2])(xsim+0,2) + (w[2]-0.5) * (*fields[2])(xsim+0+2,2));
			}
			else
			{
				*(pixbuf_B3+q) = (1.-w[0]) * (1.-w[1]) * ((0.5+w[2]) * (*fields[2])(xsim,2) + (0.5-w[2]) * (*fields[2])(xsim-2,2));
				*(pixbuf_B3+q) += (1.-w[0]) * w[1] * ((0.5+w[2]) * (*fields[2])(xsim+1,2) + (0.5-w[2]) * (*fields[2])(xsim+1-2,2));
				*(pixbuf_B3+q) += w[0] * w[1] * ((0.5+w[2]) * (*fields[2])(xsim+0+1,2) + (0.5-w[2]) * (*fields[2])(xsim+0+1-2,2));
				*(pixbuf_B3+q) += w[0] * (1.-w[1]) * ((0.5+w[2]) * (*fields[2])(xsim+0,2) + (0.5-w[2]) * (*fields[2])(xsim+0-2,2));
			}
			*(pixbuf_B1+q) /= a2 * numpts;
			*(pixbuf_B2+q) /= a2 * numpts;
			*(pixbuf_B3+q) /= a2 * numpts;
#endif
		}
		if (outputs & MASK_HIJ)
		{
			*(pixbuf_h11+q) = (1.-w[0]) * (1.-w[1]) * ((1.-w[2]) * (*fields[3])(xsim,0,0) + w[2] * (*fields[3])(xsim+2,0,0));
			*(pixbuf_h11+q) += w[0] * (1.-w[1]) * ((1.-w[2]) * (*fields[3])(xsim+0,0,0) + w[2] * (*fields[3])(xsim+0+2,0,0));
			*(pixbuf_h11+q) += w[0] * w[1] * ((1.-w[2]) * (*fields[3])(xsim+0+1,0,0) + w[2] * (*fields[3])(xsim+0+1+2,0,0));
			*(pixbuf_h11+q) += (1.-w[0]) * w[1] * ((1.-w[2]) * (*fields[3])(xsim+1,0,0) + w[2] * (*fields[3])(xsim+1+2,0,0));

			*(pixbuf_h12+q) = (1.-w[2]) * 0.25 * ((*fields[3])(xsim,0,1) + (1.-w[0]) * ((*fields[3])(xsim-0,0,1) + (1.-w[1]) * (*fields[3])(xsim-0-1,0,1) + w[1] * (*fields[3])(xsim-0+1,0,1)) + w[0] * ((*fields[3])(xsim+0,0,1) + (1.-w[1]) * (*fields[3])(xsim+0-1,0,1) + w[1] * (*fields[3])(xsim+0+1,0,1)) + (1.-w[1]) * (*fields[3])(xsim-1,0,1) + w[1] * (*fields[3])(xsim+1,0,1));
			*(pixbuf_h12+q) += w[2] * 0.25 * ((*fields[3])(xsim+2,0,1) + (1.-w[0]) * ((*fields[3])(xsim-0+2,0,1) + (1.-w[1]) * (*fields[3])(xsim-0-1+2,0,1) + w[1] * (*fields[3])(xsim-0+1+2,0,1)) + w[0] * ((*fields[3])(xsim+0+2,0,1) + (1.-w[1]) * (*fields[3])(xsim+0-1+2,0,1) + w[1] * (*fields[3])(xsim+0+1+2,0,1)) + (1.-w[1]) * (*fields[3])(xsim-1+2,0,1) + w[1] * (*fields[3])(xsim+1+2,0,1));

			*(pixbuf_h13+q) = (1.-w[1]) * 0.25 * ((*fields[3])(xsim,0,2) + (1.-w[0]) * ((*fields[3])(xsim-0,0,2) + (1.-w[2]) * (*fields[3])(xsim-0-2,0,2) + w[2] * (*fields[3])(xsim-0+2,0,2)) + w[0] * ((*fields[3])(xsim+0,0,2) + (1.-w[2]) * (*fields[3])(xsim+0-2,0,2) + w[2] * (*fields[3])(xsim+0+2,0,2)) + (1.-w[2]) * (*fields[3])(xsim-2,0,2) + w[2] * (*fields[3])(xsim+2,0,2));
			*(pixbuf_h13+q) += w[1] * 0.25 * ((*fields[3])(xsim+1,0,2) + (1.-w[0]) * ((*fields[3])(xsim-0+1,0,2) + (1.-w[2]) * (*fields[3])(xsim-0+1-2,0,2) + w[2] * (*fields[3])(xsim-0+1+2,0,2)) + w[0] * ((*fields[3])(xsim+0+1,0,2) + (1.-w[2]) * (*fields[3])(xsim+0+1-2,0,2) + w[2] * (*fields[3])(xsim+0+1+2,0,2)) + (1.-w[2]) * (*fields[3])(xsim+1-2,0,2) + w[2] * (*fields[3])(xsim+1+2,0,2));


			*(pixbuf_h22+q) = (1.-w[0]) * (1.-w[1]) * ((1.-w[2]) * (*fields[3])(xsim,1,1) + w[2] * (*fields[3])(xsim+2,1,1));
			*(pixbuf_h22+q) += w[0] * (1.-w[1]) * ((1.-w[2]) * (*fields[3])(xsim+0,1,1) + w[2] * (*fields[3])(xsim+0+2,1,1));
			*(pixbuf_h22+q) += w[0] * w[1] * ((1.-w[2]) * (*fields[3])(xsim+0+1,1,1) + w[2] * (*fields[3])(xsim+0+1+2,1,1));
			*(pixbuf_h22+q) += (1.-w[0]) * w[1] * ((1.-w[2]) * (*fields[3])(xsim+1,1,1) + w[2] * (*fields[3])(xsim+1+2,1,1));

			*(pixbuf_h23+q) = (1.-w[0]) * 0.25 * ((*fields[3])(xsim,1,2) + (1.-w[1]) * ((*fields[3])(xsim-1,1,2) + (1.-w[2]) * (*fields[3])(xsim-1-2,1,2) + w[2] * (*fields[3])(xsim-1+2,1,2)) + w[1] * ((*fields[3])(xsim+1,1,2) + (1.-w[2]) * (*fields[3])(xsim+1-2,1,2) + w[2] * (*fields[3])(xsim+1+2,1,2)) + (1.-w[2]) * (*fields[3])(xsim-2,1,2) + w[2] * (*fields[3])(xsim+2,1,2));
			*(pixbuf_h23+q) += w[0] * 0.25 * ((*fields[3])(xsim+0,1,2) + (1.-w[1]) * ((*fields[3])(xsim+0-1,1,2) + (1.-w[2]) * (*fields[3])(xsim+0-1-2,1,2) + w[2] * (*fields[3])(xsim+0-1+2,1,2)) + w[1] * ((*fields[3])(xsim+0+1,1,2) + (1.-w[2]) * (*fields[3])(xsim+0+1-2,1,2) + w[2] * (*fields[3])(xsim+0+1+2,1,2)) + (1.-w[2]) * (*fields[3])(xsim+0-2,1,2) + w[2] * (*fields[3])(xsim+0+2,1,2));
		}
	}
	else
	{
		if (outputs & MASK_PHI)
			*(pixbuf_phi+q) = 0;
		if (outputs & MASK_CHI)
			*(pixbuf_chi+q) = 0;
		if (outputs & MASK_B)
		{
			*(pixbuf_B1+q) = 0;
			*(pixbuf_B2+q) = 0;
			*(pixbuf_B3+q) = 0;
		}
		if (outputs & MASK_HIJ)
		{
			*(pixbuf_h11+q) = 0;
			*(pixbuf_h12+q) = 0;
			*(pixbuf_h13+q) = 0;
			*(pixbuf_h22+q) = 0;
			*(pixbuf_h23+q) = 0;
		}
	}
}

inline void healpix_cuda_check(cudaError_t success, const char * context);

void create_packmap(int64_t * packmap, int64_t pix, int pixbatch_size, int64_t nside, int64_t npix)
{
	int64_t * temp = (int64_t *) alloca(pixbatch_size * sizeof(int64_t));
	int q = 0;

#pragma omp parallel for
	for (int i = 0; i < pixbatch_size; i++)
	{
		nest2ring64(nside, pix+i, temp+i);
	}

	for (int i = 0; i < pixbatch_size; i++)
	{
		if (temp[i] < npix)
		{
			temp[i] = q;
			q++;
		}
		else
			temp[i] = -1;

		pix++;
	}

	healpix_cuda_check(cudaMemcpy(packmap, temp, pixbatch_size * sizeof(int64_t), cudaMemcpyDefault), "pixel packmap copy");
}

inline void healpix_cuda_check(cudaError_t success, const char * context)
{
	if (success != cudaSuccess)
	{
		cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " CUDA error in writeLightcones (" << context << "): " << cudaGetErrorString(success) << endl;
		throw std::runtime_error("CUDA error");
	}
}

inline void healpix_cuda_malloc(Real ** buffer, int64_t count)
{
	healpix_cuda_check(cudaMalloc((void **) buffer, sizeof(Real) * count), "pixel buffer allocation");
}

inline void healpix_cuda_grow(Real ** buffer, int64_t old_count, int64_t new_count)
{
	Real * new_buffer = NULL;

	healpix_cuda_check(cudaMalloc((void **) &new_buffer, sizeof(Real) * new_count), "pixel buffer reallocation");
	if (*buffer != NULL && old_count > 0)
		healpix_cuda_check(cudaMemcpy(new_buffer, *buffer, sizeof(Real) * old_count, cudaMemcpyDeviceToDevice), "pixel buffer grow copy");
	if (*buffer != NULL)
		healpix_cuda_check(cudaFree(*buffer), "pixel buffer grow free");

	*buffer = new_buffer;
}

inline void healpix_pack3(Real * dst, Real * src0, int64_t size0, Real * src1, int64_t size1, Real * src2, int64_t size2)
{
	if (size0 > 0)
		healpix_cuda_check(cudaMemcpy(dst, src0, size0 * sizeof(Real), cudaMemcpyDeviceToDevice), "pixel buffer pack");
	if (size1 > 0)
		healpix_cuda_check(cudaMemcpy(dst + size0, src1, size1 * sizeof(Real), cudaMemcpyDeviceToDevice), "pixel buffer pack");
	if (size2 > 0)
		healpix_cuda_check(cudaMemcpy(dst + size0 + size1, src2, size2 * sizeof(Real), cudaMemcpyDeviceToDevice), "pixel buffer pack");
}

__global__ void healpix_add_packed3_kernel(Real * dst0, int64_t size0, Real * dst1, int64_t size1, Real * dst2, int64_t size2, const Real * src)
{
	int64_t q = blockIdx.x * blockDim.x + threadIdx.x;
	int64_t total = size0 + size1 + size2;

	if (q >= total)
		return;

	if (q < size0)
		dst0[q] += src[q];
	else if (q < size0 + size1)
		dst1[q - size0] += src[q];
	else
		dst2[q - size0 - size1] += src[q];
}

__global__ void healpix_add_kernel(Real * dst, const Real * src, int64_t size)
{
	int64_t q = blockIdx.x * blockDim.x + threadIdx.x;

	if (q < size)
		dst[q] += src[q];
}

inline void healpix_add_packed3(Real * dst0, int64_t size0, Real * dst1, int64_t size1, Real * dst2, int64_t size2, Real * src)
{
	int64_t total = size0 + size1 + size2;

	if (total <= 0)
		return;

	healpix_add_packed3_kernel<<<(total + 255) / 256, 256>>>(dst0, size0, dst1, size1, dst2, size2, src);
	healpix_cuda_check(cudaGetLastError(), "pixel buffer accumulation launch");
}

inline void healpix_add(Real * dst, Real * src, int64_t size)
{
	if (size <= 0)
		return;

	healpix_add_kernel<<<(size + 255) / 256, 256>>>(dst, src, size);
	healpix_cuda_check(cudaGetLastError(), "pixel buffer accumulation launch");
}

inline void healpix_sync_if(bool & required_pending, bool & other_pending, const char * context)
{
	if (required_pending)
	{
		healpix_cuda_check(cudaStreamSynchronize(0), context);
		required_pending = false;
		other_pending = false;
	}
}

inline void healpix_sync_any(bool & pending_a, bool & pending_b, const char * context)
{
	if (pending_a || pending_b)
	{
		healpix_cuda_check(cudaStreamSynchronize(0), context);
		pending_a = false;
		pending_b = false;
	}
}

struct HealpixShellDesc
{
	int shell;
	healpix_header hdr;
	int pixbatch_size[3];
	int pixbatch_delim[3];
	int64_t pixbuf_offset[9];
	int pixbuf_size[9];
	vector<int> pixbatch_id;
	vector<int64_t> pixbatch_offset;
	vector<int> sender_proc;
	bool writes_on_rank;
	int write_rank_offset;
	int write_rank_count;
	int write_batch_begin;
	int write_batch_count;
	int64_t outbuf_base;
	int64_t write_file_offset;
	int64_t local_bytes;
};

struct HealpixTransferSegment
{
	int rank;
	int64_t pixbuf_offset;
	int64_t outbuf_offset;
	int count;
};

inline int healpix_batch_type(const HealpixShellDesc & desc, int batch)
{
	if (batch < desc.pixbatch_delim[0])
		return 0;
	if (batch < desc.pixbatch_delim[1])
		return 1;
	return 2;
}

inline int64_t healpix_batch_output_offset(const HealpixShellDesc & desc, int batch)
{
	int64_t offset = 268;

	if (batch <= desc.pixbatch_delim[0])
		return offset + (int64_t) batch * desc.pixbatch_size[0] * desc.hdr.precision;

	offset += (int64_t) desc.pixbatch_delim[0] * desc.pixbatch_size[0] * desc.hdr.precision;
	if (batch <= desc.pixbatch_delim[1])
		return offset + (int64_t) (batch - desc.pixbatch_delim[0]) * desc.pixbatch_size[1] * desc.hdr.precision;

	offset += (int64_t) (desc.pixbatch_delim[1] - desc.pixbatch_delim[0]) * desc.pixbatch_size[1] * desc.hdr.precision;
	return offset + (int64_t) (batch - desc.pixbatch_delim[1]) * desc.pixbatch_size[2] * desc.hdr.precision;
}

inline int healpix_shell_group_start(int shell, int shell_inner, int shell_outer)
{
	int shell_count = shell_outer + 1 - shell_inner;
	return ((shell - shell_inner) * parallel.size() + shell_count - 1) / shell_count;
}

inline int healpix_shell_group_end(int shell, int shell_inner, int shell_outer)
{
	int shell_count = shell_outer + 1 - shell_inner;
	return ((shell + 1 - shell_inner) * parallel.size() + shell_count - 1) / shell_count;
}

inline int healpix_writer_rank(const HealpixShellDesc & desc, int shell_inner, int shell_outer, int batch)
{
	int shell_count = shell_outer + 1 - shell_inner;

	if (shell_count > parallel.size())
		return ((desc.shell - shell_inner) * parallel.size()) / shell_count;

	int group_start = healpix_shell_group_start(desc.shell, shell_inner, shell_outer);
	int group_size = healpix_shell_group_end(desc.shell, shell_inner, shell_outer) - group_start;
	int batch_stride = desc.pixbatch_delim[2] / group_size;

	if (batch_stride > 0 && desc.pixbatch_delim[2] >= group_size && batch / batch_stride < group_size)
		return group_start + batch / batch_stride;

	return group_start + group_size - 1;
}

inline int64_t healpix_find_pixbatch_offset(const HealpixShellDesc & desc, int batch)
{
	for (int i = 0; i < (int) desc.pixbatch_id.size(); i++)
	{
		if (desc.pixbatch_id[i] == batch)
			return desc.pixbatch_offset[i];
	}

	cerr << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " pixel batch index mismatch! expecting " << batch << " but ID list does not contain it!" << endl;
	exit(-99);
}

#endif


//////////////////////////
// writeLightcones
//////////////////////////
// Description:
//   output of light cones
//
// Arguments:
//   sim            simulation metadata structure
//   cosmo          cosmological parameter structure
//   fourpiG        4 pi G (in code units)
//   a              scale factor
//   tau            conformal time
//   dtau           conformal time step
//   dtau_old       conformal time step of previous cycle
//   maxvel         maximum cdm velocity
//   cycle          current simulation cycle
//   h5filename     base name for HDF5 output file
//   pcls_cdm       pointer to particle handler for CDM
//   pcls_b         pointer to particle handler for baryons
//   pcls_ncdm      array of particle handlers for
//                  non-cold DM (may be set to NULL)
//   phi            pointer to allocated field
//   chi            pointer to allocated field
//   Bi             pointer to allocated field
//   Sij            pointer to allocated field
//   BiFT           pointer to allocated field
//   SijFT          pointer to allocated field
//   plan_Bi        pointer to FFT planner
//   plan_Sij       pointer to FFT planner
//   done_hij       reference to tensor projection flag
//   IDbacklog      IDs of particles written in previous cycle
//
// Returns:
//
//////////////////////////

void writeLightcones(metadata & sim, cosmology & cosmo, const double fourpiG, const double a, const double tau, const double dtau, const double dtau_old, const double maxvel, const int cycle, string h5filename, perfParticles_gevolution<part_simple,part_simple_info> * pcls_cdm, perfParticles_gevolution<part_simple,part_simple_info> * pcls_b, Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> * pcls_ncdm, Field<Real> * phi, Field<Real> * chi, Field<Real> * Bi, Field<Real> * Sij, Field<Cplx> * BiFT, Field<Cplx> * SijFT, PlanFFT<Cplx> * plan_Bi, PlanFFT<Cplx> * plan_Sij, int & done_hij, set<long> ** IDbacklog)
{
	int i, j, n, p;
	double d;
	double vertex[MAX_INTERSECTS][3];
	double domain[6];
	//double pos[3];
	double s[2];
	char filename[2*PARAM_MAX_LENGTH+24];
	char buffer[268];
	FILE * outfile = NULL;
	gadget2_header hdr;
	vector<long> ** IDprelog;
	int IDlog_multiplicity = 1;
	int IDlog_sizes[9];
	int IDlog_sizes_recv1[6];
	int IDlog_sizes_send1[3];
	int IDlog_sizes_recv0[10];
	int IDlog_sizes_send0[5];
	long * IDcombuf1 = NULL;
	long * IDcombuf2 = NULL;
	long * IDcombuf3 = NULL;
	long * IDcombuf4 = NULL;
	Site xsim;
#ifdef HAVE_HEALPIX
	int done_B = 0;
	int64_t pix, q;
	int pixbatch_type;
	int commdir[2];
	Real * pixbuf[LIGHTCONE_MAX_FIELDS][9];
	Real * commbuf;
	int pixbuf_size[9];
	int pixbuf_reserve[9];
	int64_t bytes, bytes2, offset2 = 0;
	vector<MPI_Offset> offset;
	vector<HealpixShellDesc> shell_desc;
	char ** outbuf = new char*[LIGHTCONE_MAX_FIELDS];
	healpix_header maphdr;
	double R[3][3];
	double w[3];
	double temp;
	int base_pos[3];
	int shell, shell_inner, shell_outer, shell_write;
	uint32_t blocksize;
	MPI_File mapfile;
	MPI_Status status;
	int io_group_size;

	for (j = 0; j < 9*LIGHTCONE_MAX_FIELDS; j++)
		pixbuf[j/9][j%9] = NULL;

	for (j = 0; j < LIGHTCONE_MAX_FIELDS; j++)
		outbuf[j] = NULL;

	Field<Real> * fields[4] = {phi, chi, Bi, Sij};
	int64_t * packmap[2] = {nullptr, nullptr};
	int kernels_running = 0;
	bool healpix_send_workspace_fallback_warning = false;
	bool healpix_comm_workspace_fallback_warning = false;
#endif

	done_hij = 0;

	IDprelog = new vector<long> * [sim.num_IDlogs];

	domain[0] = -0.5;
	domain[1] = phi->lattice().coordSkip()[1] - 0.5;
	domain[2] = phi->lattice().coordSkip()[0] - 0.5;
	for (i = 0; i < 3; i++)
		domain[i+3] = domain[i] + phi->lattice().sizeLocal(i) + 1.;

	if ((domain[4]-domain[1]-1.0 > dtau * 2.0 * LIGHTCONE_IDCHECK_ZONE * sim.numpts) && (domain[5]-domain[2]-1.0 > dtau * 2.0 * LIGHTCONE_IDCHECK_ZONE * sim.numpts))
		IDlog_multiplicity = 9;

	for (i = 0; i < sim.num_IDlogs; i++)
		IDprelog[i] = new vector<long> [IDlog_multiplicity * MAX_PCL_SPECIES];

	for (i = 0; i < 6; i++)
		domain[i] /= (double) sim.numpts;

	for (i = 0; i < sim.num_lightcone; i++)
	{
		if (parallel.isRoot())
		{
			if (sim.num_lightcone > 1)
				sprintf(filename, "%s%s%d_info.dat", sim.output_path, sim.basename_lightcone, i);
			else
				sprintf(filename, "%s%s_info.dat", sim.output_path, sim.basename_lightcone);

			outfile = fopen(filename, "a");
			if (outfile == NULL)
			{
				cout << " error opening file for lightcone info!" << endl;
			}
			else if (cycle == 0)
			{
				if (sim.num_lightcone > 1)
					fprintf(outfile, "# information file for lightcone %d\n# geometric parameters:\n# vertex = (%f, %f, %f) Mpc/h\n# redshift = %f\n# distance = (%f - %f) Mpc/h\n# opening half-angle = %f degrees\n# direction = (%f, %f, %f)\n# cycle   tau/boxsize    a              pcl_inner        pcl_outer        metric_inner     metric_outer\n", i, sim.lightcone[i].vertex[0]*sim.boxsize, sim.lightcone[i].vertex[1]*sim.boxsize, sim.lightcone[i].vertex[2]*sim.boxsize, sim.lightcone[i].z, sim.lightcone[i].distance[0]*sim.boxsize, sim.lightcone[i].distance[1]*sim.boxsize, (sim.lightcone[i].opening > -1.) ? acos(sim.lightcone[i].opening) * 180. / M_PI : 180., sim.lightcone[i].direction[0], sim.lightcone[i].direction[1], sim.lightcone[i].direction[2]);
				else
					fprintf(outfile, "# information file for lightcone\n# geometric parameters:\n# vertex = (%f, %f, %f) Mpc/h\n# redshift = %f\n# distance = (%f - %f) Mpc/h\n# opening half-angle = %f degrees\n# direction = (%f, %f, %f)\n# cycle   tau/boxsize    a              pcl_inner        pcl_outer        metric_inner     metric_outer\n", sim.lightcone[i].vertex[0]*sim.boxsize, sim.lightcone[i].vertex[1]*sim.boxsize, sim.lightcone[i].vertex[2]*sim.boxsize, sim.lightcone[i].z, sim.lightcone[i].distance[0]*sim.boxsize, sim.lightcone[i].distance[1]*sim.boxsize, (sim.lightcone[i].opening > -1.) ? acos(sim.lightcone[i].opening) * 180. / M_PI : 180., sim.lightcone[i].direction[0], sim.lightcone[i].direction[1], sim.lightcone[i].direction[2]);
			}
		}

		d = particleHorizon(1. / (1. + sim.lightcone[i].z), fourpiG, cosmo);

		s[0] = d - tau - 0.5 * sim.covering[i] * dtau;
		s[1] = d - tau + 0.5 * sim.covering[i] * dtau_old;

#ifdef HAVE_HEALPIX
		shell_inner = (s[0] > sim.lightcone[i].distance[1]) ? ceil(s[0] * sim.numpts * sim.shellfactor[i]) : ceil(sim.lightcone[i].distance[1] * sim.numpts * sim.shellfactor[i]);
		shell_outer = (sim.lightcone[i].distance[0] > s[1]) ? floor(s[1] * sim.numpts * sim.shellfactor[i]) : (ceil(sim.lightcone[i].distance[0] * sim.numpts * sim.shellfactor[i])-1);
		if (shell_outer < shell_inner && s[1] > 0) shell_outer = shell_inner;

		maphdr.precision = sizeof(Real);
		maphdr.Ngrid = sim.numpts;
		maphdr.direction[0] = sim.lightcone[i].direction[0];
		maphdr.direction[1] = sim.lightcone[i].direction[1];
		maphdr.direction[2] = sim.lightcone[i].direction[2];
		maphdr.boxsize = sim.boxsize;
		memset((void *) maphdr.fill, 0, 256 - 5 * 4 - 5 * 8);

		xsim.initialize(phi->lattice());

		if (sim.lightcone[i].direction[0] == 0 && sim.lightcone[i].direction[1] == 0)
		{
			R[0][0] = sim.lightcone[i].direction[2];
			R[0][1] = 0;
			R[0][2] = 0;
			R[1][0] = 0;
			R[1][1] = 1;
			R[1][2] = 0;
			R[2][0] = 0;
			R[2][1] = 0;
			R[2][2] = sim.lightcone[i].direction[2];
		}
		else
		{
			temp = atan2(sim.lightcone[i].direction[1], sim.lightcone[i].direction[0]);
			R[0][0] = cos(temp) * sim.lightcone[i].direction[2];
			R[0][1] = -sin(temp);
			R[0][2] = sim.lightcone[i].direction[0];
			R[1][0] = sin(temp) * sim.lightcone[i].direction[2];
			R[1][1] = cos(temp);
			R[1][2] = sim.lightcone[i].direction[1];
			R[2][0] = -sqrt(1. - sim.lightcone[i].direction[2]*sim.lightcone[i].direction[2]);
			R[2][1] = 0;
			R[2][2] = sim.lightcone[i].direction[2];
		}
#endif

		if (sim.lightcone[i].distance[0] > s[0] && sim.lightcone[i].distance[1] <= s[1] && s[1] > 0.)
		{
			if (parallel.isRoot() && outfile != NULL)
			{
				fprintf(outfile, "%6d   %e   %e   %2.12f   %2.12f   %2.12f   %2.12f\n", cycle, tau, a, d - tau - 0.5 * dtau, d - tau + 0.5 * dtau_old, s[0], s[1]);
				fclose(outfile);

				if (sim.num_lightcone > 1)
					sprintf(filename, "%s%s%d_info.bin", sim.output_path, sim.basename_lightcone, i);
				else
					sprintf(filename, "%s%s_info.bin", sim.output_path, sim.basename_lightcone);

				outfile = fopen(filename, "a");
				if (outfile == NULL)
				{
					cout << " error opening file for lightcone info!" << endl;
				}
				else
				{
					((double *) buffer)[0] = tau;
					((double *) buffer)[1] = a;
					((double *) buffer)[2] = d - tau - 0.5 * dtau;
					((double *) buffer)[3] = d - tau + 0.5 * dtau_old;

					fwrite((const void *) &cycle, sizeof(int), 1, outfile);
					fwrite((const void *) buffer, sizeof(double), 4, outfile);
					fwrite((const void *) s, sizeof(double), 2, outfile);

					fclose(outfile);
				}
			}

#ifdef HAVE_HEALPIX
			nvtxRangePushA("HEALPix output");
			bytes = 0;
			bytes2 = 0;

			for (j = 0; j < 9; j++)
				pixbuf_reserve[j] = PIXBUFFER;

			if (sim.out_lightcone[i] & MASK_PHI)
			{
				for (j = 0; j < 9; j++)
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_PHI_OFFSET][j], PIXBUFFER);
			}

			if (sim.out_lightcone[i] & MASK_CHI)
			{
				for (j = 0; j < 9; j++)
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_CHI_OFFSET][j], PIXBUFFER);
			}

			if (sim.out_lightcone[i] & MASK_B)
			{
				for (j = 0; j < 9; j++)
				{
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_B_OFFSET][j], PIXBUFFER);
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_B_OFFSET+1][j], PIXBUFFER);
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_B_OFFSET+2][j], PIXBUFFER);
				}
			}

			if (sim.out_lightcone[i] & MASK_HIJ)
			{
				for (j = 0; j < 9; j++)
				{
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_HIJ_OFFSET][j], PIXBUFFER);
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_HIJ_OFFSET+1][j], PIXBUFFER);
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_HIJ_OFFSET+2][j], PIXBUFFER);
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_HIJ_OFFSET+3][j], PIXBUFFER);
					healpix_cuda_malloc(&pixbuf[LIGHTCONE_HIJ_OFFSET+4][j], PIXBUFFER);
				}
			}

			if (sim.gr_flag == 0 && sim.out_lightcone[i] & MASK_B && done_B == 0)
			{
				plan_Bi->execute(FFT_BACKWARD);
				Bi->updateHalo();
				done_B = 1;
			}

			if (sim.out_lightcone[i] & MASK_HIJ && done_hij == 0)
			{
				projectFTtensor(*SijFT, *SijFT);
				plan_Sij->execute(FFT_BACKWARD);
				Sij->updateHalo();
				done_hij = 1;
			}

			if ((shell_outer + 1 - shell_inner) > parallel.size())
			{
				shell_write = ((shell_outer + 1 - shell_inner) * parallel.rank() + parallel.size() - 1) / parallel.size();
				io_group_size = 0;
			}
			else
			{
				shell_write = ((shell_outer + 1 - shell_inner) * parallel.rank()) / parallel.size();
				io_group_size = (((shell_write+1) * parallel.size() + shell_outer - shell_inner) / (shell_outer + 1 - shell_inner)) - ((shell_write * parallel.size() + shell_outer - shell_inner) / (shell_outer + 1 - shell_inner));
			}

			for (int shell_chunk_begin = shell_inner; shell_chunk_begin <= shell_outer; shell_chunk_begin += HEALPIX_SHELL_CHUNK)
			{
				int shell_chunk_end = min(shell_outer, shell_chunk_begin + HEALPIX_SHELL_CHUNK - 1);
				vector<int64_t *> packmaps_to_free;

				shell_desc.clear();
				for (j = 0; j < 9; j++)
					pixbuf_size[j] = 0;

				nvtxRangePushA("pixelisation");

				for (shell = shell_chunk_begin; shell <= shell_chunk_end; shell++)
				{
					HealpixShellDesc desc;

					desc.shell = shell;
					desc.writes_on_rank = false;
					desc.write_rank_offset = 0;
					desc.write_rank_count = 0;
					desc.write_batch_begin = 0;
					desc.write_batch_count = 0;
					desc.outbuf_base = bytes2;
					desc.write_file_offset = 0;
					desc.local_bytes = 0;
					for (j = 0; j < 9; j++)
					{
						desc.pixbuf_offset[j] = pixbuf_size[j];
						desc.pixbuf_size[j] = 0;
					}

					maphdr.distance = (double) shell / (double) sim.numpts / sim.shellfactor[i];

					for (maphdr.Nside = sim.Nside[i][0]; maphdr.Nside < sim.Nside[i][1]; maphdr.Nside *= 2)
					{
						if (12. * maphdr.Nside * maphdr.Nside > sim.pixelfactor[i] * 4. * M_PI * maphdr.distance * maphdr.distance * sim.numpts * sim.numpts) break;
					}

					for (maphdr.Nside_ring = 2; 2.137937882409166 * sim.numpts * maphdr.distance / maphdr.Nside_ring > phi->lattice().sizeLocal(1) && 2.137937882409166 * sim.numpts * maphdr.distance / maphdr.Nside_ring > phi->lattice().sizeLocal(2) && maphdr.Nside_ring < maphdr.Nside; maphdr.Nside_ring *= 2);

					if (sim.lightcone[i].opening > 2./3.)
					{
						p = 1 + (int) floor(maphdr.Nside * sqrt(3. - 3. * sim.lightcone[i].opening));
						maphdr.Npix = 2 * p * (p+1);
					}
					else if (sim.lightcone[i].opening > -2./3.)
					{
						p = 1 + (int) floor(maphdr.Nside * (2. - 1.5 * sim.lightcone[i].opening));
						maphdr.Npix = 2 * maphdr.Nside * (maphdr.Nside+1) + (p-maphdr.Nside) * 4 * maphdr.Nside;
					}
					else if (sim.lightcone[i].opening > -1.)
					{
						p = (int) floor(maphdr.Nside * sqrt(3. + 3. * sim.lightcone[i].opening));
						maphdr.Npix = 12 * maphdr.Nside * maphdr.Nside - 2 * p * (p+1);
						p = 4 * maphdr.Nside - 1 - p;
					}
					else
					{
						maphdr.Npix = 12 * maphdr.Nside * maphdr.Nside;
						p = 4 * maphdr.Nside - 1;
					}

					desc.hdr = maphdr;
					desc.pixbatch_size[0] = maphdr.Nside / maphdr.Nside_ring;
					desc.pixbatch_delim[1] = p / desc.pixbatch_size[0];
					desc.pixbatch_delim[0] = (desc.pixbatch_delim[1] > 0) ? desc.pixbatch_delim[1]-1 : 0;
					desc.pixbatch_delim[2] = desc.pixbatch_delim[1]+1;
					desc.pixbatch_size[1] = (desc.pixbatch_size[0] * (desc.pixbatch_size[0]+1) + (2*desc.pixbatch_size[0] - 1 - p%desc.pixbatch_size[0]) * (p%desc.pixbatch_size[0])) / 2;
					desc.pixbatch_size[2] = (((p%desc.pixbatch_size[0] + 1) * (p%desc.pixbatch_size[0])) / 2);
					desc.pixbatch_size[0] *= desc.pixbatch_size[0];
					for (p = 0; p < 3; p++)
					{
						if (desc.pixbatch_delim[p] <= (int) maphdr.Nside_ring)
							desc.pixbatch_delim[p] = 2 * desc.pixbatch_delim[p] * (desc.pixbatch_delim[p]+1);
						else if (desc.pixbatch_delim[p] <= (int) (3 * maphdr.Nside_ring))
							desc.pixbatch_delim[p] = 2 * maphdr.Nside_ring * (maphdr.Nside_ring+1) + (desc.pixbatch_delim[p]-maphdr.Nside_ring) * 4 * maphdr.Nside_ring;
						else if (desc.pixbatch_delim[p] < (int) (4 * maphdr.Nside_ring))
							desc.pixbatch_delim[p] = 12 * maphdr.Nside_ring * maphdr.Nside_ring - 2 * (4 * maphdr.Nside_ring - 1 - desc.pixbatch_delim[p]) * (4 * maphdr.Nside_ring - desc.pixbatch_delim[p]);
						else
							desc.pixbatch_delim[p] = 12 * maphdr.Nside_ring * maphdr.Nside_ring;
					}

					if (desc.pixbatch_size[1] == desc.pixbatch_size[0])
						desc.pixbatch_delim[0] = desc.pixbatch_delim[1];

					if (io_group_size == 0)
					{
						desc.writes_on_rank = (parallel.rank() == ((shell - shell_inner) * parallel.size()) / (shell_outer + 1 - shell_inner));
						desc.write_rank_count = 1;
						desc.write_batch_begin = 0;
						desc.write_batch_count = desc.pixbatch_delim[2];
						desc.write_file_offset = 0;
						desc.local_bytes = desc.hdr.Npix * desc.hdr.precision + 272;
					}
					else
					{
						int group_start = healpix_shell_group_start(shell, shell_inner, shell_outer);
						int group_end = healpix_shell_group_end(shell, shell_inner, shell_outer);
						int group_size = group_end - group_start;

						desc.writes_on_rank = (parallel.rank() >= group_start && parallel.rank() < group_end);
						desc.write_rank_count = group_size;
						if (desc.writes_on_rank)
						{
							desc.write_rank_offset = parallel.rank() - group_start;
							q = desc.pixbatch_delim[2] / group_size;
							desc.write_batch_begin = desc.write_rank_offset * q;
							desc.write_batch_count = ((desc.write_rank_offset == group_size - 1) ? desc.pixbatch_delim[2] : desc.write_batch_begin + q) - desc.write_batch_begin;
							desc.write_file_offset = (desc.write_rank_offset == 0) ? 0 : healpix_batch_output_offset(desc, desc.write_batch_begin);
							desc.local_bytes = healpix_batch_output_offset(desc, desc.write_batch_begin + desc.write_batch_count) - desc.write_file_offset;
							if (desc.write_rank_offset == group_size - 1)
								desc.local_bytes += 4;
						}
					}

					if (desc.writes_on_rank)
					{
						int64_t new_bytes2 = (io_group_size == 0) ? bytes2 + desc.local_bytes : desc.local_bytes;

						for (j = 0; j < LIGHTCONE_MAX_FIELDS; j++)
						{
							if (pixbuf[j][0] != NULL)
							{
								if (io_group_size == 0)
								{
									if (bytes2 == 0)
										outbuf[j] = (char *) malloc(new_bytes2);
									else
										outbuf[j] = (char *) realloc((void *) outbuf[j], new_bytes2);
								}
								else if (outbuf[j] == NULL && new_bytes2 > 0)
									outbuf[j] = (char *) malloc(new_bytes2);

								if (outbuf[j] == NULL && new_bytes2 > 0)
								{
									cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " unable to allocate " << new_bytes2 << " bytes of memory for pixelisation!" << endl;
									parallel.abortForce();
								}

								if (desc.write_file_offset == 0)
								{
									blocksize = 256;
									memcpy((void *) (outbuf[j] + desc.outbuf_base), (void *) &blocksize, 4);
									memcpy((void *) (outbuf[j] + desc.outbuf_base + 4), (void *) &desc.hdr, 256);
									memcpy((void *) (outbuf[j] + desc.outbuf_base + 260), (void *) &blocksize, 4);
									blocksize = desc.hdr.precision * desc.hdr.Npix;
									memcpy((void *) (outbuf[j] + desc.outbuf_base + 264), (void *) &blocksize, 4);
								}

								if (desc.write_batch_begin + desc.write_batch_count == desc.pixbatch_delim[2])
								{
									blocksize = desc.hdr.precision * desc.hdr.Npix;
									memcpy((void *) (outbuf[j] + desc.outbuf_base + desc.local_bytes - 4), (void *) &blocksize, 4);
								}
							}
						}

						if (io_group_size == 0)
							bytes2 = new_bytes2;
						else
						{
							bytes2 = desc.local_bytes;
							offset2 = desc.write_file_offset;
						}
					}

					offset.push_back(bytes);
					bytes += desc.hdr.Npix * desc.hdr.precision + 272;

					for (p = 0; p < desc.pixbatch_delim[2]; p++)
					{
						pix2vec_ring64(maphdr.Nside_ring, p, w);

						base_pos[1] = (int) floor((maphdr.distance * (R[1][0] * w[0] + R[1][1] * w[1] + R[1][2] * w[2]) + sim.lightcone[i].vertex[1]) * sim.numpts) % sim.numpts;
						if (base_pos[1] < 0) base_pos[1] += sim.numpts;

						commdir[1] = phi->lattice().getRankDim1(base_pos[1]);
						j = commdir[1]*parallel.grid_size()[0];
						commdir[1] -= parallel.grid_rank()[1];

						if (commdir[1] < -1) commdir[1] += parallel.grid_size()[1];
						else if (commdir[1] > 1) commdir[1] -= parallel.grid_size()[1];

						base_pos[2] = (int) floor((maphdr.distance * (R[2][0] * w[0] + R[2][1] * w[1] + R[2][2] * w[2]) + sim.lightcone[i].vertex[2]) * sim.numpts) % sim.numpts;
						if (base_pos[2] < 0) base_pos[2] += sim.numpts;

						commdir[0] = phi->lattice().getRankDim0(base_pos[2]);
						j += commdir[0];
						commdir[0] -= parallel.grid_rank()[0];

						if (commdir[0] < -1) commdir[0] += parallel.grid_size()[0];
						else if (commdir[0] > 1) commdir[0] -= parallel.grid_size()[0];

						if (desc.writes_on_rank && p >= desc.write_batch_begin && p < desc.write_batch_begin + desc.write_batch_count)
							desc.sender_proc.push_back(j);

						if (commdir[0] * commdir[0] > 1 || commdir[1] * commdir[1] > 1) continue;

						ring2nest64(maphdr.Nside_ring, p, &pix);
						pix *= desc.pixbatch_size[0];

						pixbatch_type = healpix_batch_type(desc, p);
						j = 3*commdir[0]+commdir[1]+4;

						if (pixbuf_size[j] + desc.pixbatch_size[pixbatch_type] > pixbuf_reserve[j])
						{
							nvtxRangePushA("pixel buffer reallocation");
							if (kernels_running & (1 << j))
							{
								auto success = cudaDeviceSynchronize();

								if (success != cudaSuccess)
								{
									cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " CUDA error in writeLightcones: " << cudaGetErrorString(success) << endl;
									throw std::runtime_error("CUDA error");
								}

								kernels_running = 0;
							}

							int old_reserve = pixbuf_reserve[j];

							do
							{
								pixbuf_reserve[j] += PIXBUFFER;
							}
							while (pixbuf_size[j] + desc.pixbatch_size[pixbatch_type] > pixbuf_reserve[j]);

							for (int f = 0; f < LIGHTCONE_MAX_FIELDS; f++)
							{
								if (pixbuf[f][j] != NULL)
									healpix_cuda_grow(&pixbuf[f][j], old_reserve, pixbuf_reserve[j]);
							}
							nvtxRangePop();
						}

						if (pixbatch_type)
						{
							if (packmap[pixbatch_type-1] == nullptr)
							{
								nvtxRangePushA("create pixel packmap");
								healpix_cuda_check(cudaMalloc((void **) &packmap[pixbatch_type-1], desc.pixbatch_size[0] * sizeof(int64_t)), "pixel packmap allocation");
								create_packmap(packmap[pixbatch_type-1], pix, desc.pixbatch_size[0], maphdr.Nside, maphdr.Npix);
								nvtxRangePop();
							}

							project_metric_to_healpix_batch<<<(desc.pixbatch_size[0] + 127) / 128, 128>>>(pixbuf[LIGHTCONE_PHI_OFFSET][j]+pixbuf_size[j], pixbuf[LIGHTCONE_CHI_OFFSET][j]+pixbuf_size[j], pixbuf[LIGHTCONE_B_OFFSET][j]+pixbuf_size[j], pixbuf[LIGHTCONE_B_OFFSET+1][j]+pixbuf_size[j], pixbuf[LIGHTCONE_B_OFFSET+2][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET+1][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET+2][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET+3][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET+4][j]+pixbuf_size[j], maphdr.Nside, pix, a*a, maphdr.distance, sim.lightcone[i].vertex, R, sim.numpts, fields, sim.out_lightcone[i], desc.pixbatch_size[0], packmap[pixbatch_type-1]);
						}
						else
						{
							project_metric_to_healpix_batch<<<(desc.pixbatch_size[0] + 127) / 128, 128>>>(pixbuf[LIGHTCONE_PHI_OFFSET][j]+pixbuf_size[j], pixbuf[LIGHTCONE_CHI_OFFSET][j]+pixbuf_size[j], pixbuf[LIGHTCONE_B_OFFSET][j]+pixbuf_size[j], pixbuf[LIGHTCONE_B_OFFSET+1][j]+pixbuf_size[j], pixbuf[LIGHTCONE_B_OFFSET+2][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET+1][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET+2][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET+3][j]+pixbuf_size[j], pixbuf[LIGHTCONE_HIJ_OFFSET+4][j]+pixbuf_size[j], maphdr.Nside, pix, a*a, maphdr.distance, sim.lightcone[i].vertex, R, sim.numpts, fields, sim.out_lightcone[i], desc.pixbatch_size[0], nullptr);
						}

						kernels_running |= (1 << j);
						if (j == 4)
						{
							desc.pixbatch_id.push_back(p);
							desc.pixbatch_offset.push_back(pixbuf_size[j]);
						}

						pixbuf_size[j] += desc.pixbatch_size[pixbatch_type];
						desc.pixbuf_size[j] += desc.pixbatch_size[pixbatch_type];
					} // p-loop

					if (packmap[0] != nullptr)
					{
						packmaps_to_free.push_back(packmap[0]);
						packmap[0] = nullptr;
					}

					if (packmap[1] != nullptr)
					{
						packmaps_to_free.push_back(packmap[1]);
						packmap[1] = nullptr;
					}

					shell_desc.push_back(desc);
				}

				if (kernels_running > 0)
				{
					auto success = cudaDeviceSynchronize();

					if (success != cudaSuccess)
					{
						cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " CUDA error in writeLightcones: " << cudaGetErrorString(success) << endl;
						throw std::runtime_error("CUDA error");
					}

					kernels_running = 0;
				}

				for (int map = 0; map < (int) packmaps_to_free.size(); map++)
					healpix_cuda_check(cudaFree(packmaps_to_free[map]), "pixel packmap free");

				nvtxRangePop();
				p = 0;
				for (j = 0; j < 3; j++)
				{
					if (pixbuf_size[3*j]+pixbuf_size[3*j+1]+pixbuf_size[3*j+2] > p) p = pixbuf_size[3*j]+pixbuf_size[3*j+1]+pixbuf_size[3*j+2];
				}

				if (p > 0)
				{
					nvtxRangePushA("MPI communication (pixel buffers)");
					commbuf = NULL;
					bool commbuf_private = false;
					size_t required_comm_bytes = (size_t) p * sizeof(Real);
#ifdef FFT3D
					if (LATfield2::tempMemory.deviceWorkspace() != NULL && LATfield2::tempMemory.deviceWorkspaceBytes() >= required_comm_bytes)
						commbuf = (Real *) LATfield2::tempMemory.deviceWorkspace();
					else
#endif
					{
						if (!healpix_comm_workspace_fallback_warning)
						{
#ifdef FFT3D
							cout << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " HEALPix output pixel communication buffer exceeds LATfield2 shared device workspace (" << required_comm_bytes << " bytes required, " << LATfield2::tempMemory.deviceWorkspaceBytes() << " bytes available); using private device allocation." << endl;
#else
							cout << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " HEALPix output pixel communication buffer cannot use LATfield2 shared device workspace without FFT3D; using private device allocation." << endl;
#endif
							healpix_comm_workspace_fallback_warning = true;
						}
						healpix_cuda_malloc(&commbuf, p);
						commbuf_private = true;
					}
					bool commbuf_accumulation_pending = false;
					bool edge_pixbuf_accumulation_pending = false;

					for (j = 0; j < LIGHTCONE_MAX_FIELDS; j++)
					{
						if (pixbuf[j][0] != NULL)
						{
							if (parallel.grid_rank()[0] % 2 == 0)
							{
								if (pixbuf_size[0]+pixbuf_size[1]+pixbuf_size[2] > 0)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									healpix_pack3(commbuf, pixbuf[j][0], pixbuf_size[0], pixbuf[j][1], pixbuf_size[1], pixbuf[j][2], pixbuf_size[2]);
									parallel.send_dim0<Real>(commbuf, pixbuf_size[0]+pixbuf_size[1]+pixbuf_size[2], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
								}
								if (pixbuf_size[3]+pixbuf_size[4]+pixbuf_size[5] > 0)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.receive_dim0<Real>(commbuf, pixbuf_size[3]+pixbuf_size[4]+pixbuf_size[5], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
									healpix_add_packed3(pixbuf[j][3], pixbuf_size[3], pixbuf[j][4], pixbuf_size[4], pixbuf[j][5], pixbuf_size[5], commbuf);
									commbuf_accumulation_pending = true;
									edge_pixbuf_accumulation_pending = true;
								}
							}
							else
							{
								if (pixbuf_size[3]+pixbuf_size[4]+pixbuf_size[5] > 0 && parallel.grid_size()[0] > 2)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.receive_dim0<Real>(commbuf, pixbuf_size[3]+pixbuf_size[4]+pixbuf_size[5], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
									healpix_add_packed3(pixbuf[j][3], pixbuf_size[3], pixbuf[j][4], pixbuf_size[4], pixbuf[j][5], pixbuf_size[5], commbuf);
									commbuf_accumulation_pending = true;
									edge_pixbuf_accumulation_pending = true;
								}
								if (pixbuf_size[0]+pixbuf_size[1]+pixbuf_size[2] > 0)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									healpix_pack3(commbuf, pixbuf[j][0], pixbuf_size[0], pixbuf[j][1], pixbuf_size[1], pixbuf[j][2], pixbuf_size[2]);
									parallel.send_dim0<Real>(commbuf, pixbuf_size[0]+pixbuf_size[1]+pixbuf_size[2], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
								}
							}

							if (parallel.grid_rank()[0] % 2 == 0)
							{
								if (pixbuf_size[6]+pixbuf_size[7]+pixbuf_size[8] > 0)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									healpix_pack3(commbuf, pixbuf[j][6], pixbuf_size[6], pixbuf[j][7], pixbuf_size[7], pixbuf[j][8], pixbuf_size[8]);
									parallel.send_dim0<Real>(commbuf, pixbuf_size[6]+pixbuf_size[7]+pixbuf_size[8], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
								}
								if (pixbuf_size[3]+pixbuf_size[4]+pixbuf_size[5] > 0 && parallel.grid_size()[0] > 2)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.receive_dim0<Real>(commbuf, pixbuf_size[3]+pixbuf_size[4]+pixbuf_size[5], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
									healpix_add_packed3(pixbuf[j][3], pixbuf_size[3], pixbuf[j][4], pixbuf_size[4], pixbuf[j][5], pixbuf_size[5], commbuf);
									commbuf_accumulation_pending = true;
									edge_pixbuf_accumulation_pending = true;
								}
							}
							else
							{
								if (pixbuf_size[3]+pixbuf_size[4]+pixbuf_size[5] > 0)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.receive_dim0<Real>(commbuf, pixbuf_size[3]+pixbuf_size[4]+pixbuf_size[5], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
									healpix_add_packed3(pixbuf[j][3], pixbuf_size[3], pixbuf[j][4], pixbuf_size[4], pixbuf[j][5], pixbuf_size[5], commbuf);
									commbuf_accumulation_pending = true;
									edge_pixbuf_accumulation_pending = true;
								}
								if (pixbuf_size[6]+pixbuf_size[7]+pixbuf_size[8] > 0)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									healpix_pack3(commbuf, pixbuf[j][6], pixbuf_size[6], pixbuf[j][7], pixbuf_size[7], pixbuf[j][8], pixbuf_size[8]);
									parallel.send_dim0<Real>(commbuf, pixbuf_size[6]+pixbuf_size[7]+pixbuf_size[8], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
								}
							}

							if (parallel.grid_rank()[1] % 2 == 0)
							{
								if (pixbuf_size[3] > 0)
								{
									healpix_sync_if(edge_pixbuf_accumulation_pending, commbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.send_dim1<Real>(pixbuf[j][3], pixbuf_size[3], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
								}
								if (pixbuf_size[4] > 0)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.receive_dim1<Real>(commbuf, pixbuf_size[4], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
									healpix_add(pixbuf[j][4], commbuf, pixbuf_size[4]);
									commbuf_accumulation_pending = true;
								}
							}
							else
							{
								if (pixbuf_size[4] > 0 && parallel.grid_size()[1] > 2)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.receive_dim1<Real>(commbuf, pixbuf_size[4], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
									healpix_add(pixbuf[j][4], commbuf, pixbuf_size[4]);
									commbuf_accumulation_pending = true;
								}
								if (pixbuf_size[3] > 0)
								{
									healpix_sync_if(edge_pixbuf_accumulation_pending, commbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.send_dim1<Real>(pixbuf[j][3], pixbuf_size[3], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
								}
							}

							if (parallel.grid_rank()[1] % 2 == 0)
							{
								if (pixbuf_size[5] > 0)
								{
									healpix_sync_if(edge_pixbuf_accumulation_pending, commbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.send_dim1<Real>(pixbuf[j][5], pixbuf_size[5], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
								}
								if (pixbuf_size[4] > 0 && parallel.grid_size()[1] > 2)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.receive_dim1<Real>(commbuf, pixbuf_size[4], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
									healpix_add(pixbuf[j][4], commbuf, pixbuf_size[4]);
									commbuf_accumulation_pending = true;
								}
							}
							else
							{
								if (pixbuf_size[4] > 0)
								{
									healpix_sync_if(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.receive_dim1<Real>(commbuf, pixbuf_size[4], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
									healpix_add(pixbuf[j][4], commbuf, pixbuf_size[4]);
									commbuf_accumulation_pending = true;
								}
								if (pixbuf_size[5] > 0)
								{
									healpix_sync_if(edge_pixbuf_accumulation_pending, commbuf_accumulation_pending, "pixel buffer accumulation");
									parallel.send_dim1<Real>(pixbuf[j][5], pixbuf_size[5], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
								}
							}
						}
					}

					healpix_sync_any(commbuf_accumulation_pending, edge_pixbuf_accumulation_pending, "pixel buffer accumulation");
					if (commbuf_private)
						healpix_cuda_check(cudaFree(commbuf), "pixel communication buffer free");
					nvtxRangePop();
				}

				nvtxRangePushA("MPI communication (write buffers)");

				for (j = 0; j < LIGHTCONE_MAX_FIELDS; j++)
				{
					if (pixbuf[j][0] == NULL)
						continue;

					vector<vector<HealpixTransferSegment> > send_segments(parallel.size());
					vector<vector<HealpixTransferSegment> > recv_segments(parallel.size());
					vector<HealpixTransferSegment> local_segments;
					vector<int64_t> send_counts(parallel.size(), 0);
					vector<int64_t> recv_counts(parallel.size(), 0);
					vector<Real *> sendbuf(parallel.size(), NULL);
					vector<Real *> recvbuf(parallel.size(), NULL);
					vector<MPI_Request> requests;
					Real * send_workspace = NULL;
					bool send_workspace_private = false;
					int64_t total_send_count = 0;

					for (int sidx = 0; sidx < (int) shell_desc.size(); sidx++)
					{
						HealpixShellDesc & desc = shell_desc[sidx];
						for (int pixidx = 0; pixidx < (int) desc.pixbatch_id.size(); pixidx += n)
						{
							int dest = healpix_writer_rank(desc, shell_inner, shell_outer, desc.pixbatch_id[pixidx]);
							pixbatch_type = healpix_batch_type(desc, desc.pixbatch_id[pixidx]);
							for (n = 1; pixidx+n < (int) desc.pixbatch_id.size() && desc.pixbatch_id[pixidx+n] == desc.pixbatch_id[pixidx+n-1]+1 && desc.pixbatch_id[pixidx+n] < desc.pixbatch_delim[pixbatch_type] && healpix_writer_rank(desc, shell_inner, shell_outer, desc.pixbatch_id[pixidx+n]) == dest; n++);
							if (dest != parallel.rank() && desc.pixbatch_size[pixbatch_type] > 0)
							{
								HealpixTransferSegment segment;
								segment.rank = dest;
								segment.pixbuf_offset = desc.pixbatch_offset[pixidx];
								segment.outbuf_offset = 0;
								segment.count = n * desc.pixbatch_size[pixbatch_type];
								send_segments[dest].push_back(segment);
								send_counts[dest] += segment.count;
							}
						}

						if (!desc.writes_on_rank)
							continue;

						if (desc.write_batch_count != (int) desc.sender_proc.size())
						{
							cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " pixel batch count mismatch! expecting " << desc.write_batch_count << " but sender list contains " << desc.sender_proc.size() << " entries!" << endl;
							exit(-99);
						}

						for (int p2 = desc.write_batch_begin; p2 < desc.write_batch_begin + desc.write_batch_count; p2 += n)
						{
							pixbatch_type = healpix_batch_type(desc, p2);
							int source = desc.sender_proc[p2 - desc.write_batch_begin];
							for (n = 1; p2+n < desc.write_batch_begin + desc.write_batch_count && desc.sender_proc[p2+n-desc.write_batch_begin] == source && p2+n < desc.pixbatch_delim[pixbatch_type]; n++);
							if (desc.pixbatch_size[pixbatch_type] <= 0)
								continue;

							HealpixTransferSegment segment;
							segment.rank = source;
							segment.outbuf_offset = desc.outbuf_base + healpix_batch_output_offset(desc, p2) - desc.write_file_offset;
							segment.count = n * desc.pixbatch_size[pixbatch_type];
							if (source == parallel.rank())
							{
								segment.pixbuf_offset = healpix_find_pixbatch_offset(desc, p2);
								local_segments.push_back(segment);
							}
							else
							{
								recv_segments[source].push_back(segment);
								recv_counts[source] += segment.count;
							}
						}
					}

					for (int source = 0; source < parallel.size(); source++)
					{
						if (recv_counts[source] > 0)
						{
							recvbuf[source] = (Real *) malloc(recv_counts[source] * sizeof(Real));
							if (recvbuf[source] == NULL)
							{
								cout << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " unable to allocate " << recv_counts[source] * sizeof(Real) << " bytes of memory for pixelisation!" << endl;
								parallel.abortForce();
							}
							requests.push_back(MPI_Request());
							parallel.ireceive<Real>(recvbuf[source], (int) recv_counts[source], source, &requests.back());
						}
					}

					for (int l = 0; l < (int) local_segments.size(); l++)
						healpix_cuda_check(cudaMemcpy((void *) (outbuf[j] + local_segments[l].outbuf_offset), (void *) (pixbuf[j][4] + local_segments[l].pixbuf_offset), local_segments[l].count * sizeof(Real), cudaMemcpyDeviceToHost), "write buffer local copy");

					for (int dest = 0; dest < parallel.size(); dest++)
						total_send_count += send_counts[dest];

					if (total_send_count > 0)
					{
						size_t required_send_bytes = total_send_count * sizeof(Real);
#ifdef FFT3D
						if (LATfield2::tempMemory.deviceWorkspace() != NULL && LATfield2::tempMemory.deviceWorkspaceBytes() >= required_send_bytes)
							send_workspace = (Real *) LATfield2::tempMemory.deviceWorkspace();
						else
#endif
						{
							if (!healpix_send_workspace_fallback_warning)
							{
#ifdef FFT3D
								cout << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " HEALPix output send staging exceeds LATfield2 shared device workspace (" << required_send_bytes << " bytes required, " << LATfield2::tempMemory.deviceWorkspaceBytes() << " bytes available); using private device allocation." << endl;
#else
								cout << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": proc#" << parallel.rank() << " HEALPix output send staging cannot use LATfield2 shared device workspace without FFT3D; using private device allocation." << endl;
#endif
								healpix_send_workspace_fallback_warning = true;
							}
							healpix_cuda_malloc(&send_workspace, total_send_count);
							send_workspace_private = true;
						}
					}

					int64_t send_workspace_offset = 0;
					for (int dest = 0; dest < parallel.size(); dest++)
					{
						if (send_counts[dest] > 0)
						{
							sendbuf[dest] = send_workspace + send_workspace_offset;
							send_workspace_offset += send_counts[dest];
							int64_t send_offset = 0;
							for (int sidx = 0; sidx < (int) send_segments[dest].size(); sidx++)
							{
								healpix_cuda_check(cudaMemcpy((void *) (sendbuf[dest] + send_offset), (void *) (pixbuf[j][4] + send_segments[dest][sidx].pixbuf_offset), send_segments[dest][sidx].count * sizeof(Real), cudaMemcpyDeviceToDevice), "write buffer pack");
								send_offset += send_segments[dest][sidx].count;
							}
							requests.push_back(MPI_Request());
							parallel.isend<Real>(sendbuf[dest], (int) send_counts[dest], dest, &requests.back());
						}
					}

					if (!requests.empty())
						MPI_Waitall((int) requests.size(), requests.data(), MPI_STATUSES_IGNORE);

					for (int source = 0; source < parallel.size(); source++)
					{
						if (recvbuf[source] != NULL)
						{
							int64_t recv_offset = 0;
							for (int sidx = 0; sidx < (int) recv_segments[source].size(); sidx++)
							{
								memcpy((void *) (outbuf[j] + recv_segments[source][sidx].outbuf_offset), (void *) (recvbuf[source] + recv_offset), recv_segments[source][sidx].count * sizeof(Real));
								recv_offset += recv_segments[source][sidx].count;
							}
							free(recvbuf[source]);
						}
					}

					for (int dest = 0; dest < parallel.size(); dest++)
					{
						sendbuf[dest] = NULL;
					}

					if (send_workspace_private && send_workspace != NULL)
						healpix_cuda_check(cudaFree(send_workspace), "write communication buffer free");
				}

				nvtxRangePop();
			} // shell chunk-loop

			if (io_group_size == 0)
				offset2 = 0;

			nvtxRangePushA("write maps to disk");

			for (j = 0; j < LIGHTCONE_MAX_FIELDS; j++)
			{
				if (pixbuf[j][0] == NULL || shell_outer < shell_inner) continue;

				if (sim.num_lightcone > 1)
				{
					if (j == LIGHTCONE_PHI_OFFSET)
						sprintf(filename, "%s%s%d_%04d_phi.map", sim.output_path, sim.basename_lightcone, i, cycle);
					else if (j == LIGHTCONE_CHI_OFFSET)
						sprintf(filename, "%s%s%d_%04d_chi.map", sim.output_path, sim.basename_lightcone, i, cycle);
					else if (j >= LIGHTCONE_B_OFFSET && j < LIGHTCONE_B_OFFSET+3)
						sprintf(filename, "%s%s%d_%04d_B%d.map", sim.output_path, sim.basename_lightcone, i, cycle, j+1-LIGHTCONE_B_OFFSET);
					else if (j >= LIGHTCONE_HIJ_OFFSET && j < LIGHTCONE_HIJ_OFFSET+5)
						sprintf(filename, "%s%s%d_%04d_h%d%d.map", sim.output_path, sim.basename_lightcone, i, cycle, (j - LIGHTCONE_HIJ_OFFSET < 3 ? 1 : 2), j - LIGHTCONE_HIJ_OFFSET + (j - LIGHTCONE_HIJ_OFFSET < 3 ? 1 : -1));
				}
				else
				{
					if (j == LIGHTCONE_PHI_OFFSET)
						sprintf(filename, "%s%s_%04d_phi.map", sim.output_path, sim.basename_lightcone, cycle);
					else if (j == LIGHTCONE_CHI_OFFSET)
						sprintf(filename, "%s%s_%04d_chi.map", sim.output_path, sim.basename_lightcone, cycle);
					else if (j >= LIGHTCONE_B_OFFSET && j < LIGHTCONE_B_OFFSET+3)
						sprintf(filename, "%s%s_%04d_B%d.map", sim.output_path, sim.basename_lightcone, cycle, j+1-LIGHTCONE_B_OFFSET);
					else if (j >= LIGHTCONE_HIJ_OFFSET && j < LIGHTCONE_HIJ_OFFSET+5)
						sprintf(filename, "%s%s_%04d_h%d%d.map", sim.output_path, sim.basename_lightcone, cycle, (j - LIGHTCONE_HIJ_OFFSET < 3 ? 1 : 2), j - LIGHTCONE_HIJ_OFFSET + (j - LIGHTCONE_HIJ_OFFSET < 3 ? 1 : -1));
				}

				MPI_File_open(parallel.lat_world_comm(), filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,  MPI_INFO_NULL, &mapfile);
				MPI_File_set_size(mapfile, (MPI_Offset) bytes);
				MPI_File_write_at_all(mapfile, (MPI_Offset) offset[shell_write] + offset2, (void *) outbuf[j], bytes2, MPI_BYTE, &status);
				MPI_File_close(&mapfile);
			}

			nvtxRangePop();

			offset.clear();
			shell_desc.clear();

			for (j = 0; j < 9*LIGHTCONE_MAX_FIELDS; j++)
			{
				if (pixbuf[j/9][j%9] != NULL)
				{
					healpix_cuda_check(cudaFree(pixbuf[j/9][j%9]), "pixel buffer free");
					pixbuf[j/9][j%9] = NULL;
				}
			}

			for (j = 0; j < LIGHTCONE_MAX_FIELDS; j++)
			{
				if (outbuf[j] != NULL)
				{
					free(outbuf[j]);
					outbuf[j] = NULL;
				}
			}

			nvtxRangePop();
#endif // HAVE_HEALPIX
		}
		else if (parallel.isRoot() && outfile != NULL)
			fclose(outfile);

		if (sim.out_lightcone[i] & MASK_GADGET && sim.lightcone[i].distance[0] > d - tau + 0.5 * dtau_old && sim.lightcone[i].distance[1] <= d - tau + 0.5 * dtau_old && d - tau + 0.5 * dtau_old > 0.)
		{
			nvtxRangePushA("Gadget2 output");
			n = findIntersectingLightcones(sim.lightcone[i], d - tau + (0.5 + LIGHTCONE_IDCHECK_ZONE) * dtau_old, d - tau - 0.5 * dtau, domain, vertex);

			hdr.num_files = 1;
			hdr.Omega0 = cosmo.Omega_m;
			hdr.OmegaLambda = cosmo.Omega_Lambda;
			hdr.HubbleParam = cosmo.h;
			hdr.BoxSize = sim.boxsize / GADGET_LENGTH_CONVERSION;
			hdr.flag_sfr = 0;
			hdr.flag_cooling = 0;
			hdr.flag_feedback = 0;
			hdr.flag_age = 0;
			hdr.flag_metals = 0;
			for (p = 0; p < 256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8 - 2 * 4 - 6 * 4; p++)
				hdr.fill[p] = 0;
			for (p = 0; p < 6; p++)
			{
				hdr.npart[p] = 0;
				hdr.npartTotal[p] = 0;
				hdr.npartTotalHW[p] = 0;
				hdr.mass[p] = 0.;
			}

			hdr.time = a;
			hdr.redshift = (1./a) - 1.;

			if (sim.baryon_flag)
				hdr.mass[1] = (double) sim.tracer_factor[0] * C_RHO_CRIT * cosmo.Omega_cdm * sim.boxsize * sim.boxsize * sim.boxsize / sim.numpcl[0] / GADGET_MASS_CONVERSION;
			else
				hdr.mass[1] = (double) sim.tracer_factor[0] * C_RHO_CRIT * (cosmo.Omega_cdm + cosmo.Omega_b) * sim.boxsize * sim.boxsize * sim.boxsize / sim.numpcl[0] / GADGET_MASS_CONVERSION;

			if (sim.num_lightcone > 1)
				sprintf(filename, "%d_%04d", i, cycle);
			else
				sprintf(filename, "_%04d", cycle);

			if (sim.tracer_factor[0] > 0)
			{
				if (IDlog_multiplicity > 1)
					pcls_cdm->saveGadget2<1>(h5filename + filename + "_cdm", hdr, sim.lightcone[i], d - tau, dtau, dtau_old, a * Hconf(a, fourpiG, cosmo), vertex, n, IDbacklog[sim.IDlog_mapping[i]][0], &IDprelog[sim.IDlog_mapping[i]][0], phi, sim.tracer_factor[0]);
				else
					pcls_cdm->saveGadget2(h5filename + filename + "_cdm", hdr, sim.lightcone[i], d - tau, dtau, dtau_old, a * Hconf(a, fourpiG, cosmo), vertex, n, IDbacklog[sim.IDlog_mapping[i]][0], &IDprelog[sim.IDlog_mapping[i]][0], phi, sim.tracer_factor[0]);
			}

			if (sim.baryon_flag && sim.tracer_factor[1] > 0)
			{
				hdr.mass[1] = (double) sim.tracer_factor[1] * C_RHO_CRIT * cosmo.Omega_b * sim.boxsize * sim.boxsize * sim.boxsize / sim.numpcl[1] / GADGET_MASS_CONVERSION;
				if (IDlog_multiplicity > 1)
					pcls_b->saveGadget2<1>(h5filename + filename + "_b", hdr, sim.lightcone[i], d - tau, dtau, dtau_old, a * Hconf(a, fourpiG, cosmo), vertex, n, IDbacklog[sim.IDlog_mapping[i]][1], &IDprelog[sim.IDlog_mapping[i]][IDlog_multiplicity], phi, sim.tracer_factor[1]);
				else
					pcls_b->saveGadget2(h5filename + filename + "_b", hdr, sim.lightcone[i], d - tau, dtau, dtau_old, a * Hconf(a, fourpiG, cosmo), vertex, n, IDbacklog[sim.IDlog_mapping[i]][1], &IDprelog[sim.IDlog_mapping[i]][1], phi, sim.tracer_factor[1]);
			}

			for (p = 0; p < cosmo.num_ncdm; p++)
			{
				if (sim.numpcl[1+sim.baryon_flag+p] == 0 || sim.tracer_factor[p+1+sim.baryon_flag] == 0) continue;
				sprintf(buffer, "_ncdm%d", p);
				hdr.mass[1] = (double) sim.tracer_factor[p+1+sim.baryon_flag] * C_RHO_CRIT * cosmo.Omega_ncdm[p] * sim.boxsize * sim.boxsize * sim.boxsize / sim.numpcl[p+1+sim.baryon_flag] / GADGET_MASS_CONVERSION;
				if (IDlog_multiplicity > 1)
					pcls_ncdm[p].saveGadget2<1>(h5filename + filename + buffer, hdr, sim.lightcone[i], d - tau, dtau, dtau_old, a * Hconf(a, fourpiG, cosmo), vertex, n, IDbacklog[sim.IDlog_mapping[i]][p+1+sim.baryon_flag], &IDprelog[sim.IDlog_mapping[i]][IDlog_multiplicity*(p+1+sim.baryon_flag)], phi, sim.tracer_factor[p+1+sim.baryon_flag]);
				else
					pcls_ncdm[p].saveGadget2(h5filename + filename + buffer, hdr, sim.lightcone[i], d - tau, dtau, dtau_old, a * Hconf(a, fourpiG, cosmo), vertex, n, IDbacklog[sim.IDlog_mapping[i]][p+1+sim.baryon_flag], &IDprelog[sim.IDlog_mapping[i]][p+1+sim.baryon_flag], phi, sim.tracer_factor[p+1+sim.baryon_flag]);
			}

			nvtxRangePop();
		}
	}

#ifdef HAVE_HEALPIX
	delete[] outbuf;
#endif

	nvtxRangePushA("ID log communication");
	for (int l = 0; l < sim.num_IDlogs; l++)
	{
		for (p = 0; p <= cosmo.num_ncdm + sim.baryon_flag; p++)
		{
			IDbacklog[l][p].clear();

			if (IDlog_multiplicity > 1)
			{
				for (int q = 0; q < IDlog_multiplicity; q++)
					IDlog_sizes[q] = IDprelog[l][p*IDlog_multiplicity+q].size();
			}
			else
			{
				n = IDprelog[l][p].size();
				i = 0;
				j = 0;
			}

			// first, even ranks send upwards, odd ranks receive
			if (parallel.grid_rank()[1] % 2 == 0)
			{
				if (IDlog_multiplicity > 1)
				{
					IDlog_sizes_send1[0] = IDlog_sizes[2];
					IDlog_sizes_send1[1] = IDlog_sizes[5];
					IDlog_sizes_send1[2] = IDlog_sizes[8];

					parallel.send_dim1<int>(IDlog_sizes_send1, 3, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (IDlog_sizes[2] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+2].data(), IDlog_sizes[2], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (IDlog_sizes[5] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+5].data(), IDlog_sizes[5], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (IDlog_sizes[8] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+8].data(), IDlog_sizes[8], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
				}
				else
				{
					parallel.send_dim1<int>(n, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (n > 0)
						parallel.send_dim1<long>(IDprelog[l][p].data(), n, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
				}
			}

			if (parallel.grid_rank()[1] % 2 == 1 || (parallel.grid_rank()[1] == 0 && parallel.grid_size()[1] % 2 == 1)) // odd rank or rank 0 with odd grid size
			{
				if (IDlog_multiplicity > 1)
				{
					parallel.receive_dim1<int>(IDlog_sizes_recv1, 3, (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

					if (IDlog_sizes_recv1[0] + IDlog_sizes_recv1[1] + IDlog_sizes_recv1[2] > 0)
					{
						IDcombuf1 = (long *) malloc((IDlog_sizes_recv1[0] + IDlog_sizes_recv1[1] + IDlog_sizes_recv1[2]) * sizeof(long));

						if (IDlog_sizes_recv1[0] > 0)
							parallel.receive_dim1<long>(IDcombuf1, IDlog_sizes_recv1[0], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

						if (IDlog_sizes_recv1[1] > 0)
							parallel.receive_dim1<long>(IDcombuf1+IDlog_sizes_recv1[0], IDlog_sizes_recv1[1], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

						if (IDlog_sizes_recv1[2] > 0)
							parallel.receive_dim1<long>(IDcombuf1+IDlog_sizes_recv1[0]+IDlog_sizes_recv1[1], IDlog_sizes_recv1[2], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
					}
				}
				else
				{
					parallel.receive_dim1<int>(j, (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

					if (j > 0)
					{
						IDcombuf1 = (long *) malloc(j * sizeof(long));

						parallel.receive_dim1<long>(IDcombuf1, j, (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
					}
				}
			}

			// second, odd ranks send downwards, even ranks receive
			if (parallel.grid_rank()[1] % 2 == 1)
			{
				if (IDlog_multiplicity > 1)
				{
					IDlog_sizes_send1[0] = IDlog_sizes[0];
					IDlog_sizes_send1[1] = IDlog_sizes[3];
					IDlog_sizes_send1[2] = IDlog_sizes[6];

					parallel.send_dim1<int>(IDlog_sizes_send1, 3, parallel.grid_rank()[1]-1);

					if (IDlog_sizes[0] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity].data(), IDlog_sizes[0], parallel.grid_rank()[1]-1);

					if (IDlog_sizes[3] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+3].data(), IDlog_sizes[3], parallel.grid_rank()[1]-1);

					if (IDlog_sizes[6] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+6].data(), IDlog_sizes[6], parallel.grid_rank()[1]-1);
				}
				else
				{
					parallel.send_dim1<int>(n, parallel.grid_rank()[1]-1);

					if (n > 0)
						parallel.send_dim1<long>(IDprelog[l][p].data(), n, parallel.grid_rank()[1]-1);
				}
			}
			else
			{
				if (IDlog_multiplicity > 1)
				{
					parallel.receive_dim1<int>(IDlog_sizes_recv1+3, 3, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (IDlog_sizes_recv1[3] + IDlog_sizes_recv1[4] + IDlog_sizes_recv1[5] > 0)
					{
						IDcombuf2 = (long *) malloc((IDlog_sizes_recv1[3] + IDlog_sizes_recv1[4] + IDlog_sizes_recv1[5]) * sizeof(long));

						if (IDlog_sizes_recv1[3] > 0)
							parallel.receive_dim1<long>(IDcombuf2, IDlog_sizes_recv1[3], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

						if (IDlog_sizes_recv1[4] > 0)
							parallel.receive_dim1<long>(IDcombuf2+IDlog_sizes_recv1[3], IDlog_sizes_recv1[4], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

						if (IDlog_sizes_recv1[5] > 0)
							parallel.receive_dim1<long>(IDcombuf2+IDlog_sizes_recv1[3]+IDlog_sizes_recv1[4], IDlog_sizes_recv1[5], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
					}
				}
				else
				{
					parallel.receive_dim1<int>(i, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (i > 0)
					{
						IDcombuf2 = (long *) malloc(i * sizeof(long));

						parallel.receive_dim1<long>(IDcombuf2, i, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
					}
				}
			}

			// third, even ranks send downwards, odd ranks receive
			if (parallel.grid_rank()[1] % 2 == 0)
			{
				if (IDlog_multiplicity > 1)
				{
					IDlog_sizes_send1[0] = IDlog_sizes[0];
					IDlog_sizes_send1[1] = IDlog_sizes[3];
					IDlog_sizes_send1[2] = IDlog_sizes[6];

					parallel.send_dim1<int>(IDlog_sizes_send1, 3, (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

					if (IDlog_sizes[0] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity].data(), IDlog_sizes[0], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

					if (IDlog_sizes[3] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+3].data(), IDlog_sizes[3], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

					if (IDlog_sizes[6] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+6].data(), IDlog_sizes[6], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
				}
				else
				{
					parallel.send_dim1<int>(n, (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

					if (n > 0)
						parallel.send_dim1<long>(IDprelog[l][p].data(), n, (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
				}
			}
			else
			{
				if (IDlog_multiplicity > 1)
				{
					parallel.receive_dim1<int>(IDlog_sizes_recv1+3, 3, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (IDlog_sizes_recv1[3] + IDlog_sizes_recv1[4] + IDlog_sizes_recv1[5] > 0)
					{
						IDcombuf2 = (long *) malloc((IDlog_sizes_recv1[3] + IDlog_sizes_recv1[4] + IDlog_sizes_recv1[5]) * sizeof(long));

						if (IDlog_sizes_recv1[3] > 0)
							parallel.receive_dim1<long>(IDcombuf2, IDlog_sizes_recv1[3], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

						if (IDlog_sizes_recv1[4] > 0)
							parallel.receive_dim1<long>(IDcombuf2+IDlog_sizes_recv1[3], IDlog_sizes_recv1[4], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

						if (IDlog_sizes_recv1[5] > 0)
							parallel.receive_dim1<long>(IDcombuf2+IDlog_sizes_recv1[3]+IDlog_sizes_recv1[4], IDlog_sizes_recv1[5], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
					}
				}
				else
				{
					parallel.receive_dim1<int>(i, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (i > 0)
					{
						IDcombuf2 = (long *) malloc(i * sizeof(long));

						parallel.receive_dim1<long>(IDcombuf2, i, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
					}
				}
			}

			// fourth, odd ranks send upwards, even ranks receive
			if (parallel.grid_rank()[1] % 2 == 1)
			{
				if (IDlog_multiplicity > 1)
				{
					IDlog_sizes_send1[0] = IDlog_sizes[2];
					IDlog_sizes_send1[1] = IDlog_sizes[5];
					IDlog_sizes_send1[2] = IDlog_sizes[8];

					parallel.send_dim1<int>(IDlog_sizes_send1, 3, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (IDlog_sizes[2] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+2].data(), IDlog_sizes[2], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (IDlog_sizes[5] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+5].data(), IDlog_sizes[5], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (IDlog_sizes[8] > 0)
						parallel.send_dim1<long>(IDprelog[l][p*IDlog_multiplicity+8].data(), IDlog_sizes[8], (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
				}
				else
				{
					parallel.send_dim1<int>(n, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);

					if (n > 0)
						parallel.send_dim1<long>(IDprelog[l][p].data(), n, (parallel.grid_rank()[1]+1) % parallel.grid_size()[1]);
				}
			}
			else if (parallel.grid_rank()[1] > 0 || parallel.grid_size()[1] % 2 == 0)
			{
				if (IDlog_multiplicity > 1)
				{
					parallel.receive_dim1<int>(IDlog_sizes_recv1, 3, (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

					if (IDlog_sizes_recv1[0] + IDlog_sizes_recv1[1] + IDlog_sizes_recv1[2] > 0)
					{
						IDcombuf1 = (long *) malloc((IDlog_sizes_recv1[0] + IDlog_sizes_recv1[1] + IDlog_sizes_recv1[2]) * sizeof(long));

						if (IDlog_sizes_recv1[0] > 0)
							parallel.receive_dim1<long>(IDcombuf1, IDlog_sizes_recv1[0], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

						if (IDlog_sizes_recv1[1] > 0)
							parallel.receive_dim1<long>(IDcombuf1+IDlog_sizes_recv1[0], IDlog_sizes_recv1[1], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

						if (IDlog_sizes_recv1[2] > 0)
							parallel.receive_dim1<long>(IDcombuf1+IDlog_sizes_recv1[0]+IDlog_sizes_recv1[1], IDlog_sizes_recv1[2], (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
					}
				}
				else
				{
					parallel.receive_dim1<int>(j, (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);

					if (j > 0)
					{
						IDcombuf1 = (long *) malloc(j * sizeof(long));

						parallel.receive_dim1<long>(IDcombuf1, j, (parallel.grid_size()[1]+parallel.grid_rank()[1]-1) % parallel.grid_size()[1]);
					}
				}
			}

			// remaining dimension has to communicate

			//first, even ranks send upwards, odd ranks receive
			if (parallel.grid_rank()[0] % 2 == 0)
			{
				if (IDlog_multiplicity > 1)
				{
					IDlog_sizes_send0[0] = IDlog_sizes[6];
					IDlog_sizes_send0[1] = IDlog_sizes[7];
					IDlog_sizes_send0[2] = IDlog_sizes[8];
					IDlog_sizes_send0[3] = IDlog_sizes_recv1[2];
					IDlog_sizes_send0[4] = IDlog_sizes_recv1[5];

					parallel.send_dim0<int>(IDlog_sizes_send0, 5, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					for (int q = 6; q < 9; q++)
					{
						if (IDlog_sizes[q] > 0)
							parallel.send_dim0<long>(IDprelog[l][p*IDlog_multiplicity+q].data(), IDlog_sizes[q], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
					}

					if (IDlog_sizes_recv1[2] > 0)
						parallel.send_dim0<long>(IDcombuf1+IDlog_sizes_recv1[0]+IDlog_sizes_recv1[1], IDlog_sizes_recv1[2], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv1[5] > 0)
						parallel.send_dim0<long>(IDcombuf2+IDlog_sizes_recv1[3]+IDlog_sizes_recv1[4], IDlog_sizes_recv1[5], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
				}
				else
				{
					IDlog_sizes_send0[0] = n;
					IDlog_sizes_send0[1] = j;
					IDlog_sizes_send0[2] = i;

					parallel.send_dim0<int>(IDlog_sizes_send0, 3, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (n > 0)
						parallel.send_dim0<long>(IDprelog[l][p].data(), n, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (j > 0)
						parallel.send_dim0<long>(IDcombuf1, j, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (i > 0)
						parallel.send_dim0<long>(IDcombuf2, i, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
				}
			}

			if (parallel.grid_rank()[0] % 2 == 1 || (parallel.grid_rank()[0] == 0 && parallel.grid_size()[0] % 2 == 1)) // odd rank or rank 0 with odd grid size
			{
				if (IDlog_multiplicity > 1)
				{
					parallel.receive_dim0<int>(IDlog_sizes_recv0, 5, (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv0[0] + IDlog_sizes_recv0[1] + IDlog_sizes_recv0[2] + IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4] > 0)
					{
						IDcombuf3 = (long *) malloc((IDlog_sizes_recv0[0] + IDlog_sizes_recv0[1] + IDlog_sizes_recv0[2] + IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4]) * sizeof(long));

						long * IDcombuf = IDcombuf3;

						for (int q = 0; q < 5; q++)
						{
							if (IDlog_sizes_recv0[q] > 0)
							{
								parallel.receive_dim0<long>(IDcombuf, IDlog_sizes_recv0[q], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
								IDcombuf += IDlog_sizes_recv0[q];
							}
						}
					}
				}
				else
				{
					parallel.receive_dim0<int>(IDlog_sizes_recv0, 3, (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv0[0] + IDlog_sizes_recv0[1] + IDlog_sizes_recv0[2] > 0)
					{
						IDcombuf3 = (long *) malloc((IDlog_sizes_recv0[0] + IDlog_sizes_recv0[1] + IDlog_sizes_recv0[2]) * sizeof(long));

						long * IDcombuf = IDcombuf3;

						for (int q = 0; q < 3; q++)
						{
							if (IDlog_sizes_recv0[q] > 0)
							{
								parallel.receive_dim0<long>(IDcombuf, IDlog_sizes_recv0[q], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
								IDcombuf += IDlog_sizes_recv0[q];
							}
						}
					}
				}
			}

			// second, odd ranks send downwards, even ranks receive
			if (parallel.grid_rank()[0] % 2 == 1)
			{
				if (IDlog_multiplicity > 1)
				{
					IDlog_sizes_send0[0] = IDlog_sizes[0];
					IDlog_sizes_send0[1] = IDlog_sizes[1];
					IDlog_sizes_send0[2] = IDlog_sizes[2];
					IDlog_sizes_send0[3] = IDlog_sizes_recv1[0];
					IDlog_sizes_send0[4] = IDlog_sizes_recv1[3];

					parallel.send_dim0<int>(IDlog_sizes_send0, 5, parallel.grid_rank()[0]-1);

					for(int q = 0; q < 3; q++)
					{
						if (IDlog_sizes[q] > 0)
							parallel.send_dim0<long>(IDprelog[l][p*IDlog_multiplicity+q].data(), IDlog_sizes[q], parallel.grid_rank()[0]-1);
					}

					if (IDlog_sizes_recv1[0] > 0)
						parallel.send_dim0<long>(IDcombuf1, IDlog_sizes_recv1[0], parallel.grid_rank()[0]-1);

					if (IDlog_sizes_recv1[3] > 0)
						parallel.send_dim0<long>(IDcombuf2, IDlog_sizes_recv1[3], parallel.grid_rank()[0]-1);
				}
				else
				{
					IDlog_sizes_send0[0] = n;
					IDlog_sizes_send0[1] = j;
					IDlog_sizes_send0[2] = i;

					parallel.send_dim0<int>(IDlog_sizes_send0, 3, parallel.grid_rank()[0]-1);

					if (n > 0)
						parallel.send_dim0<long>(IDprelog[l][p].data(), n, parallel.grid_rank()[0]-1);

					if (j > 0)
						parallel.send_dim0<long>(IDcombuf1, j, parallel.grid_rank()[0]-1);

					if (i > 0)
						parallel.send_dim0<long>(IDcombuf2, i, parallel.grid_rank()[0]-1);
				}
			}
			else
			{
				if (IDlog_multiplicity > 1)
				{
					parallel.receive_dim0<int>(IDlog_sizes_recv0+5, 5, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv0[5] + IDlog_sizes_recv0[6] + IDlog_sizes_recv0[7] + IDlog_sizes_recv0[8] + IDlog_sizes_recv0[9] > 0)
					{
						IDcombuf4 = (long *) malloc((IDlog_sizes_recv0[5] + IDlog_sizes_recv0[6] + IDlog_sizes_recv0[7] + IDlog_sizes_recv0[8] + IDlog_sizes_recv0[9]) * sizeof(long));

						long * IDcombuf = IDcombuf4;

						for (int q = 5; q < 10; q++)
						{
							if (IDlog_sizes_recv0[q] > 0)
							{
								parallel.receive_dim0<long>(IDcombuf, IDlog_sizes_recv0[q], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
								IDcombuf += IDlog_sizes_recv0[q];
							}
						}
					}
				}
				else
				{
					parallel.receive_dim0<int>(IDlog_sizes_recv0+3, 3, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4] + IDlog_sizes_recv0[5] > 0)
					{
						IDcombuf4 = (long *) malloc((IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4] + IDlog_sizes_recv0[5]) * sizeof(long));

						long * IDcombuf = IDcombuf4;

						for (int q = 3; q < 6; q++)
						{
							if (IDlog_sizes_recv0[q] > 0)
							{
								parallel.receive_dim0<long>(IDcombuf, IDlog_sizes_recv0[q], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
								IDcombuf += IDlog_sizes_recv0[q];
							}
						}
					}
				}
			}

			// third, even ranks send downwards, odd ranks receive
			if (parallel.grid_rank()[0] % 2 == 0)
			{
				if (IDlog_multiplicity > 1)
				{
					IDlog_sizes_send0[0] = IDlog_sizes[0];
					IDlog_sizes_send0[1] = IDlog_sizes[1];
					IDlog_sizes_send0[2] = IDlog_sizes[2];
					IDlog_sizes_send0[3] = IDlog_sizes_recv1[0];
					IDlog_sizes_send0[4] = IDlog_sizes_recv1[3];

					parallel.send_dim0<int>(IDlog_sizes_send0, 5, (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);

					for (int q = 0; q < 3; q++)
					{
						if (IDlog_sizes[q] > 0)
							parallel.send_dim0<long>(IDprelog[l][p*IDlog_multiplicity+q].data(), IDlog_sizes[q], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
					}

					if (IDlog_sizes_recv1[0] > 0)
						parallel.send_dim0<long>(IDcombuf1, IDlog_sizes_recv1[0], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv1[3] > 0)
						parallel.send_dim0<long>(IDcombuf2, IDlog_sizes_recv1[3], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
				}
				else
				{
					IDlog_sizes_send0[0] = n;
					IDlog_sizes_send0[1] = j;
					IDlog_sizes_send0[2] = i;

					parallel.send_dim0<int>(IDlog_sizes_send0, 3, (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);

					if (n > 0)
						parallel.send_dim0<long>(IDprelog[l][p].data(), n, (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);

					if (j > 0)
						parallel.send_dim0<long>(IDcombuf1, j, (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);

					if (i > 0)
						parallel.send_dim0<long>(IDcombuf2, i, (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
				}
			}
			else
			{
				if (IDlog_multiplicity > 1)
				{
					parallel.receive_dim0<int>(IDlog_sizes_recv0+5, 5, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv0[5] + IDlog_sizes_recv0[6] + IDlog_sizes_recv0[7] + IDlog_sizes_recv0[8] + IDlog_sizes_recv0[9] > 0)
					{
						IDcombuf4 = (long *) malloc((IDlog_sizes_recv0[5] + IDlog_sizes_recv0[6] + IDlog_sizes_recv0[7] + IDlog_sizes_recv0[8] + IDlog_sizes_recv0[9]) * sizeof(long));

						long * IDcombuf = IDcombuf4;

						for (int q = 5; q < 10; q++)
						{
							if (IDlog_sizes_recv0[q] > 0)
							{
								parallel.receive_dim0<long>(IDcombuf, IDlog_sizes_recv0[q], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
								IDcombuf += IDlog_sizes_recv0[q];
							}
						}
					}
				}
				else
				{
					parallel.receive_dim0<int>(IDlog_sizes_recv0+3, 3, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4] + IDlog_sizes_recv0[5] > 0)
					{
						IDcombuf4 = (long *) malloc((IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4] + IDlog_sizes_recv0[5]) * sizeof(long));

						long * IDcombuf = IDcombuf4;

						for (int q = 3; q < 6; q++)
						{
							if (IDlog_sizes_recv0[q] > 0)
							{
								parallel.receive_dim0<long>(IDcombuf, IDlog_sizes_recv0[q], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
								IDcombuf += IDlog_sizes_recv0[q];
							}
						}
					}
				}
			}

			// fourth, odd ranks send upwards, even ranks receive
			if (parallel.grid_rank()[0] % 2 == 1)
			{
				if (IDlog_multiplicity > 1)
				{
					IDlog_sizes_send0[0] = IDlog_sizes[6];
					IDlog_sizes_send0[1] = IDlog_sizes[7];
					IDlog_sizes_send0[2] = IDlog_sizes[8];
					IDlog_sizes_send0[3] = IDlog_sizes_recv1[2];
					IDlog_sizes_send0[4] = IDlog_sizes_recv1[5];

					parallel.send_dim0<int>(IDlog_sizes_send0, 5, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					for (int q = 6; q < 9; q++)
					{
						if (IDlog_sizes[q] > 0)
							parallel.send_dim0<long>(IDprelog[l][p*IDlog_multiplicity+q].data(), IDlog_sizes[q], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
					}

					if (IDlog_sizes_recv1[2] > 0)
						parallel.send_dim0<long>(IDcombuf1+IDlog_sizes_recv1[0]+IDlog_sizes_recv1[1], IDlog_sizes_recv1[2], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv1[5] > 0)
						parallel.send_dim0<long>(IDcombuf2+IDlog_sizes_recv1[3]+IDlog_sizes_recv1[4], IDlog_sizes_recv1[5], (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
				}
				else
				{
					IDlog_sizes_send0[0] = n;
					IDlog_sizes_send0[1] = j;
					IDlog_sizes_send0[2] = i;

					parallel.send_dim0<int>(IDlog_sizes_send0, 3, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (n > 0)
						parallel.send_dim0<long>(IDprelog[l][p].data(), n, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (j > 0)
						parallel.send_dim0<long>(IDcombuf1, j, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);

					if (i > 0)
						parallel.send_dim0<long>(IDcombuf2, i, (parallel.grid_rank()[0]+1) % parallel.grid_size()[0]);
				}
			}
			else if (parallel.grid_rank()[0] > 0 || parallel.grid_size()[0] % 2 == 0)
			{
				if (IDlog_multiplicity > 1)
				{
					parallel.receive_dim0<int>(IDlog_sizes_recv0, 5, (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv0[0] + IDlog_sizes_recv0[1] + IDlog_sizes_recv0[2] + IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4] > 0)
					{
						IDcombuf3 = (long *) malloc((IDlog_sizes_recv0[0] + IDlog_sizes_recv0[1] + IDlog_sizes_recv0[2] + IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4]) * sizeof(long));

						long * IDcombuf = IDcombuf3;

						for (int q = 0; q < 5; q++)
						{
							if (IDlog_sizes_recv0[q] > 0)
							{
								parallel.receive_dim0<long>(IDcombuf, IDlog_sizes_recv0[q], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
								IDcombuf += IDlog_sizes_recv0[q];
							}
						}
					}
				}
				else
				{
					parallel.receive_dim0<int>(IDlog_sizes_recv0, 3, (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);

					if (IDlog_sizes_recv0[0] + IDlog_sizes_recv0[1] + IDlog_sizes_recv0[2] > 0)
					{
						IDcombuf3 = (long *) malloc((IDlog_sizes_recv0[0] + IDlog_sizes_recv0[1] + IDlog_sizes_recv0[2]) * sizeof(long));

						long * IDcombuf = IDcombuf3;

						for (int q = 0; q < 3; q++)
						{
							if (IDlog_sizes_recv0[q] > 0)
							{
								parallel.receive_dim0<long>(IDcombuf, IDlog_sizes_recv0[q], (parallel.grid_size()[0]+parallel.grid_rank()[0]-1) % parallel.grid_size()[0]);
								IDcombuf += IDlog_sizes_recv0[q];
							}
						}
					}
				}
			}

			// communication complete, now merge data

			// compute length of IDcombuf1
			if (IDlog_multiplicity > 1)
				j = IDlog_sizes_recv1[0] + IDlog_sizes_recv1[1] + IDlog_sizes_recv1[2];

			for (int q = 0; q < j; q++)
				IDbacklog[l][p].insert(IDcombuf1[q]);

			free(IDcombuf1);
			IDcombuf1 = NULL;

			// compute length of IDcombuf2
			if (IDlog_multiplicity > 1)
				i = IDlog_sizes_recv1[3] + IDlog_sizes_recv1[4] + IDlog_sizes_recv1[5];

			for (int q = 0; q < i; q++)
				IDbacklog[l][p].insert(IDcombuf2[q]);

			free(IDcombuf2);
			IDcombuf2 = NULL;

			// compute length of IDcombuf3
			j = IDlog_sizes_recv0[0] + IDlog_sizes_recv0[1] + IDlog_sizes_recv0[2];

			if (IDlog_multiplicity > 1)
				j += IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4];

			for (int q = 0; q < j; q++)
				IDbacklog[l][p].insert(IDcombuf3[q]);

			free(IDcombuf3);
			IDcombuf3 = NULL;

			// compute length of IDcombuf4
			if (IDlog_multiplicity > 1)
				i = IDlog_sizes_recv0[5] + IDlog_sizes_recv0[6] + IDlog_sizes_recv0[7] + IDlog_sizes_recv0[8] + IDlog_sizes_recv0[9];
			else
				i = IDlog_sizes_recv0[3] + IDlog_sizes_recv0[4] + IDlog_sizes_recv0[5];

			for (int q = 0; q < i; q++)
				IDbacklog[l][p].insert(IDcombuf4[q]);

			free(IDcombuf4);
			IDcombuf4 = NULL;

			// ingest local IDs

			if (IDlog_multiplicity > 1)
			{
				for (int q = 0; q < 9; q++)
				{
					if (IDlog_sizes[q] > 0)
					{
						for (int r = 0; r < IDlog_sizes[q]; r++)
							IDbacklog[l][p].insert(IDprelog[l][p*IDlog_multiplicity+q][r]);

						IDprelog[l][p*IDlog_multiplicity+q].clear();
					}
				}
			}
			else
			{
				for (int q = 0; q < n; q++)
					IDbacklog[l][p].insert(IDprelog[l][p][q]);

				IDprelog[l][p].clear();
			}
		}
	}

	// delete IDprelog
	for (i = 0; i < sim.num_IDlogs; i++)
		delete[] IDprelog[i];

	delete[] IDprelog;

	nvtxRangePop();
}


//////////////////////////
// writeSpectra
//////////////////////////
// Description:
//   output of spectra
//
// Arguments:
//   sim            simulation metadata structure
//   cosmo          cosmological parameter structure
//   fourpiG        4 pi G (in code units)
//   a              scale factor
//   pkcount        spectrum output index
//   pcls_cdm       pointer to particle handler for CDM
//   pcls_b         pointer to particle handler for baryons
//   pcls_ncdm      array of particle handlers for
//                  non-cold DM (may be set to NULL)
//   phi            pointer to allocated field
//   chi            pointer to allocated field
//   Bi             pointer to allocated field
//   source         pointer to allocated field
//   Sij            pointer to allocated field
//   zetaFT         pointer to allocated field (or NULL)
//   scalarFT       pointer to allocated field
//   BiFT           pointer to allocated field
//   SijFT          pointer to allocated field
//   plan_phi       pointer to FFT planner
//   plan_chi       pointer to FFT planner
//   plan_Bi        pointer to FFT planner
//   plan_source    pointer to FFT planner
//   plan_Sij       pointer to FFT planner
//   Bi_check       pointer to allocated field (or NULL)
//   BiFT_check     pointer to allocated field (or NULL)
//   plan_Bi_check  pointer to FFT planner (or NULL)
//   vi             pointer to allocated field (or NULL)
//   viFT           pointer to allocated field (or NULL)
//   plan_vi        pointer to FFT planner (or NULL)
//   hijFT          pointer to allocated field (or NULL)
//   hijprimeFT     pointer to allocated field (or NULL)
//
// Returns:
//
//////////////////////////

void writeSpectra(metadata & sim, cosmology & cosmo, const double fourpiG, const double a, const int pkcount,
#ifdef HAVE_CLASS
background & class_background, perturbs & class_perturbs, icsettings & ic,
#endif
perfParticles_gevolution<part_simple,part_simple_info> * pcls_cdm, perfParticles_gevolution<part_simple,part_simple_info> * pcls_b, Particles_gevolution<part_simple,part_simple_info,part_simple_dataType> * pcls_ncdm, Field<Real> * phi, Field<Real> * chi, Field<Real> * Bi, Field<Real> * source, Field<Real> * Sij, Field<Cplx> * zetaFT, Field<Cplx> * scalarFT, Field<Cplx> * BiFT, Field<Cplx> * SijFT, PlanFFT<Cplx> * plan_phi, PlanFFT<Cplx> * plan_chi, PlanFFT<Cplx> * plan_Bi, PlanFFT<Cplx> * plan_source, PlanFFT<Cplx> * plan_Sij
#ifdef CHECK_B
, Field<Real> * Bi_check, Field<Cplx> * BiFT_check, PlanFFT<Cplx> * plan_Bi_check
#endif
#ifdef VELOCITY
, Field<Real> * vi, Field<Cplx> * viFT, PlanFFT<Cplx> * plan_vi
#endif
#ifdef TENSOR_EVOLUTION
, Field<Cplx> * hijFT, Field<Cplx> * hijprimeFT
#endif
)
{
	char filename[2*PARAM_MAX_LENGTH+64];
	char buffer[64];
	Site x(phi->lattice());
	rKSite kFT(scalarFT->lattice());
	long numpts3d = (long) sim.numpts * (long) sim.numpts * (long) sim.numpts;
	Cplx tempk;
	double Omega_ncdm;

	Real * kbin;
	Real * power;
	Real * kscatter;
	Real * pscatter;
	int * occupation;

	kbin = (Real *) malloc(sim.numbins * sizeof(Real));
	power = (Real *) malloc(sim.numbins * sizeof(Real));
	kscatter = (Real *) malloc(sim.numbins * sizeof(Real));
	pscatter = (Real *) malloc(sim.numbins * sizeof(Real));
	occupation = (int *) malloc(sim.numbins * sizeof(int));

	if (sim.out_pk & MASK_RBARE || sim.out_pk & MASK_DBARE || sim.out_pk & MASK_POT || ((sim.out_pk & MASK_T00 || sim.out_pk & MASK_DELTA) && sim.gr_flag == 0))
	{
		//projection_init(source);
		thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
#ifdef HAVE_CLASS
		if ((sim.radiation_flag > 0 || sim.fluid_flag > 0) && sim.gr_flag == 0)
		{
			projection_T00_project(class_background, class_perturbs, *source, *scalarFT, plan_source, sim, ic, cosmo, fourpiG, a, 1., zetaFT);
			if (sim.out_pk & MASK_DELTA)
			{
				Omega_ncdm = 0;
				for (int i = 0; i < cosmo.num_ncdm; i++)
				{
					if (a < 1. / (sim.z_switch_deltancdm[i] + 1.) && cosmo.Omega_ncdm[i] > 0)
						Omega_ncdm += bg_ncdm(a, cosmo, i);
				}
				plan_source->execute(FFT_FORWARD);
				extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
				sprintf(filename, "%s%s%03d_deltaclass.dat", sim.output_path, sim.basename_pk, pkcount);
				writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * ((a < 1. / (sim.z_switch_deltarad + 1.) ? sim.radiation_flag : 0) * cosmo.Omega_rad / a + sim.fluid_flag * cosmo.Omega_fld / pow(a, 3. * cosmo.w0_fld) + Omega_ncdm) * ((a < 1. / (sim.z_switch_deltarad + 1.) ? sim.radiation_flag : 0) * cosmo.Omega_rad / a + sim.fluid_flag * cosmo.Omega_fld / pow(a, 3. * cosmo.w0_fld) + Omega_ncdm), filename, "power spectrum of delta for linear fields (CLASS)", a, sim.z_pk[pkcount]);
			}
		}
#endif
		scalarProjectionCIC_project(pcls_cdm, source);
		if (sim.baryon_flag)
			scalarProjectionCIC_project(pcls_b, source);
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			scalarProjectionCIC_project(pcls_ncdm+i, source);
		}
		scalarProjectionCIC_comm(source);
		plan_source->execute(FFT_FORWARD);

		if (sim.out_pk & MASK_RBARE || sim.out_pk & MASK_DBARE || ((sim.out_pk & MASK_T00 || sim.out_pk & MASK_DELTA) && sim.gr_flag == 0))
			extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);

		if (sim.out_pk & MASK_RBARE)
		{
			sprintf(filename, "%s%s%03d_rhoN.dat", sim.output_path, sim.basename_pk, pkcount);
			writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * pow(a, 6.0), filename, "power spectrum of rho_N", a, sim.z_pk[pkcount]);
		}

		if (sim.out_pk & MASK_DBARE)
		{
			sprintf(filename, "%s%s%03d_deltaN.dat", sim.output_path, sim.basename_pk, pkcount);
			writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * cosmo.Omega_m * cosmo.Omega_m, filename, "power spectrum of delta_N", a, sim.z_pk[pkcount]);
		}

		if (sim.out_pk & MASK_T00 && sim.gr_flag == 0)
		{
			sprintf(filename, "%s%s%03d_T00.dat", sim.output_path, sim.basename_pk, pkcount);
			writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * pow(a, 6.0), filename, "power spectrum of T00", a, sim.z_pk[pkcount]);
		}

		if (sim.out_pk & MASK_DELTA && sim.gr_flag == 0)
		{
			sprintf(filename, "%s%s%03d_delta.dat", sim.output_path, sim.basename_pk, pkcount);
			writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * cosmo.Omega_m * cosmo.Omega_m, filename, "power spectrum of delta", a, sim.z_pk[pkcount]);
		}

		if (sim.out_pk & MASK_POT)
		{
			solveModifiedPoissonFT(*scalarFT, *scalarFT, fourpiG / a);
			extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
			sprintf(filename, "%s%s%03d_psiN.dat", sim.output_path, sim.basename_pk, pkcount);
			writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI, filename, "power spectrum of psi_N", a, sim.z_pk[pkcount]);
		}

		if ((cosmo.num_ncdm > 0 || sim.baryon_flag) && (sim.out_pk & MASK_DBARE || (sim.out_pk & MASK_DELTA && sim.gr_flag == 0)))
		{
			//projection_init(source);
			thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
			scalarProjectionCIC_project(pcls_cdm, source);
			scalarProjectionCIC_comm(source);
			plan_source->execute(FFT_FORWARD);
			extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
			sprintf(filename, "%s%s%03d_cdm.dat", sim.output_path, sim.basename_pk, pkcount);
			writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * (sim.baryon_flag ? (cosmo.Omega_cdm * cosmo.Omega_cdm) : ((cosmo.Omega_cdm + cosmo.Omega_b) * (cosmo.Omega_cdm + cosmo.Omega_b))), filename, "power spectrum of delta_N for cdm", a, sim.z_pk[pkcount]);
			if (sim.baryon_flag)
			{
				// store k-space information for cross-spectra using SijFT as temporary array
				if (sim.out_pk & MASK_XSPEC)
				{
#pragma omp parallel for collapse(2) default(shared) firstprivate(kFT)
					for (int i = 0; i < scalarFT->lattice().sizeLocal(1); i++)
					{
						for (int j = 0; j < scalarFT->lattice().sizeLocal(2); j++)
						{
							if (!kFT.setCoord(0, j + scalarFT->lattice().coordSkip()[0], i + scalarFT->lattice().coordSkip()[1]))
							{
								std::cerr << "proc#" << parallel.rank() << ": Error in writeSpectra! Could not set coordinates at k=(0, " << j + scalarFT->lattice().coordSkip()[0] << ", " << i + scalarFT->lattice().coordSkip()[1] << ")" << std::endl;
								throw std::runtime_error("Error in writeSpectra: Could not set coordinates.");
							}

							for (int z = 0; z < scalarFT->lattice().sizeLocal(0); z++)
							{
								(*SijFT)(kFT, 0) = (*scalarFT)(kFT);

								kFT.next();
							}
						}
					}
				}
				//projection_init(source);
				thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
				scalarProjectionCIC_project(pcls_b, source);
				scalarProjectionCIC_comm(source);
				plan_source->execute(FFT_FORWARD);
				extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
				sprintf(filename, "%s%s%03d_b.dat", sim.output_path, sim.basename_pk, pkcount);
				writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * cosmo.Omega_b * cosmo.Omega_b, filename, "power spectrum of delta_N for baryons", a, sim.z_pk[pkcount]);
				if (sim.out_pk & MASK_XSPEC)
				{
					extractCrossSpectrum(*scalarFT, *SijFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
					sprintf(filename, "%s%s%03d_cdmxb.dat", sim.output_path, sim.basename_pk, pkcount);
					writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * cosmo.Omega_cdm * cosmo.Omega_b, filename, "cross power spectrum of delta_N for cdm x baryons", a, sim.z_pk[pkcount]);
				}
			}
			Omega_ncdm = 0.;
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				//projection_init(source);
				thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
				if (sim.numpcl[1+sim.baryon_flag+i] > 0)
					scalarProjectionCIC_project(pcls_ncdm+i, source);
				scalarProjectionCIC_comm(source);
				plan_source->execute(FFT_FORWARD);
				extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
				sprintf(filename, "%s%s%03d_ncdm%d.dat", sim.output_path, sim.basename_pk, pkcount, i);
				sprintf(buffer, "power spectrum of delta_N for ncdm %d", i);
				writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * cosmo.Omega_ncdm[i] * cosmo.Omega_ncdm[i], filename, buffer, a, sim.z_pk[pkcount]);
				Omega_ncdm += cosmo.Omega_ncdm[i];
				// store k-space information for cross-spectra using SijFT as temporary array
				if (cosmo.num_ncdm > 1 && i < 6)
				{
#pragma omp parallel for collapse(2) default(shared) firstprivate(kFT)
					for (int ii = 0; ii < scalarFT->lattice().sizeLocal(1); ii++)
					{
						for (int j = 0; j < scalarFT->lattice().sizeLocal(2); j++)
						{
							if (!kFT.setCoord(0, j + scalarFT->lattice().coordSkip()[0], ii + scalarFT->lattice().coordSkip()[1]))
							{
								std::cerr << "proc#" << parallel.rank() << ": Error in writeSpectra! Could not set coordinates at k=(0, " << j + scalarFT->lattice().coordSkip()[0] << ", " << ii + scalarFT->lattice().coordSkip()[1] << ")" << std::endl;
								throw std::runtime_error("Error in writeSpectra: Could not set coordinates.");
							}

							for (int z = 0; z < scalarFT->lattice().sizeLocal(0); z++)
							{
								(*SijFT)(kFT, i) = (*scalarFT)(kFT);

								kFT.next();
							}
						}
					}
				}
			}
			if (cosmo.num_ncdm > 1 && cosmo.num_ncdm <= 7)
			{
#pragma omp parallel for collapse(2) default(shared) firstprivate(kFT)
				for (int ii = 0; ii < scalarFT->lattice().sizeLocal(1); ii++)
				{
					for (int j = 0; j < scalarFT->lattice().sizeLocal(2); j++)
					{
						if (!kFT.setCoord(0, j + scalarFT->lattice().coordSkip()[0], ii + scalarFT->lattice().coordSkip()[1]))
						{
							std::cerr << "proc#" << parallel.rank() << ": Error in writeSpectra! Could not set coordinates at k=(0, " << j + scalarFT->lattice().coordSkip()[0] << ", " << ii + scalarFT->lattice().coordSkip()[1] << ")" << std::endl;
							throw std::runtime_error("Error in writeSpectra: Could not set coordinates.");
						}

						for (int z = 0; z < scalarFT->lattice().sizeLocal(0); z++)
						{
							for (int i = 0; i < cosmo.num_ncdm-1; i++)
								(*scalarFT)(kFT) += (*SijFT)(kFT, i);

							kFT.next();
						}
					}
				}

				extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
				sprintf(filename, "%s%s%03d_ncdm.dat", sim.output_path, sim.basename_pk, pkcount);
				writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * Omega_ncdm * Omega_ncdm, filename, "power spectrum of delta_N for total ncdm", a, sim.z_pk[pkcount]);
			}
			if (cosmo.num_ncdm > 1)
			{
				for (int i = 0; i < cosmo.num_ncdm-1 && i < 5; i++)
				{
					for (int j = i+1; j < cosmo.num_ncdm && j < 6; j++)
					{
						if (sim.out_pk & MASK_XSPEC || (i == 0 && j == 1) || (i == 2 && j == 3) || (i == 4 && j == 5))
						{
							extractCrossSpectrum(*SijFT, *SijFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR, i, j);
							sprintf(filename, "%s%s%03d_ncdm%dx%d.dat", sim.output_path, sim.basename_pk, pkcount, i, j);
							sprintf(buffer, "cross power spectrum of delta_N for ncdm %d x %d", i, j);
							writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * cosmo.Omega_ncdm[i] * cosmo.Omega_ncdm[j], filename, buffer, a, sim.z_pk[pkcount]);
						}
					}
				}
			}
		}
	}

	if (sim.out_pk & MASK_PHI)
	{
		nvtxRangePushA("writeSpectra:phi");
		plan_phi->execute(FFT_FORWARD);
		extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_phi.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI, filename, "power spectrum of phi", a, sim.z_pk[pkcount]);
		nvtxRangePop();
	}

	if (sim.out_pk & MASK_CHI)
	{
		nvtxRangePushA("writeSpectra:chi");
		plan_chi->execute(FFT_FORWARD);
		extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_chi.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI, filename, "power spectrum of chi", a, sim.z_pk[pkcount]);
		nvtxRangePop();
	}

	if (sim.out_pk & MASK_HIJ)
	{
		nvtxRangePushA("writeSpectra:hij");
		//projection_init(Sij);
		thrust::fill_n(thrust::device, Sij->data(), 6*Sij->lattice().sitesLocalGross(), Real(0));
		projection_Tij_project(pcls_cdm, Sij, a, phi);
		if (sim.baryon_flag)
			projection_Tij_project(pcls_b, Sij, a, phi);
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			projection_Tij_project(pcls_ncdm+i, Sij, a, phi);
		}
		projection_Tij_comm(Sij);

		prepareFTsource(*phi, *Sij, *Sij, 2. * fourpiG / (double) sim.numpts / (double) sim.numpts / a);
		plan_Sij->execute(FFT_FORWARD);
		projectFTtensor(*SijFT, *SijFT);

		extractPowerSpectrum(*SijFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_hij.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, 2. * M_PI * M_PI, filename, "power spectrum of hij", a, sim.z_pk[pkcount]);

#ifdef TENSOR_EVOLUTION
		extractPowerSpectrum(*hijFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_hij_dyn.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, 2. * M_PI * M_PI, filename, "power spectrum of hij", a, sim.z_pk[pkcount]);

		extractPowerSpectrum(*hijprimeFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_hij_prime.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, 2. * M_PI * M_PI * Hconf(a, fourpiG, cosmo) * Hconf(a, fourpiG, cosmo), filename, "power spectrum of hij' / Hconf", a, sim.z_pk[pkcount]);
#endif
		nvtxRangePop();
	}

	if ((sim.out_pk & MASK_T00 || sim.out_pk & MASK_DELTA) && sim.gr_flag > 0)
	{
		nvtxRangePushA("writeSpectra:T00");
		//projection_init(source);
		thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
#ifdef HAVE_CLASS
		if (sim.radiation_flag > 0 || sim.fluid_flag > 0)
		{
			projection_T00_project(class_background, class_perturbs, *source, *scalarFT, plan_source, sim, ic, cosmo, fourpiG, a, 1., zetaFT);
			if (sim.out_pk & MASK_DELTA)
			{
				Omega_ncdm = 0;
				for (int i = 0; i < cosmo.num_ncdm; i++)
				{
					if (a < 1. / (sim.z_switch_deltancdm[i] + 1.) && cosmo.Omega_ncdm[i] > 0)
						Omega_ncdm += bg_ncdm(a, cosmo, i);
				}
				plan_source->execute(FFT_FORWARD);
				extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
				sprintf(filename, "%s%s%03d_deltaclass.dat", sim.output_path, sim.basename_pk, pkcount);
				writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * ((a < 1. / (sim.z_switch_deltarad + 1.) ? sim.radiation_flag : 0) * cosmo.Omega_rad / a + sim.fluid_flag * cosmo.Omega_fld / pow(a, 3. * cosmo.w0_fld) + Omega_ncdm) * ((a < 1. / (sim.z_switch_deltarad + 1.) ? sim.radiation_flag : 0) * cosmo.Omega_rad / a + sim.fluid_flag * cosmo.Omega_fld / pow(a, 3. * cosmo.w0_fld) + Omega_ncdm), filename, "power spectrum of delta for linear fields (CLASS)", a, sim.z_pk[pkcount]);
			}
		}
#endif
		projection_T00_project(pcls_cdm, source, a, phi);
		if (sim.baryon_flag)
			projection_T00_project(pcls_b, source, a, phi);
		for (int i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
			projection_T00_project(pcls_ncdm+i, source, a, phi);
		}
		projection_T00_comm(source);

		plan_source->execute(FFT_FORWARD);
		extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);

		if (sim.out_pk & MASK_T00)
		{
			sprintf(filename, "%s%s%03d_T00.dat", sim.output_path, sim.basename_pk, pkcount);
			writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * pow(a, 6.0), filename, "power spectrum of T00", a, sim.z_pk[pkcount]);
		}

		if (sim.out_pk & MASK_DELTA)
		{
			sprintf(filename, "%s%s%03d_delta.dat", sim.output_path, sim.basename_pk, pkcount);
			writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * (cosmo.Omega_cdm + cosmo.Omega_b + bg_ncdm(a, cosmo)) * (cosmo.Omega_cdm + cosmo.Omega_b + bg_ncdm(a, cosmo)), filename, "power spectrum of delta", a, sim.z_pk[pkcount]);
		}

		if (cosmo.num_ncdm > 0 || sim.baryon_flag || sim.radiation_flag > 0 || sim.fluid_flag > 0)
		{
			//projection_init(source);
			thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
			projection_T00_project(pcls_cdm, source, a, phi);
			projection_T00_comm(source);
			plan_source->execute(FFT_FORWARD);
			extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
			if (sim.out_pk & MASK_T00)
			{
				sprintf(filename, "%s%s%03d_T00cdm.dat", sim.output_path, sim.basename_pk, pkcount);
				writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * pow(a, 6.0), filename, "power spectrum of T00 for cdm", a, sim.z_pk[pkcount]);
			}
			if (sim.out_pk & MASK_DELTA)
			{
				sprintf(filename, "%s%s%03d_deltacdm.dat", sim.output_path, sim.basename_pk, pkcount);
				writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * (sim.baryon_flag ? (cosmo.Omega_cdm * cosmo.Omega_cdm) : ((cosmo.Omega_cdm + cosmo.Omega_b) * (cosmo.Omega_cdm + cosmo.Omega_b))), filename, "power spectrum of delta for cdm", a, sim.z_pk[pkcount]);
			}
			if (sim.baryon_flag)
			{
				// store k-space information for cross-spectra using SijFT as temporary array
				if (sim.out_pk & MASK_XSPEC)
				{
#pragma omp parallel for collapse(2) default(shared) firstprivate(kFT)
					for (int i = 0; i < scalarFT->lattice().sizeLocal(1); i++)
					{
						for (int j = 0; j < scalarFT->lattice().sizeLocal(2); j++)
						{
							if (!kFT.setCoord(0, j + scalarFT->lattice().coordSkip()[0], i + scalarFT->lattice().coordSkip()[1]))
							{
								std::cerr << "proc#" << parallel.rank() << ": Error in writeSpectra! Could not set coordinates at k=(0, " << j + scalarFT->lattice().coordSkip()[0] << ", " << i + scalarFT->lattice().coordSkip()[1] << ")" << std::endl;
								throw std::runtime_error("Error in writeSpectra: Could not set coordinates.");
							}

							for (int z = 0; z < scalarFT->lattice().sizeLocal(0); z++)
							{
								(*SijFT)(kFT, 0) = (*scalarFT)(kFT);

								kFT.next();
							}
						}
					}
				}
				//projection_init(source);
				thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
				projection_T00_project(pcls_b, source, a, phi);
				projection_T00_comm(source);
				plan_source->execute(FFT_FORWARD);
				extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
				if (sim.out_pk & MASK_T00)
				{
					sprintf(filename, "%s%s%03d_T00b.dat", sim.output_path, sim.basename_pk, pkcount);
					writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * pow(a, 6.0), filename, "power spectrum of T00 for baryons", a, sim.z_pk[pkcount]);
				}
				if (sim.out_pk & MASK_DELTA)
				{
					sprintf(filename, "%s%s%03d_deltab.dat", sim.output_path, sim.basename_pk, pkcount);
					writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * cosmo.Omega_b * cosmo.Omega_b, filename, "power spectrum of delta for baryons", a, sim.z_pk[pkcount]);
				}
				if (sim.out_pk & MASK_XSPEC)
				{
					extractCrossSpectrum(*scalarFT, *SijFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
					sprintf(filename, "%s%s%03d_deltacdmxb.dat", sim.output_path, sim.basename_pk, pkcount);
					writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * cosmo.Omega_b * cosmo.Omega_cdm, filename, "cross power spectrum of delta for cdm x baryons", a, sim.z_pk[pkcount]);
				}
			}
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				//projection_init(source);
				thrust::fill_n(thrust::device, source->data(), source->lattice().sitesLocalGross(), Real(0));
				if (sim.numpcl[1+sim.baryon_flag+i] > 0)
					projection_T00_project(pcls_ncdm+i, source, a, phi);
				projection_T00_comm(source);
				plan_source->execute(FFT_FORWARD);
				extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
				if (sim.out_pk & MASK_T00)
				{
					sprintf(filename, "%s%s%03d_T00ncdm%d.dat", sim.output_path, sim.basename_pk, pkcount, i);
					sprintf(buffer, "power spectrum of T00 for ncdm %d", i);
					writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * pow(a, 6.0), filename, buffer, a, sim.z_pk[pkcount]);
				}
				if (sim.out_pk & MASK_DELTA)
				{
					sprintf(filename, "%s%s%03d_deltancdm%d.dat", sim.output_path, sim.basename_pk, pkcount, i);
					sprintf(buffer, "power spectrum of delta for ncdm %d", i);
					writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * bg_ncdm(a, cosmo, i) * bg_ncdm(a, cosmo, i), filename, buffer, a, sim.z_pk[pkcount]);
				}
				// store k-space information for cross-spectra using SijFT as temporary array
				if (cosmo.num_ncdm > 1 && i < 6)
				{
#pragma omp parallel for collapse(2) default(shared) firstprivate(kFT)
					for (int ii = 0; ii < scalarFT->lattice().sizeLocal(1); ii++)
					{
						for (int j = 0; j < scalarFT->lattice().sizeLocal(2); j++)
						{
							if (!kFT.setCoord(0, j + scalarFT->lattice().coordSkip()[0], ii + scalarFT->lattice().coordSkip()[1]))
							{
								std::cerr << "proc#" << parallel.rank() << ": Error in writeSpectra! Could not set coordinates at k=(0, " << j + scalarFT->lattice().coordSkip()[0] << ", " << ii + scalarFT->lattice().coordSkip()[1] << ")" << std::endl;
								throw std::runtime_error("Error in writeSpectra: Could not set coordinates.");
							}

							for (int z = 0; z < scalarFT->lattice().sizeLocal(0); z++)
							{
								(*SijFT)(kFT, i) = (*scalarFT)(kFT);

								kFT.next();
							}
						}
					}
				}
			}
			if (cosmo.num_ncdm > 1 && cosmo.num_ncdm <= 7)
			{
#pragma omp parallel for collapse(2) default(shared) firstprivate(kFT)
				for (int ii = 0; ii < scalarFT->lattice().sizeLocal(1); ii++)
				{
					for (int j = 0; j < scalarFT->lattice().sizeLocal(2); j++)
					{
						if (!kFT.setCoord(0, j + scalarFT->lattice().coordSkip()[0], ii + scalarFT->lattice().coordSkip()[1]))
						{
							std::cerr << "proc#" << parallel.rank() << ": Error in writeSpectra! Could not set coordinates at k=(0, " << j + scalarFT->lattice().coordSkip()[0] << ", " << ii + scalarFT->lattice().coordSkip()[1] << ")" << std::endl;
							throw std::runtime_error("Error in writeSpectra: Could not set coordinates.");
						}

						for (int z = 0; z < scalarFT->lattice().sizeLocal(0); z++)
						{
							for (int i = 0; i < cosmo.num_ncdm-1; i++)
								(*scalarFT)(kFT) += (*SijFT)(kFT, i);

							kFT.next();
						}
					}
				}

				extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR);
				if (sim.out_pk & MASK_T00)
				{
					sprintf(filename, "%s%s%03d_T00ncdm.dat", sim.output_path, sim.basename_pk, pkcount);
					writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * pow(a, 6.0), filename, "power spectrum of T00 for total ncdm", a, sim.z_pk[pkcount]);
				}
				if (sim.out_pk & MASK_DELTA)
				{
					sprintf(filename, "%s%s%03d_deltancdm.dat", sim.output_path, sim.basename_pk, pkcount);
					writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * bg_ncdm(a, cosmo) * bg_ncdm(a, cosmo), filename, "power spectrum of delta for total ncdm", a, sim.z_pk[pkcount]);
				}
			}
			if (cosmo.num_ncdm > 1)
			{
				for (int i = 0; i < cosmo.num_ncdm-1 && i < 5; i++)
				{
					for (int j = i+1; j < cosmo.num_ncdm && j < 6; j++)
					{
						if (sim.out_pk & MASK_XSPEC || (i == 0 && j == 1) || (i == 2 && j == 3) || (i == 4 && j == 5))
						{
							extractCrossSpectrum(*SijFT, *SijFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, true, KTYPE_LINEAR, i, j);
							if (sim.out_pk & MASK_T00)
							{
								sprintf(filename, "%s%s%03d_T00ncdm%dx%d.dat", sim.output_path, sim.basename_pk, pkcount, i, j);
								sprintf(buffer, "cross power spectrum of T00 for ncdm %d x %d", i, j);
								writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * pow(a, 6.0), filename, buffer, a, sim.z_pk[pkcount]);
							}
							if (sim.out_pk & MASK_DELTA)
							{
								sprintf(filename, "%s%s%03d_deltancdm%dx%d.dat", sim.output_path, sim.basename_pk, pkcount, i, j);
								sprintf(buffer, "cross power spectrum of delta for ncdm %d x %d", i, j);
								writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * bg_ncdm(a, cosmo, i) * bg_ncdm(a, cosmo, j), filename, buffer, a, sim.z_pk[pkcount]);
							}
						}
					}
				}
			}
		}
		nvtxRangePop();
	}

	if (sim.out_pk & MASK_B)
	{
		nvtxRangePushA("writeSpectra:B");
		extractPowerSpectrum(*BiFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_B.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, a * a * a * a * sim.numpts * sim.numpts * 2. * M_PI * M_PI, filename, "power spectrum of B", a, sim.z_pk[pkcount]);

#ifdef CHECK_B
		if (sim.vector_flag == VECTOR_PARABOLIC)
		{
			//projection_init(Bi_check);
			thrust::fill_n(thrust::device, Bi_check->data(), 3*Bi_check->lattice().sitesLocalGross(), Real(0));
			projection_T0i_project(pcls_cdm, Bi_check, phi);
			if (sim.baryon_flag)
				projection_T0i_project(pcls_b, Bi_check, phi);
			for (int i = 0; i < cosmo.num_ncdm; i++)
			{
				if (sim.numpcl[1+sim.baryon_flag+i] == 0) continue;
				projection_T0i_project(pcls_ncdm+i, Bi_check, phi);
			}
			projection_T0i_comm(Bi_check);
			plan_Bi_check->execute(FFT_FORWARD);
			projectFTvector(*BiFT_check, *BiFT_check, fourpiG / (double) sim.numpts / (double) sim.numpts);
		}
		extractPowerSpectrum(*BiFT_check, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_B_check.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, a * a * a * a * sim.numpts * sim.numpts * 2. * M_PI * M_PI, filename, "power spectrum of B", a, sim.z_pk[pkcount]);
#endif
		nvtxRangePop();
	}

#ifdef VELOCITY
	if (sim.out_pk & MASK_VEL)
	{
		nvtxRangePushA("writeSpectra:vel");
		plan_vi->execute(FFT_FORWARD);
		extractPowerSpectrum(*viFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_v.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI, filename, "power spectrum of velocity", a, sim.z_pk[pkcount]);

		projectFTtheta(*scalarFT, *viFT);
		extractPowerSpectrum(*scalarFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_theta.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * sim.boxsize * sim.boxsize / cosmo.h / cosmo.h, filename, "power spectrum of theta (div v)", a, sim.z_pk[pkcount]);

		projectFTomega(*viFT);
		extractPowerSpectrum(*viFT, kbin, power, kscatter, pscatter, occupation, sim.numbins, false, KTYPE_LINEAR);
		sprintf(filename, "%s%s%03d_omega.dat", sim.output_path, sim.basename_pk, pkcount);
		writePowerSpectrum(kbin, power, kscatter, pscatter, occupation, sim.numbins, sim.boxsize, (Real) numpts3d * (Real) numpts3d * 2. * M_PI * M_PI * sim.boxsize * sim.boxsize / cosmo.h / cosmo.h, filename, "power spectrum of omega (curl v)", a, sim.z_pk[pkcount]);
		nvtxRangePop();
	}
#endif

	free(kbin);
	free(power);
	free(kscatter);
	free(pscatter);
	free(occupation);
}

#endif
