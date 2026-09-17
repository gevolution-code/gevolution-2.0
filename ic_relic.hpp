//////////////////////////
// ic_relic.hpp
//////////////////////////
// 
// initial condition generator for gevolution using input files in the format as provided by RELIC
// [see J. Adamek, J. Calles, T. Montandon, J. Noreña, and C. Stahl, JCAP 2204, 001 (2022)]
//
// Authors: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich)
//		    Thomas Montandon (Universität Wien & Université de Montpellier)
//
// Last modified: August 2024
//
//////////////////////////

#ifndef  IC_RELIC_HEADER
#define  IC_RELIC_HEADER

using namespace LATfield2;

void generateIC_relic(metadata & sim, icsettings & ic, cosmology & cosmo, const double fourpiG, Particles<part_simple,part_simple_info,part_simple_dataType> * pcls_cdm, Particles<part_simple,part_simple_info,part_simple_dataType> * pcls_b, Particles<part_simple,part_simple_info,part_simple_dataType> * pcls_ncdm, double * maxvel, Field<Real> * phi, Field<Real> * chi, Field<Real> * Bi, Field<Real> * source, Field<Real> * Sij, Field<Cplx> * & zetaFT, Field<Cplx> * scalarFT, Field<Cplx> * BiFT, Field<Cplx> * SijFT, PlanFFT<Cplx> * plan_phi, PlanFFT<Cplx> * plan_chi, PlanFFT<Cplx> * plan_Bi, PlanFFT<Cplx> * plan_source, PlanFFT<Cplx> * plan_Sij, parameter * params, int & numparam)
{
	double a = 1. / (1. + sim.z_in);
	float * pcldata = NULL;
	double * sinc = NULL;
	Site x(phi->lattice());
	rKSite kFT(scalarFT->lattice());
	double max_displacement;
	part_simple_info pcls_cdm_info;
	part_simple_dataType pcls_cdm_dataType;
	part_simple_info pcls_b_info;
	part_simple_dataType pcls_b_dataType;
	Real boxSize[3] = {1.,1.,1.};
	Field<Real> * ic_fields[2];
	string filename;
	int reduce = MAX;
	
	ic_fields[0] = chi;
	ic_fields[1] = chi;
	
	// generate the kernel for the 1st order displacement field

	loadHomogeneousTemplate(ic.pclfile[0], sim.numpcl[0], pcldata);
	
	if (pcldata == NULL)
	{
		COUT << " error: particle data was empty!" << endl;
		parallel.abortForce();
	}

	uint64_t capacity = (16L * sim.numpcl[0] * (long) ic.numtile[0] * (long) ic.numtile[0] * (long) ic.numtile[0]) / (parallel.size() * 15L);

	plan_source->preallocate(device_workspace_preallocation_bytes(
		(capacity + PCL_EXTRA_CAPACITY) * 3L * sizeof(Real), PCLBUFFER));
	
	if (ic.flags & ICFLAG_CORRECT_DISPLACEMENT)
		generateCICKernel(*source, sim.numpcl[0], pcldata, ic.numtile[0]);
	else
		generateCICKernel(*source);	
	
	plan_source->execute(FFT_FORWARD);
	

	// use BiFT as temporary storage for the kernel
	for (kFT.first(); kFT.test(); kFT.next())
		(*BiFT)(kFT, 0) = (*scalarFT)(kFT);
	
	// precompute sinc function
	sinc = (double *) malloc(sim.numpts * sizeof(double));

	sinc[0] = 1.;
	for (int i = 1; i <= sim.numpts/2; i++)
		sinc[i] = sin(M_PI * i / sim.numpts) / (M_PI * i / sim.numpts);

	for (int i = 1; i < sim.numpts/2; i++)
		sinc[sim.numpts-i] = sinc[i];

	if (sim.gr_flag == 1 || sim.radiation_flag > 0 || sim.fluid_flag > 0)
	{
		filename.assign(ic.metricfile[0]);

		COUT << " reading " << filename << "..." << endl;

		phi->loadHDF5(filename);
		phi->updateHalo();

        // toma
    	plan_phi->execute(FFT_FORWARD);

#ifdef HAVE_CLASS
		if (zetaFT == NULL && (sim.radiation_flag > 0 || sim.fluid_flag > 0))
		{
			zetaFT = new Field<Cplx>;
			zetaFT->initialize(scalarFT->lattice(), 1);
			zetaFT->alloc();

			for (kFT.first(); kFT.test(); kFT.next())
				(*zetaFT)(kFT) = (*scalarFT)(kFT);
		}
#endif

    	for (kFT.first(); kFT.test(); kFT.next())
    	{
    		double W = sinc[kFT.coord(0)] * sinc[kFT.coord(1)] * sinc[kFT.coord(2)];
    		(*scalarFT)(kFT) *= W*W/sim.numpts/sim.numpts/sim.numpts;
    	}

    	plan_phi->execute(FFT_BACKWARD);
    	phi->updateHalo();
    	// end toma
	}
	
	filename.assign(ic.densityfile[0]);

	COUT << " reading " << filename << "..." << endl;

	source->loadHDF5(filename);
	
	// toma
    source->updateHalo();
    plan_source->execute(FFT_FORWARD);

    for (kFT.first(); kFT.test(); kFT.next())
    {
    	double W = sinc[kFT.coord(0)] * sinc[kFT.coord(1)] * sinc[kFT.coord(2)];
    	(*scalarFT)(kFT) *= W*W/sim.numpts/sim.numpts/sim.numpts;
    }

    plan_source->execute(FFT_BACKWARD);
    source->updateHalo();
    // end toma


	COUT << " computing 1st order displacement..." << endl;
	
	if (sim.gr_flag == 1)
	{
		for (x.first(); x.test(); x.next())
			(*chi)(x) = (*source)(x) - 3. * (*phi)(x);
	}
	else
	{
		for (x.first(); x.test(); x.next())
			(*chi)(x) = (*source)(x);
	}

	plan_chi->execute(FFT_FORWARD);
	
	for (kFT.first(); kFT.test(); kFT.next())
	{
		if ((*BiFT)(kFT, 0).norm() > 1.0e-16)
			(*scalarFT)(kFT) = (*scalarFT)(kFT) / (*BiFT)(kFT, 0);
	}
	
	plan_chi->execute(FFT_BACKWARD);
	chi->updateHalo();
	
	//filename.assign("1st_order_displacement.h5");
	//chi->saveHDF5(filename);
	
	strcpy(pcls_cdm_info.type_name, "part_simple");
	if (sim.baryon_flag == 1)
		pcls_cdm_info.mass = cosmo.Omega_cdm / (Real) (sim.numpcl[0]*(long)ic.numtile[0]*(long)ic.numtile[0]*(long)ic.numtile[0]);
	else
		pcls_cdm_info.mass = (cosmo.Omega_cdm + cosmo.Omega_b) / (Real) (sim.numpcl[0]*(long)ic.numtile[0]*(long)ic.numtile[0]*(long)ic.numtile[0]);
	pcls_cdm_info.relativistic = false;
	
	pcls_cdm->initialize(pcls_cdm_info, pcls_cdm_dataType, &(phi->lattice()), boxSize);
	
	initializeParticlePositions(sim.numpcl[0], pcldata, ic.numtile[0], *pcls_cdm);

	if (sim.baryon_flag == 3)	// baryon treatment = hybrid; displace particles using both displacement fields
		pcls_cdm->moveParticles(displace_pcls_ic_basic, 1./sim.numpts/sim.numpts/sim.numpts, ic_fields, 2, NULL, &max_displacement, &reduce, 1);
	else
		pcls_cdm->moveParticles(displace_pcls_ic_basic, 1./sim.numpts/sim.numpts/sim.numpts, &chi, 1, NULL, &max_displacement, &reduce, 1);	// displace CDM particles
	
	sim.numpcl[0] *= (long) ic.numtile[0] * (long) ic.numtile[0] * (long) ic.numtile[0];
	
	COUT << " " << sim.numpcl[0] << " cdm particles initialized: maximum displacement (1st order) = " << max_displacement * sim.numpts << " lattice units." << endl;
	
	filename.assign(ic.velocityfile[0]); // not useful for Newton 

	COUT << " reading " << filename << "..." << endl;

	chi->loadHDF5(filename);
	chi->updateHalo();

	// toma
	plan_chi->execute(FFT_FORWARD);
	
	for (kFT.first(); kFT.test(); kFT.next())
	{
		double W = sinc[kFT.coord(0)] * sinc[kFT.coord(1)] * sinc[kFT.coord(2)];
    	(*scalarFT)(kFT) *= W*W/sim.numpts/sim.numpts/sim.numpts;
	}
	
	plan_chi->execute(FFT_BACKWARD);
	chi->updateHalo();
	// end toma

	maxvel[0] = pcls_cdm->updateVel(initialize_q_ic_basic, -a/sim.boxsize, &chi, 1) / a;

	COUT << " computing 1st order density..." << endl;
	
	projection_init(chi);
	projection_T00_project(pcls_cdm, chi, a, sim.gr_flag ? phi : NULL); // gr_flag=1 -> GR, else Newton
	scalarProjectionCIC_comm(chi);
	
	for (x.first(); x.test(); x.next())
		(*chi)(x) = (*chi)(x) / (cosmo.Omega_cdm + cosmo.Omega_b) - 1.;
		
	//filename.assign("1st_order_density.h5");
	//chi->saveHDF5(filename);

	for (x.first(); x.test(); x.next())
		(*chi)(x) = (*source)(x) - (*chi)(x);
		
	plan_chi->execute(FFT_FORWARD);
	
	for (kFT.first(); kFT.test(); kFT.next())
	{
		if ((*BiFT)(kFT, 0).norm() > 1.0e-16)
			(*scalarFT)(kFT) = (*scalarFT)(kFT) / (*BiFT)(kFT, 0);
	}
	
	plan_chi->execute(FFT_BACKWARD);
	chi->updateHalo();
	
	//filename.assign("2nd_order_displacement.h5");
	//chi->saveHDF5(filename);
	
	pcls_cdm->moveParticles(displace_pcls_ic_basic, 1./sim.numpts/sim.numpts/sim.numpts, &chi, 1, NULL, &max_displacement, &reduce, 1);
	
	COUT << " " << sim.numpcl[0] << " cdm particles initialized: maximum displacement (2nd order) = " << max_displacement * sim.numpts << " lattice units." << endl;
	
	free(pcldata);

	COUT << " computing 2nd order density..." << endl;

    projection_init(chi);
    projection_T00_project(pcls_cdm, chi, a, sim.gr_flag ? phi : NULL); // toma
    scalarProjectionCIC_comm(chi);

    for (x.first(); x.test(); x.next())
            (*chi)(x) = (*source)(x) - ((*chi)(x) / (cosmo.Omega_cdm + cosmo.Omega_b) - 1.);

	plan_chi->execute(FFT_FORWARD);

    for (kFT.first(); kFT.test(); kFT.next())
    {
        if ((*BiFT)(kFT, 0).norm() > 1.0e-16)
            (*scalarFT)(kFT) = (*scalarFT)(kFT) / (*BiFT)(kFT, 0);
    }

    plan_chi->execute(FFT_BACKWARD);
    chi->updateHalo();

    //filename.assign("3rd_order_displacement.h5");
    //chi->saveHDF5(filename);

	pcls_cdm->moveParticles(displace_pcls_ic_basic, 1./sim.numpts/sim.numpts/sim.numpts, &chi, 1, NULL, &max_displacement, &reduce, 1);

    COUT << " " << sim.numpcl[0] << " cdm particles initialized: maximum displacement (3rd order) = " << max_displacement * sim.numpts << " lattice units." << endl;
	
	if (sim.baryon_flag == 1)
	{
		loadHomogeneousTemplate(ic.pclfile[1], sim.numpcl[1], pcldata);
	
		if (pcldata == NULL)
		{
			COUT << " error: particle data was empty!" << endl;
			parallel.abortForce();
		}
		
		strcpy(pcls_b_info.type_name, "part_simple");
		pcls_b_info.mass = cosmo.Omega_b / (Real) (sim.numpcl[1]*(long)ic.numtile[1]*(long)ic.numtile[1]*(long)ic.numtile[1]);
		pcls_b_info.relativistic = false;
	
		pcls_b->initialize(pcls_b_info, pcls_b_dataType, &(phi->lattice()), boxSize);
	
		initializeParticlePositions(sim.numpcl[1], pcldata, ic.numtile[1], *pcls_b);
		
		pcls_b->moveParticles(displace_pcls_ic_basic, 1./sim.boxsize/sim.boxsize, &phi, 1, NULL, &max_displacement, &reduce, 1);	// displace baryon particles
	
		sim.numpcl[1] *= (long) ic.numtile[1] * (long) ic.numtile[1] * (long) ic.numtile[1];
	
		COUT << " " << sim.numpcl[1] << " baryon particles initialized: maximum displacement = " << max_displacement * sim.numpts << " lattice units." << endl;
	
		free(pcldata);
	}
	
	if (ic.pkfile[0] == '\0')	// set velocities using transfer functions
	{
		filename.assign(ic.velocityfile[1]);
		
		chi->loadHDF5(filename);
		
		chi->updateHalo();
		
		if (sim.baryon_flag == 3)	// baryon treatment = hybrid; set velocities using both velocity potentials
			maxvel[0] = pcls_cdm->updateVel(initialize_q_ic_basic, -a/sim.boxsize, ic_fields, 2) / a;
		else
			maxvel[0] = pcls_cdm->updateVel(initialize_q_ic_basic, -a/sim.boxsize, &chi, 1) / a;	// set CDM velocities
		
		if (sim.baryon_flag == 1)
			maxvel[1] = pcls_b->updateVel(initialize_q_ic_basic, -a/sim.boxsize, &phi, 1) / a;	// set baryon velocities
	}
	
	if (sim.baryon_flag > 1) sim.baryon_flag = 0;
	
	projection_init(Bi);
	projection_T0i_project(pcls_cdm, Bi,  phi);
	if (sim.baryon_flag)
		projection_T0i_project(pcls_b, Bi, phi);
	projection_T0i_comm(Bi);
	prepareFTsource(*Bi, *phi, 3. * a * a * Hconf(a, fourpiG, cosmo) * (double) sim.numpts / fourpiG);
	plan_Bi->execute(FFT_FORWARD);
	projectFTvector(*BiFT, *BiFT, fourpiG / (double) sim.numpts / (double) sim.numpts);	
	plan_Bi->execute(FFT_BACKWARD);	
	Bi->updateHalo();	// B initialized
	
	ic_fields[1] = Bi;
	
	projection_init(Sij);
	projection_Tij_project(pcls_cdm, Sij, a, phi);
	if (sim.baryon_flag)
		projection_Tij_project(pcls_b, Sij, a, phi);
	projection_Tij_comm(Sij);
	
	prepareFTsource<Real>(*phi, *Sij, *Sij, 2. * fourpiG / a / (double) sim.numpts / (double) sim.numpts);	
	plan_Sij->execute(FFT_FORWARD);	
	projectFTscalar(*SijFT, *scalarFT);
	plan_chi->execute(FFT_BACKWARD);		
	chi->updateHalo();	// chi now finally contains chi

	free(sinc);
}

#endif
