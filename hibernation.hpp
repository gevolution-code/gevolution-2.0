//////////////////////////
// hibernation.hpp
//////////////////////////
//
// Auxiliary functions for hibernation
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich)
//
// Last modified: August 2024
//
// Modified for the GPU backend:
//  - the restart settings file is produced by copying the _settings_used.ini file
//    generated at initialisation and overriding only the restart-relevant
//    parameters, so that all other settings (cosmology, output, lightcones,
//    hibernation schedule, ...) are carried over automatically;
//  - particles are stored in Gadget2 format (this version has no HDF5 particle
//    I/O), saving the raw phase-space state (no drift/kick) so that the restart
//    read continues the leapfrog exactly (ic_read restores dtau_old).
//
//////////////////////////

#ifndef HIBERNATION_HEADER
#define HIBERNATION_HEADER

//////////////////////////
// restartOverridesParameter
//////////////////////////
// Description:
//   helper for writeRestartSettings: returns true if the named parameter is one
//   that the restart settings file overrides. Such parameters are stripped from
//   the copy of the _settings_used.ini file to avoid duplicates, since
//   parseParameter() picks the *first* occurrence of a parameter name.
//
// Arguments:
//   pname   trimmed parameter name (as returned by readline())
//
// Returns:
//   true if the parameter is overridden by the restart settings, false otherwise
//
//////////////////////////

inline bool restartOverridesParameter(const char * pname)
{
	static const char * names[] = {
		"IC generator", "template file", "particle file", "metric file",
		"restart redshift", "cycle", "tau", "dtau", "gevolution version"
#ifdef TENSOR_EVOLUTION
		, "GWreadFields", "hijfile", "hijprimefile"
#endif
	};

	for (unsigned int i = 0; i < sizeof(names) / sizeof(names[0]); i++)
		if (strcmp(pname, names[i]) == 0) return true;

	return false;
}

//////////////////////////
// writeRestartSettings
//////////////////////////
// Description:
//   writes a settings file containing all the relevant metadata for restarting
//   a run from a hibernation point. The file is seeded from the
//   _settings_used.ini file produced at initialisation (so that every setting of
//   the original run is carried over) and the restart-relevant parameters are
//   then overridden.
//
// Arguments:
//   sim            simulation metadata structure
//   ic             settings for IC generation
//   cosmo          cosmological parameter structure
//   a              scale factor
//   tau            conformal coordinate time
//   dtau           time step
//   cycle          current main control loop cycle count
//   restartcount   restart counter aka number of hibernation point (default -1)
//                  if < 0 no number is associated to the hibernation point
//
// Returns:
//
//////////////////////////

void writeRestartSettings(metadata & sim, icsettings & ic, cosmology & cosmo, const double a, const double tau, const double dtau, const int cycle, const int restartcount = -1)
{
	char buffer[2*PARAM_MAX_LENGTH+24];
	char numtag[8];
	char line[PARAM_MAX_LINESIZE];
	char pname[PARAM_MAX_LENGTH];
	char pvalue[PARAM_MAX_LENGTH];
	FILE * outfile;
	FILE * infile;
	int i;

	if (!parallel.isRoot()) return;

	// express particle files are always written one-per-rank as "<base>.<rank>"
	// (saveExpress appends the rank suffix even for a single task); the reader
	// (ic_read) is given the first file "<base>.0" and reconstructs the per-rank
	// names from it.
	const char * pclsuffix = ".0";

	if (restartcount >= 0)
		sprintf(numtag, "%03d", restartcount);
	else
		numtag[0] = '\0';

	if (restartcount >= 0)
		sprintf(buffer, "%s%s%03d.ini", sim.restart_path, sim.basename_restart, restartcount);
	else
		sprintf(buffer, "%s%s.ini", sim.restart_path, sim.basename_restart);

	outfile = fopen(buffer, "w");
	if (outfile == NULL)
	{
		cout << " error opening file for restart settings!" << endl;
		return;
	}

	fprintf(outfile, "# automatically generated settings for restart after hibernation ");
	if (restartcount < 0)
		fprintf(outfile, "due to wallclock limit ");
	else
		fprintf(outfile, "requested ");
	fprintf(outfile, "at redshift z=%f\n", (1./a)-1.);

	// seed the restart settings from the settings file that was actually used for
	// the current run, stripping the parameters that we override below; everything
	// else (cosmology, output, lightcones, hibernation schedule, ...) is carried
	// over verbatim
	sprintf(buffer, "%s%s_settings_used.ini", sim.output_path, sim.basename_generic);
	infile = fopen(buffer, "r");
	if (infile == NULL)
	{
		cout << " /!\\ warning: unable to open " << buffer << " to seed restart settings; restart settings will be incomplete." << endl;
	}
	else
	{
		fprintf(outfile, "# (carried over from %s, restart-relevant parameters overridden below)\n\n", buffer);
		while (fgets(line, PARAM_MAX_LINESIZE, infile) != NULL)
		{
			if (readline(line, pname, pvalue) && restartOverridesParameter(pname))
				continue;
			fputs(line, outfile);
		}
		fclose(infile);
	}

	fprintf(outfile, "\n\n# ==== restart-specific parameters (auto-generated) ====\n\n");

	// "express" selects the Gadget-2 reader that loads raw code units (no unit
	// conversion), matching what saveExpress wrote
	fprintf(outfile, "IC generator       = express\n");

	fprintf(outfile, "particle file      = %s%s%s_cdm%s", sim.restart_path, sim.basename_restart, numtag, pclsuffix);
	if (sim.baryon_flag)
		fprintf(outfile, ", %s%s%s_b%s", sim.restart_path, sim.basename_restart, numtag, pclsuffix);
	for (i = 0; i < cosmo.num_ncdm; i++)
	{
		if (sim.numpcl[1+sim.baryon_flag+i] < 1)
			fprintf(outfile, ", /dev/null");
		else
			fprintf(outfile, ", %s%s%s_ncdm%d%s", sim.restart_path, sim.basename_restart, numtag, i, pclsuffix);
	}
	fprintf(outfile, "\n");

	if (sim.gr_flag > 0)
	{
		fprintf(outfile, "metric file        = %s%s%s_phi.h5", sim.restart_path, sim.basename_restart, numtag);
		fprintf(outfile, ", %s%s%s_chi.h5", sim.restart_path, sim.basename_restart, numtag);
		if (sim.vector_flag == VECTOR_PARABOLIC)
			fprintf(outfile, ", %s%s%s_B.h5\n", sim.restart_path, sim.basename_restart, numtag);
		else
#ifdef CHECK_B
			fprintf(outfile, ", %s%s%s_B_check.h5\n", sim.restart_path, sim.basename_restart, numtag);
#else
			fprintf(outfile, "\n");
#endif
	}
	else if (sim.vector_flag == VECTOR_PARABOLIC)
		fprintf(outfile, "metric file        = %s%s%s_B.h5\n", sim.restart_path, sim.basename_restart, numtag);
#ifdef CHECK_B
	else
		fprintf(outfile, "metric file        = %s%s%s_B_check.h5\n", sim.restart_path, sim.basename_restart, numtag);
#endif

	fprintf(outfile, "restart redshift   = %.15lf\n", (1./a) - 1.);
	fprintf(outfile, "cycle              = %d\n", cycle);
	fprintf(outfile, "tau                = %.15le\n", tau);
	fprintf(outfile, "dtau               = %.15le\n", dtau);
	fprintf(outfile, "gevolution version = %g\n", GEVOLUTION_VERSION);

#ifdef TENSOR_EVOLUTION
	// dynamical tensor degrees of freedom: force reading hij/hij' from the snapshots
	fprintf(outfile, "GWreadFields       = 1\n");
	fprintf(outfile, "hijfile            = %s%s%s_hij.h5\n", sim.restart_path, sim.basename_restart, numtag);
	fprintf(outfile, "hijprimefile       = %s%s%s_hijprime.h5\n", sim.restart_path, sim.basename_restart, numtag);
#endif

	fclose(outfile);
}


//////////////////////////
// hibernateSaveParticles
//////////////////////////
// Description:
//   writes a single particle species to a Gadget2 file as part of a hibernation
//   point. The full set of particles is stored (tracer factor 1) in their raw
//   phase-space state (dtau_pos = dtau_vel = 0, i.e. no drift/kick), so that the
//   restart read reproduces the exact leapfrog state.
//
// Arguments:
//   pcls           pointer to particle handler
//   filebase       base name of the Gadget2 file ("<base>" or "<base>.<rank>")
//   numpcl         total number of particles of this species
//   Omega_species  density parameter of this species (only used for the Gadget2
//                  mass entry; the restart recomputes the mass from cosmology)
//   sim            simulation metadata structure
//   cosmo          cosmological parameter structure
//   a              scale factor
//
//////////////////////////

inline void hibernateSaveParticles(perfParticles_gevolution<part_simple,part_simple_info> * pcls, const string & filebase, const long numpcl, const double Omega_species, metadata & sim, cosmology & cosmo, const double a)
{
	gadget2_header hdr;

	memset(&hdr, 0, sizeof(hdr));
	hdr.num_files = parallel.size();
	hdr.Omega0 = cosmo.Omega_m;
	hdr.OmegaLambda = cosmo.Omega_Lambda;
	hdr.HubbleParam = cosmo.h;
	hdr.BoxSize = sim.boxsize / GADGET_LENGTH_CONVERSION;
	hdr.time = a;
	hdr.redshift = (1. / a) - 1.;
	// npart[1] is the per-file count; for a single file it must be the total, for
	// the multi-file (one-per-rank) case saveGadget2 overwrites it with the local
	// count. npartTotal[1] / npartTotalHW[1] always hold the grand total.
	hdr.npart[1] = (uint32_t) (numpcl % (1ll << 32));
	hdr.npartTotal[1] = (uint32_t) (numpcl % (1ll << 32));
	hdr.npartTotalHW[1] = (uint32_t) (numpcl / (1ll << 32));
	hdr.mass[1] = (double) C_RHO_CRIT * Omega_species * sim.boxsize * sim.boxsize * sim.boxsize / (double) numpcl / GADGET_MASS_CONVERSION;

	// express format: a valid Gadget-2 snapshot whose positions/momenta are stored
	// in raw code units (no drift/kick, no unit conversion), so the restart read
	// reproduces the exact leapfrog state while the files stay readable by generic
	// Gadget tools (python, ...).
	pcls->saveExpress(filebase, hdr);
}


//////////////////////////
// hibernate
//////////////////////////
// Description:
//   creates a hibernation point by writing snapshots of the simulation data and metadata
//
// Arguments:
//   sim            simulation metadata structure
//   ic             settings for IC generation
//   cosmo          cosmological parameter structure
//   pcls_cdm       pointer to particle handler for CDM
//   pcls_b         pointer to particle handler for baryons
//   pcls_ncdm      array of particle handlers for non-cold DM
//   phi            reference to field containing first Bardeen potential
//   chi            reference to field containing difference of Bardeen potentials
//   Bi             reference to vector field containing frame-dragging potential
//   a              scale factor
//   tau            conformal coordinate time
//   dtau           time step
//   cycle          current main control loop cycle count
//   restartcount   restart counter aka number of hibernation point (default -1)
//                  if < 0 no number is associated to the hibernation point
//
// Returns:
//
//////////////////////////

void hibernate(metadata & sim, icsettings & ic, cosmology & cosmo, perfParticles_gevolution<part_simple,part_simple_info> * pcls_cdm, perfParticles_gevolution<part_simple,part_simple_info> * pcls_b, perfParticles_gevolution<part_simple,part_simple_info> * pcls_ncdm, Field<Real> & phi, Field<Real> & chi, Field<Real> & Bi,
#ifdef TENSOR_EVOLUTION
	Field<Cplx> & hijFT, Field<Cplx> & hijprimeFT,
#endif
	const double a, const double tau, const double dtau, const int cycle, const int restartcount = -1)
{
	string h5filename;
	char buffer[16];
	int i;
	Site x(Bi.lattice());

	h5filename.reserve(2*PARAM_MAX_LENGTH);
	h5filename.assign(sim.restart_path);
	h5filename += sim.basename_restart;
	if (restartcount >= 0)
	{
		sprintf(buffer, "%03d", restartcount);
		h5filename += buffer;
	}

	writeRestartSettings(sim, ic, cosmo, a, tau, dtau, cycle, restartcount);

#ifndef CHECK_B
	if (sim.vector_flag == VECTOR_PARABOLIC)
#endif
	for (x.first(); x.test(); x.next())
	{
		Bi(x,0) /= a * a * sim.numpts;
		Bi(x,1) /= a * a * sim.numpts;
		Bi(x,2) /= a * a * sim.numpts;
	}

	// ---- particles (Gadget2; this version has no HDF5 particle I/O) ----
	hibernateSaveParticles(pcls_cdm, h5filename + "_cdm", sim.numpcl[0], (sim.baryon_flag ? cosmo.Omega_cdm : (cosmo.Omega_cdm + cosmo.Omega_b)), sim, cosmo, a);
	if (sim.baryon_flag)
		hibernateSaveParticles(pcls_b, h5filename + "_b", sim.numpcl[1], cosmo.Omega_b, sim, cosmo, a);
	for (i = 0; i < cosmo.num_ncdm; i++)
	{
		if (sim.numpcl[1+sim.baryon_flag+i] < 1) continue;
		sprintf(buffer, "%d", i);
		hibernateSaveParticles(&pcls_ncdm[i], h5filename + "_ncdm" + buffer, sim.numpcl[1+sim.baryon_flag+i], cosmo.Omega_ncdm[i], sim, cosmo, a);
	}

	// ---- fields ----
	if (sim.gr_flag > 0)
	{
		phi.saveHDF5(h5filename + "_phi.h5");
		chi.saveHDF5(h5filename + "_chi.h5");
	}

	if (sim.vector_flag == VECTOR_PARABOLIC)
		Bi.saveHDF5(h5filename + "_B.h5");
#ifdef CHECK_B
	else
		Bi.saveHDF5(h5filename + "_B_check.h5");
#endif

#ifdef TENSOR_EVOLUTION
	hijFT.saveHDF5(h5filename + "_hij.h5");
	hijprimeFT.saveHDF5(h5filename + "_hijprime.h5");
#endif
}

#endif
