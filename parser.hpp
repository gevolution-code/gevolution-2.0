//////////////////////////
// parser.hpp
//////////////////////////
// 
// Parser for settings file
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich & ETH Zürich)
//
// Last modified: June 2026
//
//////////////////////////

#ifndef PARSER_HEADER
#define PARSER_HEADER

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <cstdint>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "metadata.hpp"

using namespace std;

struct parameter
{
	char name[PARAM_MAX_LENGTH];
	char value[PARAM_MAX_LENGTH];
	bool used;
	int line;
};

struct parser_diagnostic
{
	bool error;
	int line;
	string parameter;
	string value;
	string message;
};

static vector<parser_diagnostic> parser_diagnostics;

void clearParserDiagnostics()
{
	parser_diagnostics.clear();
}

void addParserDiagnostic(const bool error, const int line, const char * parameter, const char * value, const string & message)
{
	parser_diagnostic diagnostic;
	diagnostic.error = error;
	diagnostic.line = line;
	diagnostic.parameter = parameter == NULL ? "" : parameter;
	diagnostic.value = value == NULL ? "" : value;
	diagnostic.message = message;
	parser_diagnostics.push_back(diagnostic);
}

void addParserError(const parameter & param, const string & message)
{
	addParserDiagnostic(true, param.line, param.name, param.value, message);
}

int parserErrorCount()
{
	int count = 0;
	for (size_t i = 0; i < parser_diagnostics.size(); i++)
		if (parser_diagnostics[i].error) count++;
	return count;
}

void printParserDiagnostics()
{
#ifdef LATFIELD2_HPP
	if (!parallel.isRoot()) return;
#endif
	for (size_t i = 0; i < parser_diagnostics.size(); i++)
	{
		const parser_diagnostic & diagnostic = parser_diagnostics[i];
		cout << (diagnostic.error ? COLORTEXT_RED " error" : COLORTEXT_YELLOW " /!\\ warning") << COLORTEXT_RESET;
		if (diagnostic.line > 0) cout << " at settings line " << diagnostic.line;
		if (!diagnostic.parameter.empty()) cout << " for '" << diagnostic.parameter << "'";
		if (!diagnostic.value.empty()) cout << " = '" << diagnostic.value << "'";
		cout << ": " << diagnostic.message << endl;
	}
}

void abortOnParserErrors()
{
	printParserDiagnostics();
	if (parserErrorCount() > 0)
	{
#ifdef LATFIELD2_HPP
		if (parallel.isRoot())
#endif
			cout << COLORTEXT_RED << " aborting" << COLORTEXT_RESET << ": settings validation found " << parserErrorCount() << " error(s)." << endl;
#ifdef LATFIELD2_HPP
		parallel.abortForce();
#else
		throw runtime_error("settings validation failed");
#endif
	}
	clearParserDiagnostics();
}

int findParameter(parameter * params, const int numparam, const char * pname)
{
	for (int i = 0; i < numparam; i++)
		if (strcmp(params[i].name, pname) == 0) return i;
	return -1;
}

void addParameterError(parameter * params, const int numparam, const char * pname, const string & message)
{
	const int index = findParameter(params, numparam, pname);
	if (index >= 0)
		addParserError(params[index], message);
	else
		addParserDiagnostic(true, 0, pname, NULL, message);
}

void addParameterWarning(parameter * params, const int numparam, const char * pname, const string & message)
{
	const int index = findParameter(params, numparam, pname);
	if (index >= 0)
		addParserDiagnostic(false, params[index].line, params[index].name, params[index].value, message);
	else
		addParserDiagnostic(false, 0, pname, NULL, message);
}

bool strictInteger(const char * text, long & value)
{
	char * end;
	errno = 0;
	value = strtol(text, &end, 10);
	while (*end == ' ' || *end == '\t') end++;
	return end != text && *end == '\0' && errno != ERANGE;
}

bool strictDouble(const char * text, double & value)
{
	char * end;
	errno = 0;
	value = strtod(text, &end);
	while (*end == ' ' || *end == '\t') end++;
	return end != text && *end == '\0' && errno != ERANGE && isfinite(value);
}


int sort_descending(const void * z1, const void * z2)
{
	return (* (double *) z1 > * (double *) z2) ? -1 : ((* (double *) z1 < * (double *) z2) ? 1 : 0);
}


//////////////////////////
// readline
//////////////////////////
// Description:
//   reads a line of characters and checks if it declares a parameter; if yes,
//   i.e. the line has the format "<parameter name> = <parameter value>" (with
//   an optional comment added, preceded by a hash-symbol, '#'), the parameter
//   name and value are copied to the corresponding arrays, and 'true' is returned.
//   If the format is not recognized (or the line is commented using the hash-symbol)
//   'false' is returned instead.
// 
// Arguments:
//   line       string containing the line to be read
//   pname      will contain the name of the declared parameter (if found)
//   pvalue     will contain the value of the declared parameter (if found)
//
// Returns:
//   'true' if a parameter is declared in the line, 'false' otherwise.
// 
//////////////////////////

bool readline(char * line, char * pname, char * pvalue)
{
	char * pequal;
	char * phash;
	char * l;
	char * r;
	
	pequal = strchr(line, '=');
	
	if (pequal == NULL || pequal == line) return false;
	
	phash = strchr(line, '#');
	
	if (phash != NULL && phash < pequal) return false;
	
	l = line;
	while (*l == ' ' || *l == '\t') l++;
	
	r = pequal-1;
	while ((*r == ' ' || *r == '\t') && r > line) r--;
	
	if (r < l) return false;
	
	if (r-l+1 >= PARAM_MAX_LENGTH) return false;
	
	strncpy(pname, l, r-l+1);
	pname[r-l+1] = '\0';
	
	l = pequal+1;
	while (*l == ' ' || *l == '\t') l++;
	
	if (phash == NULL)
		r = line+strlen(line)-1;
	else
		r = phash-1;
	  
	while (*r == ' ' || *r == '\t' || *r == '\n' || *r == '\r') r--;
	
	if (r < l) return false;
	
	if (r-l+1 >= PARAM_MAX_LENGTH) return false;
	
	strncpy(pvalue, l, r-l+1);
	pvalue[r-l+1] = '\0';
	
	return true;
}


//////////////////////////
// loadParameterFile
//////////////////////////
// Description:
//   loads a parameter file and creates an array of parameters declared therein
// 
// Arguments:
//   filename   string containing the path to the parameter file
//   params     will contain the array of parameters (memory will be allocated)
//
// Returns:
//   number of parameters defined in the parameter file (= length of parameter array)
// 
//////////////////////////

int loadParameterFile(const char * filename, parameter * & params)
{
	int numparam = 0;
	int i = 0;
	int line_number = 0;

#ifdef LATFIELD2_HPP	
	if (parallel.grid_rank()[0] == 0) // read file
	{
#endif
		FILE * paramfile;
		char line[PARAM_MAX_LINESIZE];
		char pname[PARAM_MAX_LENGTH];
		char pvalue[PARAM_MAX_LENGTH];
		
		paramfile = fopen(filename, "r");
		
		if (paramfile == NULL)
		{
#ifdef LATFIELD2_HPP
			cerr << " proc#" << parallel.rank() << ": error in loadParameterFile! Unable to open parameter file " << filename << "." << endl;
			parallel.abortForce();
#else
			cerr << " error in loadParameterFile! Unable to open parameter file " << filename << "." << endl;
			return -1;
#endif
		}
		
		while (!feof(paramfile) && !ferror(paramfile))
		{
			if (fgets(line, PARAM_MAX_LINESIZE, paramfile) == NULL) break;
			line_number++;

			if (strchr(line, '\n') == NULL && !feof(paramfile))
			{
				int ch;
				while ((ch = fgetc(paramfile)) != '\n' && ch != EOF);
				addParserDiagnostic(true, line_number, NULL, NULL, "line exceeds PARAM_MAX_LINESIZE");
				continue;
			}
			
			if (readline(line, pname, pvalue) == true)
				numparam++;
			else
			{
				char * equal = strchr(line, '=');
				char * hash = strchr(line, '#');
				if (equal != NULL && (hash == NULL || equal < hash))
					addParserDiagnostic(true, line_number, NULL, NULL, "malformed parameter declaration or parameter/value exceeds PARAM_MAX_LENGTH");
			}
		}
		
		if (numparam == 0)
		{
			fclose(paramfile);
#ifdef LATFIELD2_HPP
			cerr << " proc#" << parallel.rank() << ": error in loadParameterFile! No valid data found in file " << filename << "." << endl;
			parallel.abortForce();
#else
			cerr << " error in loadParameterFile! No valid data found in file " << filename << "." << endl;
			return -1;
#endif
		}
		
		params = (parameter *) malloc(sizeof(parameter) * numparam);
		
		if (params == NULL)
		{
			fclose(paramfile);
#ifdef LATFIELD2_HPP
			cerr << " proc#" << parallel.rank() << ": error in loadParameterFile! Memory error." << endl;
			parallel.abortForce();
#else
			cerr << " error in loadParameterFile! Memory error." << endl;
			return -1;
#endif
		}
		
		rewind(paramfile);
		line_number = 0;
		
		while (!feof(paramfile) && !ferror(paramfile) && i < numparam)
		{
			if (fgets(line, PARAM_MAX_LINESIZE, paramfile) == NULL) break;
			line_number++;
			if (strchr(line, '\n') == NULL && !feof(paramfile))
			{
				int ch;
				while ((ch = fgetc(paramfile)) != '\n' && ch != EOF);
				continue;
			}
			
			if (readline(line, params[i].name, params[i].value) == true)
			{
				params[i].used = false;
				params[i].line = line_number;
				i++;
			}
		}
		
		fclose(paramfile);
		
		if (i < numparam)
		{
			free(params);
#ifdef LATFIELD2_HPP
			cerr << " proc#" << parallel.rank() << ": error in loadParameterFile! File may have changed or file pointer corrupted." << endl;
			parallel.abortForce();
#else
			cerr << " error in loadParameterFile! File may have changed or file pointer corrupted." << endl;
			return -1;
#endif
		}

		for (i = 0; i < numparam; i++)
		{
			for (int j = 0; j < i; j++)
			{
				if (strcmp(params[i].name, params[j].name) == 0)
				{
					if (strcmp(params[i].value, params[j].value) == 0)
						addParserDiagnostic(false, params[i].line, params[i].name, params[i].value, "duplicate parameter repeats the value from an earlier line; the first occurrence is used");
					else
						addParserDiagnostic(true, params[i].line, params[i].name, params[i].value, "conflicting duplicate parameter; the first occurrence is used");
					break;
				}
			}
		}

#ifdef LATFIELD2_HPP		
		parallel.broadcast_dim0<int>(numparam, 0);
	}
	else
	{
		parallel.broadcast_dim0<int>(numparam, 0);
		
		params = (parameter *) malloc(sizeof(parameter) * numparam);
		
		if (params == NULL)
		{
			cerr << " proc#" << parallel.rank() << ": error in loadParameterFile! Memory error." << endl;
			parallel.abortForce();
		}
	}
	
	parallel.broadcast_dim0<parameter>(params, numparam, 0);
#endif
	
	return numparam;
}


//////////////////////////
// saveParameterFile
//////////////////////////
// Description:
//   saves a parameter file
// 
// Arguments:
//   filename   string containing the path to the parameter file
//   params     array of parameters
//   numparam   length of parameter array
//   used_only  if 'true', only the used parameters will be written (default)
//
// Returns:
// 
//////////////////////////

void saveParameterFile(const char * filename, parameter * params, const int numparam, bool used_only = true)
{
#ifdef LATFIELD2_HPP
	if (parallel.isRoot())
#endif
	{
		FILE * paramfile;
		
		paramfile = fopen(filename, "w");
		
		if (paramfile == NULL)
		{
			cout << " error in saveParameterFile! Unable to open file " << filename << "." << endl;
		}
		else
		{
			for (int i = 0; i < numparam; i++)
			{
				if (!used_only || params[i].used)
					fprintf(paramfile, "%s = %s\n", params[i].name, params[i].value);
			}
			
			fclose(paramfile);
		}
	}
}


//////////////////////////
// parseParameter (int)
//////////////////////////
// Description:
//   searches parameter array for specified parameter name and parses its value as integer
// 
// Arguments:
//   params     array of parameters
//   numparam   length of parameter array
//   pname      name of parameter to search for
//   pvalue     reference to integer which will contain the parsed parameter value (if found)
//
// Returns:
//   'true' if parameter is found and parsed successfully, 'false' otherwise
// 
//////////////////////////

bool parseParameter(parameter * & params, const int numparam, const char * pname, int & pvalue)
{   
	for (int i = 0; i < numparam; i++)
	{
		if (strcmp(params[i].name, pname) == 0)
		{
			long value;
			if (strictInteger(params[i].value, value) && value >= INT_MIN && value <= INT_MAX)
			{
				pvalue = (int) value;
				params[i].used = true;
				return true;
			}
			addParserError(params[i], "expected one integer with no trailing characters");
			params[i].used = true;
			return false;
		}
	}
	
	return false;
}


//////////////////////////
// parseParameter (long)
//////////////////////////
// Description:
//   searches parameter array for specified parameter name and parses its value as integer
// 
// Arguments:
//   params     array of parameters
//   numparam   length of parameter array
//   pname      name of parameter to search for
//   pvalue     reference to integer which will contain the parsed parameter value (if found)
//
// Returns:
//   'true' if parameter is found and parsed successfully, 'false' otherwise
// 
//////////////////////////

bool parseParameter(parameter * & params, const int numparam, const char * pname, long & pvalue)
{   
	for (int i = 0; i < numparam; i++)
	{
		if (strcmp(params[i].name, pname) == 0)
		{
			long value;
			if (strictInteger(params[i].value, value))
			{
				pvalue = value;
				params[i].used = true;
				return true;
			}
			addParserError(params[i], "expected one integer with no trailing characters");
			params[i].used = true;
			return false;
		}
	}
	
	return false;
}


//////////////////////////
// parseParameter (double)
//////////////////////////
// Description:
//   searches parameter array for specified parameter name and parses its value as double
// 
// Arguments:
//   params     array of parameters
//   numparam   length of parameter array
//   pname      name of parameter to search for
//   pvalue     reference to double which will contain the parsed parameter value (if found)
//
// Returns:
//   'true' if parameter is found and parsed successfully, 'false' otherwise
// 
//////////////////////////

bool parseParameter(parameter * & params, const int numparam, const char * pname, double & pvalue)
{   
	for (int i = 0; i < numparam; i++)
	{
		if (strcmp(params[i].name, pname) == 0)
		{
			double value;
			if (strictDouble(params[i].value, value))
			{
				pvalue = value;
				params[i].used = true;
				return true;
			}
			addParserError(params[i], "expected one finite number with no trailing characters");
			params[i].used = true;
			return false;
		}
	}
	
	return false;
}


//////////////////////////
// parseParameter (char *)
//////////////////////////
// Description:
//   searches parameter array for specified parameter name and retrieves its value as string
// 
// Arguments:
//   params     array of parameters
//   numparam   length of parameter array
//   pname      name of parameter to search for
//   pvalue     character string which will contain a copy of the parameter value (if found)
//
// Returns:
//   'true' if parameter is found, 'false' otherwise
// 
//////////////////////////

bool parseParameter(parameter * & params, const int numparam, const char * pname, char * pvalue)
{   
	for (int i = 0; i < numparam; i++)
	{
		if (strcmp(params[i].name, pname) == 0)
		{
			strcpy(pvalue, params[i].value);
			params[i].used = true;
			return true;
		}
	}
	
	return false;
}


//////////////////////////
// parseParameter (double *)
//////////////////////////
// Description:
//   searches parameter array for specified parameter name and parses it as a list of comma-separated double values
// 
// Arguments:
//   params     array of parameters
//   numparam   length of parameter array
//   pname      name of parameter to search for
//   pvalue     array of double values which will contain the list of parsed parameters (if found)
//   nmax       maximum size of array; will be set to the actual size at return
//
// Returns:
//   'true' if parameter is found, 'false' otherwise
// 
//////////////////////////

bool parseParameter(parameter * & params, const int numparam, const char * pname, double * pvalue, int & nmax)
{
	const int index = findParameter(params, numparam, pname);
	if (index < 0) { nmax = 0; return false; }
	const int capacity = nmax;
	string value(params[index].value);
	size_t start = 0;
	int n = 0;
	while (true)
	{
		const size_t comma = value.find(',', start);
		const string item = value.substr(start, comma == string::npos ? string::npos : comma-start);
		double parsed;
		if (n >= capacity)
		{
			addParserError(params[index], "list exceeds the fixed capacity for this parameter");
			nmax = 0;
			params[index].used = true;
			return false;
		}
		if (!strictDouble(item.c_str(), parsed))
		{
			addParserError(params[index], "expected a comma-separated list of finite numbers");
			nmax = 0;
			params[index].used = true;
			return false;
		}
		pvalue[n++] = parsed;
		if (comma == string::npos) break;
		start = comma + 1;
	}
	nmax = n;
	params[index].used = true;
	return true;
}


//////////////////////////
// parseParameter (int *)
//////////////////////////
// Description:
//   searches parameter array for specified parameter name and parses it as a list of comma-separated integer values
// 
// Arguments:
//   params     array of parameters
//   numparam   length of parameter array
//   pname      name of parameter to search for
//   pvalue     array of integer values which will contain the list of parsed parameters (if found)
//   nmax       maximum size of array; will be set to the actual size at return
//
// Returns:
//   'true' if parameter is found, 'false' otherwise
// 
//////////////////////////

bool parseParameter(parameter * & params, const int numparam, const char * pname, int * pvalue, int & nmax)
{
	const int index = findParameter(params, numparam, pname);
	if (index < 0) { nmax = 0; return false; }
	const int capacity = nmax;
	string value(params[index].value);
	size_t start = 0;
	int n = 0;
	while (true)
	{
		const size_t comma = value.find(',', start);
		const string item = value.substr(start, comma == string::npos ? string::npos : comma-start);
		long parsed;
		if (n >= capacity)
		{
			addParserError(params[index], "list exceeds the fixed capacity for this parameter");
			nmax = 0;
			params[index].used = true;
			return false;
		}
		if (!strictInteger(item.c_str(), parsed) || parsed < INT_MIN || parsed > INT_MAX)
		{
			addParserError(params[index], "expected a comma-separated list of integers");
			nmax = 0;
			params[index].used = true;
			return false;
		}
		pvalue[n++] = (int) parsed;
		if (comma == string::npos) break;
		start = comma + 1;
	}
	nmax = n;
	params[index].used = true;
	return true;
}


//////////////////////////
// parseParameter (char **)
//////////////////////////
// Description:
//   searches parameter array for specified parameter name and parses it as a list of comma-separated strings
// 
// Arguments:
//   params     array of parameters
//   numparam   length of parameter array
//   pname      name of parameter to search for
//   pvalue     array of character strings which will contain the list of parsed parameters (if found)
//   nmax       maximum size of array; will be set to the actual size at return
//
// Returns:
//   'true' if parameter is found, 'false' otherwise
// 
//////////////////////////

bool parseParameter(parameter * & params, const int numparam, const char * pname, char ** pvalue, int & nmax)
{
	const int index = findParameter(params, numparam, pname);
	if (index < 0) { nmax = 0; return false; }
	const int capacity = nmax;
	string value(params[index].value);
	size_t start = 0;
	int n = 0;
	while (true)
	{
		const size_t comma = value.find(',', start);
		string item = value.substr(start, comma == string::npos ? string::npos : comma-start);
		const size_t first = item.find_first_not_of(" \t");
		const size_t last = item.find_last_not_of(" \t");
		if (n >= capacity)
		{
			addParserError(params[index], "list exceeds the fixed capacity for this parameter");
			nmax = 0;
			params[index].used = true;
			return false;
		}
		if (first == string::npos)
		{
			addParserError(params[index], "list contains an empty item");
			nmax = 0;
			params[index].used = true;
			return false;
		}
		item = item.substr(first, last-first+1);
		if (item.size() >= PARAM_MAX_LENGTH)
		{
			addParserError(params[index], "list item exceeds PARAM_MAX_LENGTH");
			nmax = 0;
			params[index].used = true;
			return false;
		}
		strcpy(pvalue[n++], item.c_str());
		if (comma == string::npos) break;
		start = comma + 1;
	}
	nmax = n;
	params[index].used = true;
	return true;
}


//////////////////////////
// parseFieldSpecifiers
//////////////////////////
bool knownFieldSpecifier(const char * item)
{
	return strcmp(item, "Phi") == 0 || strcmp(item, "phi") == 0
		|| strcmp(item, "Chi") == 0 || strcmp(item, "chi") == 0
		|| strcmp(item, "Pot") == 0 || strcmp(item, "pot") == 0 || strcmp(item, "Psi_N") == 0 || strcmp(item, "psi_N") == 0 || strcmp(item, "PsiN") == 0 || strcmp(item, "psiN") == 0
		|| strcmp(item, "B") == 0 || strcmp(item, "Bi") == 0
		|| strcmp(item, "P") == 0 || strcmp(item, "p") == 0
		|| strcmp(item, "T00") == 0 || strcmp(item, "rho") == 0
		|| strcmp(item, "Tij") == 0
		|| strcmp(item, "rho_N") == 0 || strcmp(item, "rhoN") == 0
		|| strcmp(item, "hij") == 0 || strcmp(item, "GW") == 0
		|| strcmp(item, "Gadget") == 0 || strcmp(item, "Gadget2") == 0 || strcmp(item, "gadget") == 0 || strcmp(item, "gadget2") == 0
		|| strcmp(item, "multi-Gadget") == 0 || strcmp(item, "multi-Gadget2") == 0 || strcmp(item, "multi-gadget") == 0 || strcmp(item, "multi-gadget2") == 0
		|| strcmp(item, "Particles") == 0 || strcmp(item, "particles") == 0 || strcmp(item, "pcls") == 0 || strcmp(item, "part") == 0
		|| strcmp(item, "cross") == 0 || strcmp(item, "X-spectra") == 0 || strcmp(item, "x-spectra") == 0
		|| strcmp(item, "delta") == 0 || strcmp(item, "Ds") == 0 || strcmp(item, "D_s") == 0
		|| strcmp(item, "delta_N") == 0 || strcmp(item, "deltaN") == 0
		|| strcmp(item, "v") == 0 || strcmp(item, "velocity") == 0;
}

// Description:
//   searches parameter array for specified parameter name and parses it as a list of comma-separated field specifiers
// 
// Arguments:
//   params     array of parameters
//   numparam   length of parameter array
//   pname      name of parameter to search for
//   pvalue     integer which will contain the binary-encoded list of parsed specifiers (if found)
//
// Returns:
//   'true' if parameter is found, 'false' otherwise
// 
//////////////////////////

bool parseFieldSpecifiers(parameter * & params, const int numparam, const char * pname, int & pvalue)
{
	char * start;
	char * comma;
	int pos;
	char item[PARAM_MAX_LENGTH];
		   
	for (int i = 0; i < numparam; i++)
	{
		if (strcmp(params[i].name, pname) == 0)
		{
			pvalue = 0;
			start = params[i].value;
			while ((comma = strchr(start, ',')) != NULL)
			{
				strncpy(item, start, comma-start);
				for (pos = comma-start; pos > 0; pos--)
				{
					if (item[pos-1] != ' ' && item[pos-1] != '\t') break;
				}
				item[pos] = '\0';
				if (!knownFieldSpecifier(item))
					addParserDiagnostic(false, params[i].line, params[i].name, item, "unknown output specifier will be ignored");
				
				if (strcmp(item, "Phi") == 0 || strcmp(item, "phi") == 0)
					pvalue |= MASK_PHI;
				else if (strcmp(item, "Chi") == 0 || strcmp(item, "chi") == 0)
					pvalue |= MASK_CHI;
				else if (strcmp(item, "Pot") == 0 || strcmp(item, "pot") == 0 || strcmp(item, "Psi_N") == 0 || strcmp(item, "psi_N") == 0 || strcmp(item, "PsiN") == 0 || strcmp(item, "psiN") == 0)
					pvalue |= MASK_POT;
				else if (strcmp(item, "B") == 0 || strcmp(item, "Bi") == 0)
					pvalue |= MASK_B;
				else if (strcmp(item, "P") == 0 || strcmp(item, "p") == 0)
					pvalue |= MASK_P;
				else if (strcmp(item, "T00") == 0 || strcmp(item, "rho") == 0)
					pvalue |= MASK_T00;
				else if (strcmp(item, "Tij") == 0)
					pvalue |= MASK_TIJ;
				else if (strcmp(item, "rho_N") == 0 || strcmp(item, "rhoN") == 0)
					pvalue |= MASK_RBARE;
				else if (strcmp(item, "hij") == 0 || strcmp(item, "GW") == 0)
					pvalue |= MASK_HIJ;
				else if (strcmp(item, "Gadget") == 0 || strcmp(item, "Gadget2") == 0 || strcmp(item, "gadget") == 0 || strcmp(item, "gadget2") == 0)
					pvalue |= MASK_GADGET;
				else if (strcmp(item, "multi-Gadget") == 0 || strcmp(item, "multi-Gadget2") == 0 || strcmp(item, "multi-gadget") == 0 || strcmp(item, "multi-gadget2") == 0)
					pvalue |= MASK_GADGET | MASK_MULTI;
				else if (strcmp(item, "Particles") == 0 || strcmp(item, "particles") == 0 || strcmp(item, "pcls") == 0 || strcmp(item, "part") == 0)
					pvalue |= MASK_PCLS;
				else if (strcmp(item, "cross") == 0 || strcmp(item, "X-spectra") == 0 || strcmp(item, "x-spectra") == 0)
					pvalue |= MASK_XSPEC;
				else if (strcmp(item, "delta") == 0 || strcmp(item, "Ds") == 0 || strcmp(item, "D_s") == 0)
					pvalue |= MASK_DELTA;
				else if (strcmp(item, "delta_N") == 0 || strcmp(item, "deltaN") == 0)
					pvalue |= MASK_DBARE;
				else if (strcmp(item, "v") == 0 || strcmp(item, "velocity") == 0)
					pvalue |= MASK_VEL;
					
				start = comma+1;
				while (*start == ' ' || *start == '\t') start++;
			}  

			if (!knownFieldSpecifier(start))
				addParserDiagnostic(false, params[i].line, params[i].name, start, "unknown output specifier will be ignored");
			
			if (strcmp(start, "Phi") == 0 || strcmp(start, "phi") == 0)
				pvalue |= MASK_PHI;
			else if (strcmp(start, "Chi") == 0 || strcmp(start, "chi") == 0)
				pvalue |= MASK_CHI;
			else if (strcmp(start, "Pot") == 0 || strcmp(start, "pot") == 0 || strcmp(start, "Psi_N") == 0 || strcmp(start, "psi_N") == 0 || strcmp(start, "PsiN") == 0 || strcmp(start, "psiN") == 0)
				pvalue |= MASK_POT;
			else if (strcmp(start, "B") == 0 || strcmp(start, "Bi") == 0)
				pvalue |= MASK_B;
			else if (strcmp(start, "P") == 0 || strcmp(start, "p") == 0)
				pvalue |= MASK_P;
			else if (strcmp(start, "T00") == 0 || strcmp(start, "rho") == 0)
				pvalue |= MASK_T00;
			else if (strcmp(start, "Tij") == 0)
				pvalue |= MASK_TIJ;
			else if (strcmp(start, "rho_N") == 0 || strcmp(start, "rhoN") == 0)
				pvalue |= MASK_RBARE;
			else if (strcmp(start, "hij") == 0 || strcmp(start, "GW") == 0)
				pvalue |= MASK_HIJ;
			else if (strcmp(start, "Gadget") == 0 || strcmp(start, "Gadget2") == 0 || strcmp(start, "gadget") == 0 || strcmp(start, "gadget2") == 0)
				pvalue |= MASK_GADGET;
			else if (strcmp(start, "multi-Gadget") == 0 || strcmp(start, "multi-Gadget2") == 0 || strcmp(start, "multi-gadget") == 0 || strcmp(start, "multi-gadget2") == 0)
				pvalue |= MASK_GADGET | MASK_MULTI;
			else if (strcmp(start, "Particles") == 0 || strcmp(start, "particles") == 0 || strcmp(start, "pcls") == 0 || strcmp(start, "part") == 0)
				pvalue |= MASK_PCLS;
			else if (strcmp(start, "cross") == 0 || strcmp(start, "X-spectra") == 0 || strcmp(start, "x-spectra") == 0)
				pvalue |= MASK_XSPEC;
			else if (strcmp(start, "delta") == 0 || strcmp(start, "Ds") == 0 || strcmp(start, "D_s") == 0)
				pvalue |= MASK_DELTA;
			else if (strcmp(start, "delta_N") == 0 || strcmp(start, "deltaN") == 0)
				pvalue |= MASK_DBARE;
			else if (strcmp(start, "v") == 0 || strcmp(start, "velocity") == 0)
					pvalue |= MASK_VEL;
			
			params[i].used = true;
			return true;
		}
	}
	
	return false;
}


//////////////////////////
// parseMetadata
//////////////////////////
// Description:
//   parses all metadata from the parameter array
// 
// Arguments:
//   params     array of parameters
//   numparam   length of parameter array
//   sim        reference to metadata stucture (holds simulation parameters)
//   cosmo      reference to cosmology structure (holds cosmological parameters)
//   ic         reference to icsettings structure (holds settings for IC generation)
//
// Returns:
//   number of parameters parsed
// 
//////////////////////////

#ifndef LATFIELD2_HPP
#define COUT cout
#endif

int parseMetadata(parameter * & params, const int numparam, metadata & sim, cosmology & cosmo, icsettings & ic)
{
	char par_string[PARAM_MAX_LENGTH];
	char * pptr[MAX_PCL_SPECIES];
	int usedparams = 0;
	int i;
	double tmp;

	// Gravity is needed by some IC validation before the main metadata block.
	sim.gr_flag = 1;
	const int gravity_index = findParameter(params, numparam, "gravity theory");
	if (gravity_index >= 0 && (params[gravity_index].value[0] == 'N' || params[gravity_index].value[0] == 'n'))
		sim.gr_flag = 0;

	// parse settings for IC generator
	
	ic.pkfile[0] = '\0';
	ic.tkfile[0] = '\0';
	ic.metricfile[0][0] = '\0';
	ic.metricfile[1][0] = '\0';
	ic.metricfile[2][0] = '\0';
	ic.seed = 0;
	ic.flags = 0;
	ic.z_ic = -2.;
	ic.z_relax = -2.;
	ic.Cf = -1.0;
	ic.A_s = P_SPECTRAL_AMP;
	ic.n_s = P_SPECTRAL_INDEX;
	ic.k_pivot = P_PIVOT_SCALE;
	ic.restart_cycle = -1;
	ic.restart_tau = 0.;
	ic.restart_dtau = 0.;
	ic.restart_version = -1.;
	
	parseParameter(params, numparam, "seed", ic.seed);
	
	if (parseParameter(params, numparam, "IC generator", par_string))
	{
		if (par_string[0] == 'B' || par_string[0] == 'b')
			ic.generator = ICGEN_BASIC;
		else if ((par_string[0] == 'R' || par_string[0] == 'r') && par_string[2] != 'L' && par_string[2] != 'l')
			ic.generator = ICGEN_READ_FROM_DISK;
		else if (par_string[0] == 'E' || par_string[0] == 'e')
		{
			ic.generator = ICGEN_READ_FROM_DISK;
			ic.flags |= ICFLAG_EXPRESSREADER;
		}
		else if (par_string[0] == 'P' || par_string[0] == 'p')
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": IC generator = prevolution is not included in gevolution 2.0." << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
#ifdef ICGEN_SONG
		else if (par_string[0] == 'S' || par_string[0] == 's')
			ic.generator = ICGEN_SONG;
#endif
#ifdef ICGEN_RELIC
		else if ((par_string[0] == 'R' || par_string[0] == 'r') && (par_string[2] == 'L' || par_string[2] == 'l'))
			ic.generator = ICGEN_RELIC;
#endif
		else if (par_string[0] == 'F' || par_string[0] == 'f')
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": IC generator = FalconIC is not included in gevolution 2.0." << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
		else
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": IC generator not recognized!" << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
	}
	else
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": IC generator not specified, selecting default (basic)" << endl;
		ic.generator = ICGEN_BASIC;
	}
	
	for (i = 0; i < MAX_PCL_SPECIES; i++)
		pptr[i] = ic.pclfile[i];

	if (ic.generator == ICGEN_READ_FROM_DISK)
	{
		if (!parseParameter(params, numparam, "particle file", pptr, i))
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no particle file specified!" << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
	}
	else if (!parseParameter(params, numparam, "template file", pptr, i)
	#ifdef ICGEN_SONG
		&& ic.generator != ICGEN_SONG
	#endif
	#ifdef ICGEN_RELIC
		&& ic.generator != ICGEN_RELIC
	#endif
	)
	{
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no template file specified!" << endl;
#ifdef LATFIELD2_HPP
		parallel.abortForce();
#endif
	}
	
	for (; i < MAX_PCL_SPECIES; i++)
	{
		if (ic.generator == ICGEN_READ_FROM_DISK)
			strcpy(ic.pclfile[i], "/dev/null");
		else
			strcpy(ic.pclfile[i], ic.pclfile[i-1]);
	}

	if ((!parseParameter(params, numparam, "mPk file", ic.pkfile) && !parseParameter(params, numparam, "Tk file", ic.tkfile)
#ifdef ICGEN_SONG
		&& ic.generator != ICGEN_SONG
#endif
#ifdef ICGEN_RELIC
		&& ic.generator != ICGEN_RELIC
#endif
		&& ic.generator != ICGEN_READ_FROM_DISK))
	{
#ifdef HAVE_CLASS
		COUT << " initial transfer functions will be computed by calling CLASS" << endl;
#else
		COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no power spectrum file nor transfer function file specified!" << endl;
#ifdef LATFIELD2_HPP
		parallel.abortForce();
#endif
#endif
	}
	
	if (parseParameter(params, numparam, "correct displacement", par_string))
	{
		if (par_string[0] == 'Y' || par_string[0] == 'y')
			ic.flags |= ICFLAG_CORRECT_DISPLACEMENT;
		else if (par_string[0] != 'N' && par_string[0] != 'n')
			COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": setting chosen for deconvolve displacement option not recognized, using default (no)" << endl;
	}
	
	if (parseParameter(params, numparam, "k-domain", par_string))
	{
		if (par_string[0] == 'S' || par_string[0] == 's')
			ic.flags |= ICFLAG_KSPHERE;
		else if (par_string[0] != 'C' && par_string[0] != 'c')
			COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": setting chosen for k-domain option not recognized, using default (cube)" << endl;
	}
	
	for (i = 0; i < MAX_PCL_SPECIES; i++)
		ic.numtile[i] = 0;
	
	if(!parseParameter(params, numparam, "tiling factor", ic.numtile, i) && ic.generator != ICGEN_READ_FROM_DISK)
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": tiling factor not specified, using default value for all species (1)" << endl;
		ic.numtile[0] = 1;
		i = 1;
	}
	
	for (; i < MAX_PCL_SPECIES; i++)
		ic.numtile[i] = ic.numtile[i-1];

	if (ic.numtile[0] <= 0 && ic.generator != ICGEN_READ_FROM_DISK)
	{
		addParameterError(params, numparam, "tiling factor", "CDM tiling factor must be greater than zero");
		ic.numtile[0] = 1;
	}
	
	for (i = 1; i < MAX_PCL_SPECIES; i++)
	{
		if (ic.numtile[i] < 0)
		{
			addParameterError(params, numparam, "tiling factor", "tiling factors must not be negative");
			ic.numtile[i] = 0;
		}
		else if (ic.generator == ICGEN_READ_FROM_DISK && strcmp(ic.pclfile[i], "/dev/null") != 0)
			ic.numtile[i] = 1;
	}
	
	if (ic.pkfile[0] != '\0')
	{
		sim.baryon_flag = 0;
		if (parseParameter(params, numparam, "baryon treatment", par_string))
		{
			if (par_string[0] != 'i' && par_string[0] != 'I')
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": using mPk file forces baryon treatment = ignore" << endl;
			}
		}
	}
	else if (parseParameter(params, numparam, "baryon treatment", par_string))
	{
		if (par_string[0] == 'i' || par_string[0] == 'I')
		{
			COUT << " baryon treatment set to: " << COLORTEXT_CYAN << "ignore" << COLORTEXT_RESET << endl;
			sim.baryon_flag = 0;
		}
		else if (par_string[0] == 's' || par_string[0] == 'S')
		{
			COUT << " baryon treatment set to: " << COLORTEXT_CYAN << "sample" << COLORTEXT_RESET << endl;
			sim.baryon_flag = 1;
		}
		else if (par_string[0] == 'b' || par_string[0] == 'B')
		{
			COUT << " baryon treatment set to: " << COLORTEXT_CYAN << "blend" << COLORTEXT_RESET << endl;
			sim.baryon_flag = 2;
		}
		else if (par_string[0] == 'h' || par_string[0] == 'H')
		{
			COUT << " baryon treatment set to: " << COLORTEXT_CYAN << "hybrid" << COLORTEXT_RESET << endl;
			sim.baryon_flag = 3;
		}
		else
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": baryon treatment not supported!" << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
	}
	else if (ic.generator == ICGEN_READ_FROM_DISK)
	{
		sim.baryon_flag = 0;
	}
	else
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": baryon treatment not specified, using default (blend)" << endl;
		sim.baryon_flag = 2;
	}

	if (sim.baryon_flag == 1 && ic.numtile[1] <= 0 && ic.generator != ICGEN_READ_FROM_DISK)
	{
		addParameterError(params, numparam, "tiling factor", "sampled baryon tiling factor must be greater than zero");
		ic.numtile[1] = 1;
	}
	
	if (parseParameter(params, numparam, "radiation treatment", par_string))
	{
		if (par_string[0] == 'i' || par_string[0] == 'I' || par_string[0] == 'b' || par_string[0] == 'B')
		{
			sim.radiation_flag = 0;
			COUT << " radiation treatment set to: " << COLORTEXT_CYAN << "background" << COLORTEXT_RESET << endl;
		}
#ifdef HAVE_CLASS
		else if (par_string[0] == 'c' || par_string[0] == 'C')
		{
			sim.radiation_flag = 1;
			COUT << " radiation treatment set to: " << COLORTEXT_CYAN << "CLASS" << COLORTEXT_RESET << endl;
			if (ic.pkfile[0] != '\0' || ic.tkfile[0] != '\0')
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": using radiation treatment = CLASS and providing initial power spectra / transfer functions independently" << endl;
				COUT << "              is dangerous! In order to ensure consistency, it is recommended to call CLASS directly." << endl;
			}
		}
		else
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": radiation treatment not supported!" << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
#else
		else
		{
			sim.radiation_flag = 0;
			COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": CLASS is not available, setting radiation treatment = background" << endl;
		}
#endif
	}
	else
			sim.radiation_flag = 0;

	if (parseParameter(params, numparam, "fluid treatment", par_string))
	{
		if (par_string[0] == 'b' || par_string[0] == 'B')
		{
			sim.fluid_flag = 0;
			COUT << " fluid treatment set to: " << COLORTEXT_CYAN << "background" << COLORTEXT_RESET << endl;
		}
#ifdef HAVE_CLASS
		else if (par_string[0] == 'c' || par_string[0] == 'C')
		{
			sim.fluid_flag = 1;
			COUT << " fluid treatment set to: " << COLORTEXT_CYAN << "CLASS" << COLORTEXT_RESET << endl;
			if (ic.pkfile[0] != '\0' || ic.tkfile[0] != '\0')
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": using fluid treatment = CLASS and providing initial power spectra / transfer functions independently" << endl;
				COUT << "              is dangerous! In order to ensure consistency, it is recommended to call CLASS directly." << endl;
			}
		}
		else
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": fluid treatment not supported!" << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
#else
		else
		{
			sim.fluid_flag = 0;
			COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": CLASS is not available, setting fluid treatment = background" << endl;
		}
#endif
	}
	else
			sim.fluid_flag = 0;
	
	parseParameter(params, numparam, "relaxation redshift", ic.z_relax);

	if (ic.generator == ICGEN_READ_FROM_DISK)
	{
		parseParameter(params, numparam, "restart redshift", ic.z_ic);
		if(!parseParameter(params, numparam, "cycle", ic.restart_cycle))
		{
			if (sim.radiation_flag)
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": using radiation treatment = CLASS and IC generator = read from disk does not guarantee" << endl;
				COUT << "              that the realization of the radiation perturbations is consistent!" << endl;
			}
		}
		parseParameter(params, numparam, "tau", ic.restart_tau);
		parseParameter(params, numparam, "dtau", ic.restart_dtau);
		for (i = 0; i < 3; i++)
			pptr[i] = ic.metricfile[i];
		parseParameter(params, numparam, "metric file", pptr, i);
		if (parseParameter(params, numparam, "gevolution version", ic.restart_version))
		{
			if (ic.restart_version - GEVOLUTION_VERSION > 0.0001)
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": version number of settings file (" << ic.restart_version << ") is higher than version of executable (" << GEVOLUTION_VERSION << ")!" << endl;
			}
		}
	}
#ifdef ICGEN_RELIC
	else if (ic.generator == ICGEN_RELIC)
	{
		for (i = 0; i < 3; i++)
			pptr[i] = ic.metricfile[i];
		
		if (!parseParameter(params, numparam, "metric file", pptr, i) && (sim.gr_flag > 0 || sim.radiation_flag > 0 || sim.fluid_flag > 0))
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no metric file specified for IC generator = RELIC" << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}

		for (i = 0; i < 2; i++)
			pptr[i] = ic.densityfile[i];

		if (!parseParameter(params, numparam, "density file", pptr, i))
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no density file specified for IC generator = RELIC" << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
		
		for (i = 0; i < 2; i++)
			pptr[i] = ic.velocityfile[i];

		if (!parseParameter(params, numparam, "velocity file", pptr, i))
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": no velocity file specified for IC generator = RELIC" << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
	}
#endif

	if (!parseParameter(params, numparam, "A_s", ic.A_s) && (
		sim.radiation_flag > 0 || (ic.pkfile[0] == '\0' && ic.generator != ICGEN_READ_FROM_DISK)))
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": power spectrum normalization not specified, using default value (2.215e-9)" << endl;
	}

	if (!parseParameter(params, numparam, "n_s", ic.n_s) && (	
		sim.radiation_flag > 0 || (ic.pkfile[0] == '\0' && ic.generator != ICGEN_READ_FROM_DISK)))
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": scalar spectral index not specified, using default value (0.9619)" << endl;
	}
	
	if (!parseParameter(params, numparam, "k_pivot", ic.k_pivot) && (
		sim.radiation_flag > 0 || (ic.pkfile[0] == '\0' && ic.generator != ICGEN_READ_FROM_DISK)))
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": pivot scale not specified, using default value (0.05 / Mpc)" << endl;
	}
	
	// parse metadata
	
	sim.numpts = 0;
	sim.downgrade_factor = 1;
	for (i = 0; i < MAX_PCL_SPECIES; i++) sim.numpcl[i] = 0;
	sim.vector_flag = VECTOR_PARABOLIC;
	sim.gr_flag = 0;
	sim.out_pk = 0;
	sim.out_snapshot = 0;
	sim.out_lightcone[0] = 0;
	sim.num_pk = MAX_OUTPUTS;
	sim.numbins = 0;
	sim.num_snapshot = MAX_OUTPUTS;
	sim.num_lightcone = 0;
	sim.num_IDlogs = 0;
	sim.IDlog_mapping[0] = 0;
	sim.num_restart = MAX_OUTPUTS;
	for (i = 0; i < MAX_PCL_SPECIES; i++) sim.tracer_factor[i] = 1;
	sim.Cf = 1.;
	sim.steplimit = 1.;
	sim.boxsize = -1.;
	sim.wallclocklimit = -1.;
	sim.z_in = 0.;

	if (parseParameter(params, numparam, "vector method", par_string))
	{
		if (par_string[0] == 'p' || par_string[0] == 'P')
		{
			COUT << " vector method set to: " << COLORTEXT_CYAN << "parabolic" << COLORTEXT_RESET << endl;
			sim.vector_flag = VECTOR_PARABOLIC;
		}
		else if (par_string[0] == 'e' || par_string[0] == 'E')
		{
			COUT << " vector method set to: " << COLORTEXT_CYAN << "elliptic" << COLORTEXT_RESET << endl;
			sim.vector_flag = VECTOR_ELLIPTIC;
		}
		else
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": vector method not supported!" << endl;
#ifdef LATFIELD2_HPP
			parallel.abortForce();
#endif
		}
	}
	
	if (!parseParameter(params, numparam, "generic file base", sim.basename_generic))
		sim.basename_generic[0] = '\0';
	
	if (!parseParameter(params, numparam, "snapshot file base", sim.basename_snapshot))
		strcpy(sim.basename_snapshot, "snapshot");
		
	if (!parseParameter(params, numparam, "Pk file base", sim.basename_pk))
		strcpy(sim.basename_pk, "pk");

	if (!parseParameter(params, numparam, "lightcone file base", sim.basename_lightcone))
		strcpy(sim.basename_lightcone, "lightcone");
		
	if (!parseParameter(params, numparam, "output path", sim.output_path))
		sim.output_path[0] = '\0';
		
	if (!parseParameter(params, numparam, "hibernation path", sim.restart_path))
		strcpy(sim.restart_path, sim.output_path);
		
	if (!parseParameter(params, numparam, "hibernation file base", sim.basename_restart))
		strcpy(sim.basename_restart, "restart");
		
	parseParameter(params, numparam, "boxsize", sim.boxsize);
	if (sim.boxsize <= 0. || !isfinite(sim.boxsize))
	{
		addParameterError(params, numparam, "boxsize", "must be a finite value greater than zero");
		sim.boxsize = 1.;
	}
	
	parseParameter(params, numparam, "Ngrid", sim.numpts);
	if (sim.numpts < 2)
	{
		addParameterError(params, numparam, "Ngrid", "must be an integer greater than or equal to 2");
		sim.numpts = 2;
	}
	else if ((double) sim.numpts > cbrt((double) LONG_MAX))
	{
		addParameterError(params, numparam, "Ngrid", "Ngrid cubed exceeds the supported mesh-size range");
		sim.numpts = 2;
	}

	if (parseParameter(params, numparam, "downgrade factor", sim.downgrade_factor))
	{
		if (sim.downgrade_factor < 1 || sim.downgrade_factor >= sim.numpts || !isfinite(sim.downgrade_factor))
		{
			addParameterError(params, numparam, "downgrade factor", "must be at least 1 and smaller than Ngrid");
			sim.downgrade_factor = 1;
		}
#ifdef LATFIELD2_HPP
		if (sim.downgrade_factor > 1 && (sim.numpts % parallel.grid_size()[0] || sim.numpts % parallel.grid_size()[1] || (sim.numpts / parallel.grid_size()[0]) % sim.downgrade_factor || (sim.numpts / parallel.grid_size()[1]) % sim.downgrade_factor))
		{
			addParameterError(params, numparam, "downgrade factor", "is incompatible with Ngrid and the selected process layout");
		}
#endif
	}

	parseParameter(params, numparam, "Courant factor", sim.Cf);
	if (!isfinite(sim.Cf) || sim.Cf <= 0.)
	{
		addParameterError(params, numparam, "Courant factor", "must be a finite value greater than zero");
		sim.Cf = 1.;
	}

	if (ic.Cf < 0.) ic.Cf = sim.Cf;
	
	parseParameter(params, numparam, "time step limit", sim.steplimit);
	if (!isfinite(sim.steplimit) || sim.steplimit <= 0.)
	{
		addParameterError(params, numparam, "time step limit", "must be a finite value greater than zero");
		sim.steplimit = 1.;
	}
	
	if (!parseParameter(params, numparam, "move limit", sim.movelimit))
		sim.movelimit = (double) sim.numpts;
	if (!isfinite(sim.movelimit) || sim.movelimit <= 0.)
	{
		addParameterError(params, numparam, "move limit", "must be a finite value greater than zero");
		sim.movelimit = sim.numpts;
	}
	
	if (!parseParameter(params, numparam, "initial redshift", sim.z_in))
	{
		addParameterError(params, numparam, "initial redshift", "is required");
		sim.z_in = 0.;
	}
	else if (!isfinite(sim.z_in) || sim.z_in <= -1.)
	{
		addParameterError(params, numparam, "initial redshift", "must be finite and greater than -1");
		sim.z_in = 0.;
	}
	if (ic.z_relax < -1.) ic.z_relax = sim.z_in;

	if (ic.z_ic < sim.z_in && ic.generator != ICGEN_READ_FROM_DISK) ic.z_ic = sim.z_in;
	
	parseParameter(params, numparam, "snapshot redshifts", sim.z_snapshot, sim.num_snapshot);
	if (sim.num_snapshot > 0)
		qsort((void *) sim.z_snapshot, (size_t) sim.num_snapshot, sizeof(double), sort_descending);
	
	parseParameter(params, numparam, "Pk redshifts", sim.z_pk, sim.num_pk);
	if (sim.num_pk > 0)
		qsort((void *) sim.z_pk, (size_t) sim.num_pk, sizeof(double), sort_descending);
		
	parseParameter(params, numparam, "hibernation redshifts", sim.z_restart, sim.num_restart);
	if (sim.num_restart > 0)
		qsort((void *) sim.z_restart, (size_t) sim.num_restart, sizeof(double), sort_descending);
		
	parseParameter(params, numparam, "hibernation wallclock limit", sim.wallclocklimit);
	
	parseFieldSpecifiers(params, numparam, "lightcone outputs", sim.out_lightcone[0]);	
	parseFieldSpecifiers(params, numparam, "snapshot outputs", sim.out_snapshot);
	parseFieldSpecifiers(params, numparam, "Pk outputs", sim.out_pk);

	if(!parseParameter(params, numparam, "lightcone pixel factor", sim.pixelfactor[0]))
		sim.pixelfactor[0] = 0.5;

	if(!parseParameter(params, numparam, "lightcone shell factor", sim.shellfactor[0]))
		sim.shellfactor[0] = 1.;
	if (!isfinite(sim.shellfactor[0]) || sim.shellfactor[0] <= 0.)
	{
		addParameterError(params, numparam, "lightcone shell factor", "must be a finite value greater than zero");
		sim.shellfactor[0] = 1.;
	}

	if(!parseParameter(params, numparam, "lightcone covering", sim.covering[0]))
		sim.covering[0] = 2. + 4. / sim.Cf / sim.shellfactor[0];
	if (!isfinite(sim.covering[0]) || sim.covering[0] <= 0.)
	{
		addParameterError(params, numparam, "lightcone covering", "must be a finite value greater than zero");
		sim.covering[0] = 2. + 4. / sim.Cf / sim.shellfactor[0];
	}
	if (!isfinite(sim.pixelfactor[0]) || sim.pixelfactor[0] <= 0.)
	{
		addParameterError(params, numparam, "lightcone pixel factor", "must be a finite value greater than zero");
		sim.pixelfactor[0] = 0.5;
	}

	i = 2;
	if(parseParameter(params, numparam, "lightcone Nside", sim.Nside[0], i))
	{
		if (i < 1)
		{
			COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": parsing of lightcone Nside parameter failed, assuming minimum value Nside=2" << endl;
			sim.Nside[0][0] = 2;
		}
		if (i < 2) sim.Nside[0][1] = sim.Nside[0][0];
		if (sim.Nside[0][0] > 16384 || sim.Nside[0][1] > 16384)
		{
			addParameterError(params, numparam, "lightcone Nside", "must not exceed 16384 because the pixel index uses 32-bit storage");
			sim.Nside[0][0] = 2;
			sim.Nside[0][1] = 2;
		}
		if (sim.Nside[0][0] < 2)
		{
			COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone Nside parameter out of bounds, assuming minimum value Nside=2" << endl;
			sim.Nside[0][0] = 2;
		}
		else
		{
			for (i = 2; i < sim.Nside[0][0]; i *= 2);
			if (i != sim.Nside[0][0])
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone Nside parameter was found to be no power of two, assuming nearest value Nside=" << i << endl;
				sim.Nside[0][0] = i;
			}
		}
		if (sim.Nside[0][1] < sim.Nside[0][0])
		{
			COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone Nside parameter out of bounds, assuming minimum value Nside=" << sim.Nside[0][0] << endl;
			sim.Nside[0][1] = sim.Nside[0][0];
		}
		else
		{
			for (i = 2; i < sim.Nside[0][1]; i *= 2);
			if (i != sim.Nside[0][1])
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone Nside parameter was found to be no power of two, assuming nearest value Nside=" << i << endl;
				sim.Nside[0][1] = i;
			}
		}
	}
	else
	{
		sim.Nside[0][0] = 2;
		for (sim.Nside[0][1] = 2; sim.Nside[0][1] < sim.numpts && sim.Nside[0][1] < 16384; sim.Nside[0][1] *= 2);
		if (sim.Nside[0][1] > 16384) sim.Nside[0][1] = 16384;
	}

	for (i = 1; i < MAX_OUTPUTS; i++)
	{
		sim.out_lightcone[i] = sim.out_lightcone[0];
		sim.Nside[i][0] = sim.Nside[0][0];
		sim.Nside[i][1] = sim.Nside[0][1];
		sim.pixelfactor[i] = sim.pixelfactor[0];
		sim.shellfactor[i] = sim.shellfactor[0];
		sim.covering[i] = sim.covering[0];
	}

	i = 3;
	if (parseParameter(params, numparam, "lightcone vertex", sim.lightcone[0].vertex, i))
	{
		if (i != 3 || sim.lightcone[0].vertex[0] < 0. || sim.lightcone[0].vertex[0] >= sim.boxsize || sim.lightcone[0].vertex[1] < 0. || sim.lightcone[0].vertex[1] >= sim.boxsize || sim.lightcone[0].vertex[2] < 0. || sim.lightcone[0].vertex[2] >= sim.boxsize)
		{
			COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": parsing of lightcone geometry failed, no lightcone will be written!" << endl;
		}
		else
		{
			sim.num_lightcone = 1;
			for (i = 0; i < 3; i++) sim.lightcone[0].vertex[i] /= sim.boxsize;

			if (parseParameter(params, numparam, "lightcone opening half-angle", sim.lightcone[0].opening))
			{
				if (sim.lightcone[0].opening > 180. || sim.lightcone[0].opening <= 0.)
				{
					COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone opening half-angle out of bounds, assuming full sky" << endl;
					sim.lightcone[0].opening = -1.;
				}
				else sim.lightcone[0].opening = cos(sim.lightcone[0].opening * M_PI / 180.);
			}
			else
				sim.lightcone[0].opening = -1.;

			i = 2;
			if (parseParameter(params, numparam, "lightcone distance", sim.lightcone[0].distance, i))
			{
				sim.lightcone[0].distance[0] /= sim.boxsize;
				if (i > 1)
				{
					sim.lightcone[0].distance[1] /= sim.boxsize;
					qsort((void *) sim.lightcone[0].distance, 2, sizeof(double), sort_descending);
				}
				else
					sim.lightcone[0].distance[1] = 0.;
			}
			else
			{
				sim.lightcone[0].distance[0] = 0.5;
				sim.lightcone[0].distance[1] = 0.;
			}

			if (!parseParameter(params, numparam, "lightcone redshift", sim.lightcone[0].z))
				sim.lightcone[0].z = 0.;

			i = 3;
			if(parseParameter(params, numparam, "lightcone direction", sim.lightcone[0].direction, i))
			{
				if (i == 2)
				{
					tmp = sin(sim.lightcone[0].direction[0] * M_PI / 180.);
					sim.lightcone[0].direction[2] = cos(sim.lightcone[0].direction[0] * M_PI / 180.);
					sim.lightcone[0].direction[0] = tmp * cos(sim.lightcone[0].direction[1] * M_PI / 180.);
					sim.lightcone[0].direction[1] = tmp * sin(sim.lightcone[0].direction[1] * M_PI / 180.);
				}
				else if (i == 3)
				{
					tmp = sqrt(sim.lightcone[0].direction[0] * sim.lightcone[0].direction[0] + sim.lightcone[0].direction[1] * sim.lightcone[0].direction[1] + sim.lightcone[0].direction[2] * sim.lightcone[0].direction[2]);
					if (!isfinite(tmp) || tmp <= 0.)
					{
						addParameterError(params, numparam, "lightcone direction", "Cartesian direction must have a finite, non-zero norm");
						sim.lightcone[0].direction[0] = 0.;
						sim.lightcone[0].direction[1] = 0.;
						sim.lightcone[0].direction[2] = 1.;
					}
					else
					{
						sim.lightcone[0].direction[0] /= tmp;
						sim.lightcone[0].direction[1] /= tmp;
						sim.lightcone[0].direction[2] /= tmp;
					}
				}
				else
				{
					COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": parsing of lightcone geometry failed, no lightcone will be written!" << endl;
					sim.num_lightcone = 0;
				}
			}
			else
			{
				sim.lightcone[0].direction[0] = 0.;
				sim.lightcone[0].direction[1] = 0.;
				sim.lightcone[0].direction[2] = 1.;
			}
		}
	}
	else
	{
		std::map<std::tuple<double, double, double, double>, int> lightcone_to_IDlog_mapping;

		for (sim.num_lightcone = 0; sim.num_lightcone < MAX_OUTPUTS; sim.num_lightcone++)
		{
			sprintf(par_string, "lightcone %d vertex", sim.num_lightcone);
			i = 3;
			if (parseParameter(params, numparam, par_string, sim.lightcone[sim.num_lightcone].vertex, i))
			{
				if (i != 3 || sim.lightcone[sim.num_lightcone].vertex[0] < 0. || sim.lightcone[sim.num_lightcone].vertex[0] >= sim.boxsize || sim.lightcone[sim.num_lightcone].vertex[1] < 0. || sim.lightcone[sim.num_lightcone].vertex[1] >= sim.boxsize || sim.lightcone[sim.num_lightcone].vertex[2] < 0. || sim.lightcone[sim.num_lightcone].vertex[2] >= sim.boxsize)
				{
					COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": parsing of lightcone geometry failed, not all lightcones will be written!" << endl;
					break;
				}
				else
				{
					for (i = 0; i < 3; i++) sim.lightcone[sim.num_lightcone].vertex[i] /= sim.boxsize;

					sprintf(par_string, "lightcone %d outputs", sim.num_lightcone);
					parseFieldSpecifiers(params, numparam, par_string, sim.out_lightcone[sim.num_lightcone]);

					sprintf(par_string, "lightcone %d opening half-angle", sim.num_lightcone);
					if (parseParameter(params, numparam, par_string, sim.lightcone[sim.num_lightcone].opening))
					{
						if (sim.lightcone[sim.num_lightcone].opening > 180. || sim.lightcone[sim.num_lightcone].opening <= 0.)
						{
							COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone opening half-angle out of bounds, assuming full sky" << endl;
							sim.lightcone[sim.num_lightcone].opening = -1.;
						}
						else sim.lightcone[sim.num_lightcone].opening = cos(sim.lightcone[sim.num_lightcone].opening * M_PI / 180.);
					}
					else
						sim.lightcone[sim.num_lightcone].opening = -1.;
					
					sprintf(par_string, "lightcone %d distance", sim.num_lightcone);
					i = 2;
					if (parseParameter(params, numparam, par_string, sim.lightcone[sim.num_lightcone].distance, i))
					{
						sim.lightcone[sim.num_lightcone].distance[0] /= sim.boxsize;
						if (i > 1)
						{
							sim.lightcone[sim.num_lightcone].distance[1] /= sim.boxsize;
							qsort((void *) sim.lightcone[sim.num_lightcone].distance, 2, sizeof(double), sort_descending);
						}
						else
							sim.lightcone[sim.num_lightcone].distance[1] = 0.;
					}
					else
					{
						sim.lightcone[sim.num_lightcone].distance[0] = 0.5;
						sim.lightcone[sim.num_lightcone].distance[1] = 0.;
					}

					sprintf(par_string, "lightcone %d redshift", sim.num_lightcone);
					if (!parseParameter(params, numparam, par_string, sim.lightcone[sim.num_lightcone].z))
						sim.lightcone[sim.num_lightcone].z = 0.;

					sprintf(par_string, "lightcone %d direction", sim.num_lightcone);
					i = 3;
					if(parseParameter(params, numparam, par_string, sim.lightcone[sim.num_lightcone].direction, i))
					{
						if (i == 2)
						{
							tmp = sin(sim.lightcone[sim.num_lightcone].direction[0] * M_PI / 180.);
							sim.lightcone[sim.num_lightcone].direction[2] = cos(sim.lightcone[sim.num_lightcone].direction[0] * M_PI / 180.);
							sim.lightcone[sim.num_lightcone].direction[0] = tmp * cos(sim.lightcone[sim.num_lightcone].direction[1] * M_PI / 180.);
							sim.lightcone[sim.num_lightcone].direction[1] = tmp * sin(sim.lightcone[sim.num_lightcone].direction[1] * M_PI / 180.);
						}
						else if (i == 3)
						{
							tmp = sqrt(sim.lightcone[sim.num_lightcone].direction[0] * sim.lightcone[sim.num_lightcone].direction[0] + sim.lightcone[sim.num_lightcone].direction[1] * sim.lightcone[sim.num_lightcone].direction[1] + sim.lightcone[sim.num_lightcone].direction[2] * sim.lightcone[sim.num_lightcone].direction[2]);
							if (!isfinite(tmp) || tmp <= 0.)
							{
								addParameterError(params, numparam, par_string, "Cartesian direction must have a finite, non-zero norm");
								sim.lightcone[sim.num_lightcone].direction[0] = 0.;
								sim.lightcone[sim.num_lightcone].direction[1] = 0.;
								sim.lightcone[sim.num_lightcone].direction[2] = 1.;
							}
							else
							{
								sim.lightcone[sim.num_lightcone].direction[0] /= tmp;
								sim.lightcone[sim.num_lightcone].direction[1] /= tmp;
								sim.lightcone[sim.num_lightcone].direction[2] /= tmp;
							}
						}
						else
						{
							COUT << COLORTEXT_RED << " error" << COLORTEXT_RESET << ": parsing of lightcone geometry failed, not all lightcones will be written!" << endl;
							break;
						}
					}
					else
					{
						sim.lightcone[sim.num_lightcone].direction[0] = 0.;
						sim.lightcone[sim.num_lightcone].direction[1] = 0.;
						sim.lightcone[sim.num_lightcone].direction[2] = 1.;
					}
					sprintf(par_string, "lightcone %d pixel factor", sim.num_lightcone);
					parseParameter(params, numparam, par_string, sim.pixelfactor[sim.num_lightcone]);
					sprintf(par_string, "lightcone %d shell factor", sim.num_lightcone);
					parseParameter(params, numparam, par_string, sim.shellfactor[sim.num_lightcone]);
					sprintf(par_string, "lightcone %d covering", sim.num_lightcone);
					parseParameter(params, numparam, par_string, sim.covering[sim.num_lightcone]);
					sprintf(par_string, "lightcone %d Nside", sim.num_lightcone);
					i = 2;
					if(parseParameter(params, numparam, par_string, sim.Nside[sim.num_lightcone], i))
					{
						if (i < 1)
						{
							COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": parsing of lightcone Nside parameter failed, assuming minimum value Nside=2" << endl;
							sim.Nside[sim.num_lightcone][0] = 2;
						}
						if (i < 2) sim.Nside[sim.num_lightcone][1] = sim.Nside[sim.num_lightcone][0];
						if (sim.Nside[sim.num_lightcone][0] > 16384 || sim.Nside[sim.num_lightcone][1] > 16384)
						{
							addParameterError(params, numparam, par_string, "must not exceed 16384 because the pixel index uses 32-bit storage");
							sim.Nside[sim.num_lightcone][0] = 2;
							sim.Nside[sim.num_lightcone][1] = 2;
						}
						if (sim.Nside[sim.num_lightcone][0] < 2)
						{
							COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone Nside parameter out of bounds, assuming minimum value Nside=2" << endl;
							sim.Nside[sim.num_lightcone][0] = 2;
						}
						else
						{
							for (i = 2; i < sim.Nside[sim.num_lightcone][0]; i *= 2);
							if (i != sim.Nside[sim.num_lightcone][0])
							{
								COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone Nside parameter was found to be no power of two, assuming nearest value Nside=" << i << endl;
								sim.Nside[sim.num_lightcone][0] = i;
							}
						}
						if (sim.Nside[sim.num_lightcone][1] < sim.Nside[sim.num_lightcone][0])
						{
							COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone Nside parameter out of bounds, assuming minimum value Nside=" << sim.Nside[sim.num_lightcone][0] << endl;
							sim.Nside[sim.num_lightcone][1] = sim.Nside[sim.num_lightcone][0];
						}
						else
						{
							for (i = 2; i < sim.Nside[sim.num_lightcone][1]; i *= 2);
							if (i != sim.Nside[sim.num_lightcone][1])
							{
								COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": lightcone Nside parameter was found to be no power of two, assuming nearest value Nside=" << i << endl;
								sim.Nside[sim.num_lightcone][1] = i;
							}
						}
					}
				}

				std::tuple<double, double, double, double> lightcone_key(sim.lightcone[sim.num_lightcone].vertex[0], sim.lightcone[sim.num_lightcone].vertex[1], sim.lightcone[sim.num_lightcone].vertex[2], sim.lightcone[sim.num_lightcone].z);

				if (lightcone_to_IDlog_mapping.find(lightcone_key) == lightcone_to_IDlog_mapping.end())
				{
					lightcone_to_IDlog_mapping[lightcone_key] = sim.num_IDlogs;
					sim.IDlog_mapping[sim.num_lightcone] = sim.num_IDlogs;
					sim.num_IDlogs++;
				}
				else
				{
					sim.IDlog_mapping[sim.num_lightcone] = lightcone_to_IDlog_mapping[lightcone_key];
				}
			}
			else break;
		}
	}

	if (sim.num_IDlogs == 0)
	{
		sim.num_IDlogs = 1;
	}
	else
	{
		COUT << " For light cone consistency, particle IDs will be logged for " << sim.num_IDlogs << " distinct observer(s)." << endl;
	}
	
	i = MAX_PCL_SPECIES;
	parseParameter(params, numparam, "tracer factor", sim.tracer_factor, i);
	for (; i > 0; i--)
	{
		if (sim.tracer_factor[i-1] < 0)
		{
			COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": tracer factor not set properly; using default value (1)" << endl;
			sim.tracer_factor[i-1] = 1;
		}
	}
	
	if ((sim.num_snapshot <= 0 || sim.out_snapshot == 0) && (sim.num_pk <= 0 || sim.out_pk == 0) && sim.num_lightcone == 0)
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": no output specified!" << endl;
	}
	
	if (!parseParameter(params, numparam, "Pk bins", sim.numbins))
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": number of Pk bins not set properly; using default value (64)" << endl;
		sim.numbins = 64;
	}
	else if (sim.numbins <= 0)
	{
		addParameterError(params, numparam, "Pk bins", "must be an integer greater than zero");
		sim.numbins = 64;
	}
	
	if (parseParameter(params, numparam, "gravity theory", par_string))
	{
		if (par_string[0] == 'N' || par_string[0] == 'n')
		{
			COUT << " gravity theory set to: " << COLORTEXT_CYAN << "Newtonian" << COLORTEXT_RESET << endl;
			sim.gr_flag = 0;
			if (ic.pkfile[0] == '\0' && ic.tkfile[0] != '\0')
			{
				COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": gauge transformation to N-body gauge can only be performed for the positions; the transformation for" << endl;
				COUT << "              the velocities requires time derivatives of transfer functions. Call CLASS directly to avoid this issue." << endl;
			}
		}
		else if (par_string[0] == 'G' || par_string[0] == 'g')
		{
			COUT << " gravity theory set to: " << COLORTEXT_CYAN << "General Relativity" << COLORTEXT_RESET << endl;
			sim.gr_flag = 1;
		}
		else
		{
			COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": gravity theory unknown, using default (General Relativity)" << endl;
			sim.gr_flag = 1;
		}
	}
	else
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": gravity theory not selected, using default (General Relativity)" << endl;
		sim.gr_flag = 1;
	}
	
	
	// parse cosmological parameters
	
	if (!parseParameter(params, numparam, "h", cosmo.h))
	{
		cosmo.h = P_HUBBLE;
	}
	if (!isfinite(cosmo.h) || cosmo.h <= 0.)
	{
		addParameterError(params, numparam, "h", "must be a finite value greater than zero");
		cosmo.h = P_HUBBLE;
	}
	
	cosmo.num_ncdm = MAX_PCL_SPECIES-2;
	if (!parseParameter(params, numparam, "m_ncdm", cosmo.m_ncdm, cosmo.num_ncdm))
	{
		for (i = 0; i < MAX_PCL_SPECIES-2; i++) cosmo.m_ncdm[i] = 0.;
		cosmo.num_ncdm = 0;
	}
	
	if (parseParameter(params, numparam, "N_ncdm", i))
	{
		if (i < 0)
		{
			addParameterError(params, numparam, "N_ncdm", "must not be negative");
			i = 0;
		}
		if (i > cosmo.num_ncdm)
		{
			addParameterError(params, numparam, "N_ncdm", "must not exceed the number of entries in m_ncdm");
			i = cosmo.num_ncdm;
		}
		cosmo.num_ncdm = i;
	}
	else if (cosmo.num_ncdm > 0)
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": N_ncdm not specified, inferring from number of mass parameters in m_ncdm (" << cosmo.num_ncdm << ")!" << endl;
	}
	
	for (i = 0; i < MAX_PCL_SPECIES-2; i++)
	{
		cosmo.T_ncdm[i] = P_T_NCDM;
		cosmo.deg_ncdm[i] = 1.0;	
	}
	parseParameter(params, numparam, "T_ncdm", cosmo.T_ncdm, i);
	i = MAX_PCL_SPECIES-2;
	parseParameter(params, numparam, "deg_ncdm", cosmo.deg_ncdm, i);
	
	for (i = 0; i < cosmo.num_ncdm; i++)
	{
		if (!isfinite(cosmo.m_ncdm[i]) || cosmo.m_ncdm[i] <= 0.)
		{
			addParameterError(params, numparam, "m_ncdm", "active non-CDM masses must be finite and greater than zero");
			cosmo.m_ncdm[i] = 1.;
		}
		if (!isfinite(cosmo.T_ncdm[i]) || cosmo.T_ncdm[i] <= 0.)
		{
			addParameterError(params, numparam, "T_ncdm", "active non-CDM temperatures must be finite and greater than zero");
			cosmo.T_ncdm[i] = P_T_NCDM;
		}
		if (!isfinite(cosmo.deg_ncdm[i]) || cosmo.deg_ncdm[i] <= 0.)
		{
			addParameterError(params, numparam, "deg_ncdm", "active non-CDM degeneracies must be finite and greater than zero");
			cosmo.deg_ncdm[i] = 1.;
		}
		cosmo.Omega_ncdm[i] = cosmo.m_ncdm[i] * cosmo.deg_ncdm[i] / P_NCDM_MASS_OMEGA / cosmo.h / cosmo.h;
	}
	
	if (parseParameter(params, numparam, "T_cmb", cosmo.Omega_g))
	{
		cosmo.Omega_g = cosmo.Omega_g * cosmo.Omega_g / cosmo.h;
		cosmo.Omega_g = cosmo.Omega_g * cosmo.Omega_g * C_PLANCK_LAW; // Planck's law
	}
	else if (parseParameter(params, numparam, "omega_g", cosmo.Omega_g))
	{
		cosmo.Omega_g /= cosmo.h * cosmo.h;
	}
	else if (!parseParameter(params, numparam, "Omega_g", cosmo.Omega_g))
	{
		cosmo.Omega_g = 0.;
	}
	
	if (parseParameter(params, numparam, "N_ur", cosmo.Omega_ur))
	{
		cosmo.Omega_ur *= (7./8.) * pow(4./11., 4./3.) * cosmo.Omega_g;
	}
	else if (parseParameter(params, numparam, "N_eff", cosmo.Omega_ur))
	{
		cosmo.Omega_ur *= (7./8.) * pow(4./11., 4./3.) * cosmo.Omega_g;
	}
	else if (parseParameter(params, numparam, "omega_ur", cosmo.Omega_ur))
	{
		cosmo.Omega_ur /= cosmo.h * cosmo.h;
	}
	else if (!parseParameter(params, numparam, "Omega_ur", cosmo.Omega_ur))
	{
		cosmo.Omega_ur = P_N_UR * (7./8.) * pow(4./11., 4./3.) * cosmo.Omega_g;
	}
	
	cosmo.Omega_rad = cosmo.Omega_g + cosmo.Omega_ur;

	if (parseParameter(params, numparam, "omega_fld", cosmo.Omega_fld))
		cosmo.Omega_fld /= cosmo.h * cosmo.h;
	else if (!parseParameter(params, numparam, "Omega_fld", cosmo.Omega_fld))
		cosmo.Omega_fld = 0.;
	if (!parseParameter(params, numparam, "w0_fld", cosmo.w0_fld))
		cosmo.w0_fld = -1.;
	if (!parseParameter(params, numparam, "wa_fld", cosmo.wa_fld))
		cosmo.wa_fld = 0.;
	if (!parseParameter(params, numparam, "cs2_fld", cosmo.cs2_fld))
		cosmo.cs2_fld = 1.;

	if (cosmo.Omega_fld > 0 && cosmo.w0_fld == -1.)
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": w0_fld = -1 is singular, setting Omega_fld = 0." << endl;
		cosmo.Omega_fld = 0.;
	}

	if (!parseParameter(params, numparam, "Omega_smg", cosmo.Omega_smg))
		cosmo.Omega_smg = 0.;
	
	if (parseParameter(params, numparam, "omega_b", cosmo.Omega_b))
	{
		cosmo.Omega_b /= cosmo.h * cosmo.h;
	}
	else if (!parseParameter(params, numparam, "Omega_b", cosmo.Omega_b))
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": Omega_b not found in settings file, setting to default (0)." << endl;
		cosmo.Omega_b = 0.;
	}
	
	if (parseParameter(params, numparam, "omega_cdm", cosmo.Omega_cdm))
	{
		cosmo.Omega_cdm /= cosmo.h * cosmo.h;
	}
	else if (!parseParameter(params, numparam, "Omega_cdm", cosmo.Omega_cdm))
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": Omega_cdm not found in settings file, setting to default (1)." << endl;
		cosmo.Omega_cdm = 1.;
	}
	
	cosmo.Omega_m = cosmo.Omega_cdm + cosmo.Omega_b;
	for (i = 0; i < cosmo.num_ncdm; i++) cosmo.Omega_m += cosmo.Omega_ncdm[i];
	
	if (cosmo.Omega_m <= 0. || cosmo.Omega_m > 1.)
	{
		addParserDiagnostic(true, 0, "cosmological densities", NULL, "total matter density must be greater than zero and no larger than one");
		cosmo.Omega_Lambda = 0.;
	}
	else if (cosmo.Omega_rad < 0. || cosmo.Omega_rad > 1. - cosmo.Omega_m)
	{
		addParserDiagnostic(true, 0, "cosmological densities", NULL, "total radiation density is outside the supported range");
		cosmo.Omega_Lambda = 0.;
	}
	else
	{
		COUT << " cosmological parameters are: Omega_m0 = " << cosmo.Omega_m << ", Omega_rad0 = " << cosmo.Omega_rad << ", h = " << cosmo.h << endl;
		cosmo.Omega_Lambda = 1. - cosmo.Omega_m - cosmo.Omega_rad - cosmo.Omega_fld;
	}

	if(!parseParameter(params, numparam, "switch delta_rad", sim.z_switch_deltarad))
		sim.z_switch_deltarad = 0.;

	i = MAX_PCL_SPECIES-2;
	if (!parseParameter(params, numparam, "switch delta_ncdm", sim.z_switch_deltancdm, i))
	{
		for (i = 0; i < MAX_PCL_SPECIES-2; i++)
			sim.z_switch_deltancdm[i] = (ic.numtile[(sim.baryon_flag == 1) ? 2+i : 1+i] > 0) ? sim.z_in : 0.;
	}
	else
	{
		for (; i < MAX_PCL_SPECIES-2; i++)
			sim.z_switch_deltancdm[i] = sim.z_switch_deltancdm[i-1];
	}

	if(!parseParameter(params, numparam, "switch linear chi", sim.z_switch_linearchi))
	{
		if (sim.gr_flag > 0)
			sim.z_switch_linearchi = 0.;
		else
			sim.z_switch_linearchi = 0.011;
			
		for (i = 0; i < cosmo.num_ncdm; i++)
		{
			if (sim.z_switch_linearchi < sim.z_switch_deltancdm[i])
				sim.z_switch_linearchi = sim.z_switch_deltancdm[i];
		}
	}
	else if (sim.gr_flag == 0 && sim.z_switch_linearchi <= 0.01)
	{
		COUT << COLORTEXT_YELLOW << " /!\\ warning" << COLORTEXT_RESET << ": with garavity theory = Newton the switch linear chi redshift must be larger than 0.01." << endl;
		COUT << "              setting switch linear chi = 0.011" << endl;
		sim.z_switch_linearchi = 0.011;
	}

	i = MAX_PCL_SPECIES-2;
	if (!parseParameter(params, numparam, "switch B ncdm", sim.z_switch_Bncdm, i))
	{
		for (i = 0; i < MAX_PCL_SPECIES-2; i++)
			sim.z_switch_Bncdm[i] = sim.z_switch_deltancdm[i];
	}
	for (; i < MAX_PCL_SPECIES-2; i++)
		sim.z_switch_Bncdm[i] = sim.z_switch_Bncdm[i-1];

	if (!isfinite(ic.A_s) || ic.A_s <= 0.)
		addParameterError(params, numparam, "A_s", "must be a finite value greater than zero");
	if (!isfinite(ic.k_pivot) || ic.k_pivot <= 0.)
		addParameterError(params, numparam, "k_pivot", "must be a finite value greater than zero");
	if (!isfinite(ic.n_s))
		addParameterError(params, numparam, "n_s", "must be finite");
	if (!isfinite(ic.z_relax) || ic.z_relax <= -1.)
		addParameterError(params, numparam, "relaxation redshift", "must be finite and greater than -1");
	if (findParameter(params, numparam, "restart redshift") >= 0 && (!isfinite(ic.z_ic) || ic.z_ic <= -1.))
		addParameterError(params, numparam, "restart redshift", "must be finite and greater than -1");

	for (i = 0; i < MAX_PCL_SPECIES; i++)
	{
		if (ic.numtile[i] < 0)
			addParameterError(params, numparam, "tiling factor", "tiling factors must not be negative");
		if ((double) ic.numtile[i] > cbrt((double) LONG_MAX))
			addParameterError(params, numparam, "tiling factor", "tiling-factor cube exceeds the supported particle-count range");
	}

	for (i = 0; i < sim.num_snapshot; i++)
		if (!isfinite(sim.z_snapshot[i]) || sim.z_snapshot[i] <= -1.)
			addParameterError(params, numparam, "snapshot redshifts", "all redshifts must be finite and greater than -1");
	for (i = 0; i < sim.num_pk; i++)
		if (!isfinite(sim.z_pk[i]) || sim.z_pk[i] <= -1.)
			addParameterError(params, numparam, "Pk redshifts", "all redshifts must be finite and greater than -1");
	for (i = 0; i < sim.num_restart; i++)
		if (!isfinite(sim.z_restart[i]) || sim.z_restart[i] <= -1.)
			addParameterError(params, numparam, "hibernation redshifts", "all redshifts must be finite and greater than -1");
	if (!isfinite(sim.z_switch_deltarad) || sim.z_switch_deltarad <= -1.)
		addParameterError(params, numparam, "switch delta_rad", "must be finite and greater than -1");
	if (!isfinite(sim.z_switch_linearchi) || sim.z_switch_linearchi <= -1.)
		addParameterError(params, numparam, "switch linear chi", "must be finite and greater than -1");
	for (i = 0; i < cosmo.num_ncdm; i++)
	{
		if (!isfinite(sim.z_switch_deltancdm[i]) || sim.z_switch_deltancdm[i] <= -1.)
			addParameterError(params, numparam, "switch delta_ncdm", "all active-species redshifts must be finite and greater than -1");
		if (!isfinite(sim.z_switch_Bncdm[i]) || sim.z_switch_Bncdm[i] <= -1.)
			addParameterError(params, numparam, "switch B ncdm", "all active-species redshifts must be finite and greater than -1");
	}

	if (sim.num_restart > 0)
		addParameterError(params, numparam, "hibernation redshifts", "GPU hibernation output is not implemented; continuing would falsely report a checkpoint");
	if (sim.wallclocklimit > 0.)
		addParameterError(params, numparam, "hibernation wallclock limit", "GPU hibernation output is not implemented; reaching this limit would stop without a checkpoint");

	for (i = 0; i < sim.num_lightcone; i++)
	{
		if (!isfinite(sim.lightcone[i].z) || sim.lightcone[i].z <= -1.)
			addParserDiagnostic(true, 0, "lightcone redshift", NULL, "all light-cone redshifts must be finite and greater than -1");
		if (!isfinite(sim.lightcone[i].distance[0]) || !isfinite(sim.lightcone[i].distance[1])
			|| sim.lightcone[i].distance[0] < 0. || sim.lightcone[i].distance[1] < 0.
			|| sim.lightcone[i].distance[0] < sim.lightcone[i].distance[1])
			addParserDiagnostic(true, 0, "lightcone distance", NULL, "distances must be finite, non-negative, and ordered outer-to-inner");
		if (!isfinite(sim.pixelfactor[i]) || sim.pixelfactor[i] <= 0.)
			addParserDiagnostic(true, 0, "lightcone pixel factor", NULL, "must be finite and greater than zero");
		if (!isfinite(sim.shellfactor[i]) || sim.shellfactor[i] <= 0.)
			addParserDiagnostic(true, 0, "lightcone shell factor", NULL, "must be finite and greater than zero");
		if (!isfinite(sim.covering[i]) || sim.covering[i] <= 0.)
			addParserDiagnostic(true, 0, "lightcone covering", NULL, "must be finite and greater than zero");
		if (sim.Nside[i][0] < 1 || sim.Nside[i][1] < sim.Nside[i][0])
			addParserDiagnostic(true, 0, "lightcone Nside", NULL, "values must be positive and the maximum must not be smaller than the minimum");
	}

#ifndef VELOCITY
	if (sim.out_snapshot & MASK_VEL)
	{
		addParameterWarning(params, numparam, "snapshot outputs", "velocity output is unavailable without VELOCITY and will be ignored");
		sim.out_snapshot &= ~MASK_VEL;
	}
	if (sim.out_pk & MASK_VEL)
	{
		addParameterWarning(params, numparam, "Pk outputs", "velocity output is unavailable without VELOCITY and will be ignored");
		sim.out_pk &= ~MASK_VEL;
	}
#endif
#ifndef HAVE_HEALPIX
	for (i = 0; i < sim.num_lightcone; i++)
	{
		const int metric_mask = MASK_PHI | MASK_CHI | MASK_B | MASK_HIJ;
		if (sim.out_lightcone[i] & metric_mask)
		{
			addParserDiagnostic(false, 0, "lightcone outputs", NULL, "metric light-cone output is unavailable without HAVE_HEALPIX and will be ignored");
			sim.out_lightcone[i] &= ~metric_mask;
		}
	}
#endif
	
	for (i = 0; i < numparam; i++)
	{
		if (params[i].used) usedparams++;
		else
			addParserDiagnostic(false, params[i].line, params[i].name, params[i].value, "not used by gevolution; retained for CLASS or another compatible build");
	}
	
	return usedparams;
}

#endif
