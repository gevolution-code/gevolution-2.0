#include <cassert>
#include <fstream>
#include <string>

#define COLORTEXT_RED ""
#define COLORTEXT_YELLOW ""
#define COLORTEXT_CYAN ""
#define COLORTEXT_WHITE ""
#define COLORTEXT_RESET ""

#include "../parser.hpp"

static parameter make_parameter(const char * name, const char * value, const int line)
{
	parameter param;
	strcpy(param.name, name);
	strcpy(param.value, value);
	param.used = false;
	param.line = line;
	return param;
}

int main()
{
	parameter * params = new parameter[4];
	params[0] = make_parameter("integer", "12", 1);
	params[1] = make_parameter("bad integer", "12foo", 2);
	params[2] = make_parameter("finite", "0.5", 3);
	params[3] = make_parameter("not finite", "NaN", 4);

	clearParserDiagnostics();
	int integer = 0;
	double value = 0.;
	assert(parseParameter(params, 4, "integer", integer) && integer == 12);
	assert(!parseParameter(params, 4, "bad integer", integer));
	assert(parseParameter(params, 4, "finite", value) && value == 0.5);
	assert(!parseParameter(params, 4, "not finite", value));
	assert(parserErrorCount() == 2);
	delete[] params;

	params = new parameter[2];
	params[0] = make_parameter("list", "1, 2, 3", 1);
	params[1] = make_parameter("overflow", "1, 2, 3", 2);
	clearParserDiagnostics();
	int list[3];
	int size = 3;
	assert(parseParameter(params, 2, "list", list, size) && size == 3);
	size = 2;
	assert(!parseParameter(params, 2, "overflow", list, size));
	assert(parserErrorCount() == 1);
	delete[] params;

	params = new parameter[1];
	params[0] = make_parameter("snapshot outputs", "phi, unavailable", 1);
	clearParserDiagnostics();
	int outputs = 0;
	assert(parseFieldSpecifiers(params, 1, "snapshot outputs", outputs));
	assert(outputs == MASK_PHI);
	assert(parserErrorCount() == 0);
	assert(parser_diagnostics.size() == 1);
	delete[] params;

	const std::string filename = "/tmp/gevolution_parser_test.ini";
	std::ofstream file(filename.c_str());
	file << "Ngrid = 64\n";
	file << "Ngrid = 128\n";
	file << "bad = \n";
	file << "long = " << std::string(PARAM_MAX_LINESIZE, 'x') << "\n";
	file.close();

	clearParserDiagnostics();
	params = NULL;
	const int count = loadParameterFile(filename.c_str(), params);
	assert(count == 2);
	assert(params[0].line == 1);
	assert(params[1].line == 2);
	assert(parserErrorCount() == 3);
	free(params);

	return 0;
}
