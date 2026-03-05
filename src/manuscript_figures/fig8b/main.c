/*

	code for vae-ibl paper

	- compute xi^* and gap

	next:
	1. translate from gap and slope (which give sigms) to phat
	2. plot optimal phat_r. but maybe naoki can make those?

*/

// ---headers
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string>
#include <string.h>
#include <ctype.h>
using namespace std;

// ---parameters
#define MLL_lib 1048576

// ---structures

struct charplus
{
	char*** x;
	int* m;
	int n;
};

struct floatplus
{
//	float** x;
	// ---data from body of file.
	double** x;
	int* m;
	int n;
};

// ---prototypes
float xistar(float sigma, float p, int nc, float* c);
double phi(double x);
double invphi(double x);
double f(float x, float sigma, int nc, float* c);
floatplus get_cols(FILE* f);
charplus parse_file(FILE* f, int min_elements);
double** newdouble(int n1, int* n2);
void write_err(const char* s1, const char* s2, const char* s3, int n, const char* s4);
int* parse_c(char* buf, int& n);
int write_comline();

int main(int argc, char** argv)
{
	if (argc > 1) if (!strcmp(argv[1], "-h")) write_comline();

	// ---decleration
	float sigma;
	FILE* xis = fopen("out.xi", "w");
	FILE* gap = fopen("out.gap", "w");

	// ---set contrasts
	int nc=5;
	float c[] = {0.0, 0.0625, 0.125, 0.25, 1.0};

	//
	// first, comput xi* and gap versus phat_r
	//

	float pi = 4.0*atan(1.0);

	for (float p=0.5; p < 0.999; p+=0.001)
	{
		fprintf(xis, "%10.6f ", p);
		fprintf(gap, "%10.6f ", p);
		for (int s=1; s < 7; s++)
		{
			sigma = 1/(sqrt(2*pi)*s);
			float xi = xistar(sigma, p, nc, c);
			double x = -xi/sigma;
			fprintf(xis, "%10.6f ", -xi);
			fprintf(gap, "%10.6f ", 2*phi(x)-1.0);
		}
		fprintf(xis, "\n");
		fprintf(gap, "\n");
	}

/*	******************** following commented out ********************
	//
	// ---d gap/d phat_r versus sigma at phat_r=1/2, just to check.
	//    it worked!!!
	//
	for (int s=1; s < 7; s++)
	{
		sigma = 1/(sqrt(2*pi)*s);
		float num=0.0, den=0.0;
		float pref = 4.0*sigma/sqrt(2*pi);
		for (int n=0; n < nc; n++)
		{
			float tmp = (c[n]/sigma)*(c[n]/sigma);
			num += exp(-0.5*tmp);
			den += c[n] * exp(-0.5*tmp);
		}
		cout << sigma << " " << pref * num/den << endl;
	}
	************************** end comments ************************* */

	//
	// second, map slope and gap to sigma and phat_r
	//

/*	can use this to check invphi
	for (float x=0.01; x < 0.99; x += 0.01)
		cout << x << " " << invphi(x) << endl;
*/

	// ---get data
	FILE *fin = stdin;
	FILE* convert = fopen("out.convert", "w");

	struct floatplus z = get_cols(fin);

	for (int i=0; i < z.n; i++)
	{
		float gp = z.x[i][1];
		float sl = z.x[i][2];

		float sigma = 1.0/(sqrt(2*pi)*sl);
		double xi = -sigma*invphi(0.5*(1.0+gp));
		float pr = f(-xi, sigma, nc, c)/(f(xi, sigma, nc, c) + f(-xi, sigma, nc, c));
		fprintf(convert, "%10.6f %10.6f %10.6f\n", z.x[i][0], sigma, pr);

	}

	exit(0);
}

float xistar(float sigma, float p, int nc, float* c)
{
/*	find the solution to the equation

	p f(xi, sigma) = p_l f(-xi, sigma)

	non-convex, so search by cutting interval in half
*/

	// ---easy case
	if (p == 0.5) return 0.0;

	// ---otherwise, gotta search, using bisection method.
	float x0=0, x1;
	x1 = p > 0.5 ? -1.0 : 1.0;

	// ---initialize functions
	float f0, f1;
	f0=p*f(x0, sigma, nc, c) - (1-p)*f(-x0, sigma, nc, c);
	f1=p*f(x1, sigma, nc, c) - (1-p)*f(-x1, sigma, nc, c);

	int cnt=0;
	while ((f0 > 0 && f1 > 0) || (f0 < 0 && f1 < 0))
	{
		x0+= x0 > 0 ? 0.1 : -0.1;
		x1+= x1 > 0 ? 0.1 : -0.1;
		f0=p*f(x0, sigma, nc, c) - (1-p)*f(-x0, sigma, nc, c);
		f1=p*f(x1, sigma, nc, c) - (1-p)*f(-x1, sigma, nc, c);
		if (cnt++ > 1000)
		{
			cerr << "problems with initialization" << endl;
			cerr << "gotta modify code" << endl;
			cerr << x0 << " " << x1 << " " << endl;
			cerr << f0 << " " << f1 << " " << endl << endl;
			exit(1);
		}
	}

	float err=1.0, tol=1.0e-6;
	cnt=0;
	while (err > tol)
	{
		float x = 0.5*(x0+x1);
		float fnew=p*f(x, sigma, nc, c) - (1-p)*f(-x, sigma, nc, c);

		if (fnew*f0 > 0)
		{
			x0=x;
			f0=fnew;
		}
		else
		{
			x1=x;
			f1=fnew;
		}

		err = abs(f0-f1);
		
		if (cnt++ > 1000)
		{
			cerr << "did not converge" << endl;
			cerr << "gotta modify code" << endl;
			cerr << x0 << " " << x1 << " " << endl;
			cerr << f0 << " " << f1 << " " << endl << endl;
			exit(1);
		}
	}
	
	return (x0+x1)/2.0;
}

double invphi(double phi0)
{
/*
	Use Haley's method to compute inverse cumulative normal function
*/
	// ---check that x is in range
	if (phi0 < 0 || phi0 > 1)
	{
		cerr << "phi not in range in call to invphi" << endl;
		exit(1);
	}

	double pi=4.0*atan(1.0), tol=1.e-5;

	// ---itereate
	double x=0;
	for (int n=0; n < 100; n++)
	{
		x -= 2*(phi(x)-phi0)/(2*exp(-x*x/2)/sqrt(2*pi) + x*phi(x));
		if (abs(phi0 - phi(x)) < tol) break;
	}

	return x;
}

double f(float x, float sigma, int nc, float* c)
{
/*
	f(x, sigma), which is needed for psychometric curve

	relative to the definition in the paper, this is f(sigma*x)
*/

        double fout=0;
	double tmp=0.5/(sigma*sigma);
	for (int n=0; n < nc; n++)
		fout += exp(-tmp*(x- c[n])*(x -c[n]));

        return fout/nc;
}

floatplus get_cols(FILE* f)
{
/*
	read from file f, except skips lines whose first element is #.
	call:
		floatplus z = get_cols(stdin);
	or (I believe)
		floatplus z = get_cols(fopen("filename", "r"));

	INPUT:
	f	- input file.

	RETURNS:
	floatplus z:
	z.x[i][j]	- jth element in line i
	z.m[i]		- number of elements on line i
	z.n		- number of lines.  returns 0 if the file
			  does not exist.
*/
	// ---declarations
	struct floatplus z;

	// ---check that file exists.  if not, return with z.n = 0.
	//    reserve space just in case.
	if (!f)
	{
		z.n=0;
		z.m = new int[1];
		z.x = new double*[1];
		z.x[0] = new double[1];
		return z;
	}

	charplus cdat = parse_file(f, 0);

	z.n = cdat.n;
	z.m = (int*) calloc(z.n, sizeof(int));
	for (int i=0; i < z.n; i++) z.m[i] = cdat.m[i];
	z.x = newdouble(z.n, z.m);
	for (int i=0; i < z.n; i++) for (int j=0; j < z.m[i]; j++)
		z.x[i][j] = atof(cdat.x[i][j]);

	// ---cleanup
	fclose(f);
	for (int i=0; i < cdat.n; i++) free(cdat.x[i]);
	free(cdat.x);
	free(cdat.m);

	return z;
}

charplus parse_file(FILE* f, int min_elements)
{
/*
	parse file. ignore lines starting with "#" or empty lines

	INPUT:
	f		- input file.
	min_elements	- only lines with at least min_elements will be saved.

	RETURNS:
	cdat		- cdat.x[i][j] - character in row i, column j.
			  cdat.m[i]    - number of elements in row i.
			  cdat.n       - number of lines.
			  
*/
	// ---count lines. this includes comments, so it will be a
	//    slight overestimate.
	char* line = new char[MLL_lib];
	int lines_total=0;
	while (fgets(line, MLL_lib, f) != NULL) lines_total++;
	rewind(f);

	// ---reserve space
	struct charplus cdat;
	cdat.x = (char***) calloc(lines_total, sizeof(char**));
	cdat.m = (int*) calloc(lines_total, sizeof(int));

	// ---initilize
	int line_no=0, n_elements;
	int lines=0;
	int cnt=0;

	while (fgets(line, MLL_lib, f))
	{
		line_no++;

		// ---parse
		int* nw = parse_c(line, n_elements);

		// check that there is at least one element on the line
		if (n_elements > min_elements && line[nw[0]] != '#')
		{
			cdat.m[lines] = n_elements;
			cdat.x[lines]=(char**)calloc(n_elements, sizeof(char*));

			for (int i=0; i < n_elements; i++)
			{
				cdat.x[lines][i] = (char*)
					calloc(1+strlen(&line[nw[i]]),
					sizeof(char));
				strcpy(cdat.x[lines][i], &line[nw[i]]);
			}
			lines++;
		}

		delete [] nw;
	}
	cdat.n = lines;

	delete [] line;
	return cdat;
}

double** newdouble(int n1, int* n2)
{
	double** x = new double*[n1];
	for (int n=0; n < n1; n++) x[n] = new double[n2[n]];
	return x;
}

void write_err(const char* s1, const char* s2, const char* s3, int n, const char* s4)
{
	fprintf(stderr, "[01;31m");
	fprintf(stderr, "%s%s%s%d%s\n", s1, s2, s3, n, s4);
	fprintf(stderr, "[00m");
	exit(1);
}

int* parse_c(char* buf, int& n)
{
/*
	find all instances of whitespace delimited strings.

	INPUT:
	buf	- character array containing line to be parsed.
		  assumed to end in '\0'.

	OUTPUT:
	buf	- '\0' placed at end of every string.
	n	- number of strings in buf.

	RETURNS:
	nw	- strings are contained in buf[nw[i]], i=0, n-1.
*/
	// ---scratch
	static int* nw_tmp = (int*) calloc(MLL_lib/2, sizeof(int));

	// ---initialize
	n = 0;
	nw_tmp[0] = 0;

	// ---skip past any blank spaces
	int i=0;
	while (isspace(buf[i])) i++;

	// parse
	while (buf[i])
	{
		while (isspace(buf[i])) i++;
		if (buf[i] == '\0') break;

		nw_tmp[n++] = i;

		while (!isspace(buf[i]) && buf[i] != '\0') i++;
		if (buf[i] == '\0') break;
		buf[i++] = '\0';
	}

	int* nw = (int*) calloc((n == 0 ? 1 : n), sizeof(int));
	for (int i=0; i < (n == 0 ? 1 : n); i++) nw[i] = nw_tmp[i];

	return nw;
}

double phi(double x)
{
/*
	Return Phi(x) = cumulative normal function.

	This origiinally computed erfcc(x), defined by

		erfcc(x) = [2/sqrt(pi)] * int_x^inf dy exp(-y*y)

	note that

		int_x^inf dz exp(-z*z/2) / sqrt(2*pi) = (1/2) erfcc(x/sqrt(2))

	thus,

		Phi(x) = 1 - (1/2) erfcc(x/sqrt(2))

	where Phi(x) = cumulative normal function.

	thus, I divide x by sqrt(2) and return 1 (1-ans/2)

	this was copied from numerical recipes.
*/
	double t,z,ans,phi;

	// --turn erfcc into Phi (modulo scaling and translation)
	x/=sqrt(2);

	z=fabs(x);
	t=1.0/(1.0+0.5*z);
	ans=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
		t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
		t*(-0.82215223+t*0.17087277)))))))));
	phi = x >= 0.0 ? ans : 2.0-ans;

	return 1-0.5*phi;
}

int write_comline()
{
char s[80];
strcpy(s, "rpn.TMP_MORE_TMP");
FILE* f = fopen(s, "w");

fprintf(f, "\n\
   command line:  gap < infile\n\
\n\
   does two things:\n\
\n\
   1. computes xi^* and gap versus p for sigma=1/(sqrt(2 pi) n), n=1, ..., 6.\n\
      output = out.xi and out.gap.\n\
\n\
   2. reads from infile and, for each line, finds the corresponding slope\n\
      and prior prob that the grating is on the right.\n\
      output = out.convert.\n\
\n\
   assumes input is in the format:\n\
\n\
      column 1: z-score\n\
      column 2: gap at zero contrast\n\
      column 3: slope of psyhcometric curve at zero contrast\n\
\n\
\n\
   output:\n\
\n\
   out.xi:   xi^* versus phat\n\
      column 1:    phat_r\n\
      columns 2-7: xi^* with sigma = 1/(sqrt(2 pi) (col-1))\n\
\n\
\n\
   out.gap:  gap versus phat\n\
      column 1:    phat_r\n\
      columns 2-7: gap with sigma = 1/(sqrt(2 pi) (col-1))\n\
\n\
\n\
   out.convert:\n\
      column 1: z-score (copied from input)\n\
      column 2: sigma\n\
      column 3: phat_r\n\
      column 4: error in phat_r\n\
\n\
");

fclose(f);

char t[80];
sprintf(t, "more %s", s);
system(t);

sprintf(t, "rm %s", s);
system(t);

exit(1);
}
