

#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include <utility>
#include <limits.h>
#include "header.h"
#include <stdio.h>
#include <stdlib.h>

int process_data(const char* grid_path, const char* q_path, double** input, int* ni_p, int* nj_p, int* nk_p)
{

	int read_nz(FILE * grid);
	int read_dims(FILE * grid, const int nz, int* ni, int* nj, int* nk);
	int read_grid(FILE * grid, const int zone_size, double* x, double* y, double* z);
	int check_q_nz(FILE * qfile, int nz);
	int check_q_dims(FILE * qfile, const int nz, int* ni, int* nj, int* nk);
	int read_q(FILE * qfile, const int zone_size, double* fsmach, double* alpha, double* time, double* reynolds, double* q);
	int read_q(FILE * qfile, const int zone_size, double* fsmach, double* alpha, double* time, double* reynolds, double* q);
	int calc_uv(int zone_size, double* q, double* x, double* y, double* z, double* ans);

	printf("starting...");

	//Declarations
	int n = 0, nz = 0, ierr = 0, grid_size = 0;
	int* ni = NULL, * nj = NULL, * nk = NULL, * ipg = NULL, * zone_size = NULL, * ipq = NULL;

	double fsmach = 0.0, alpha = 0.0, time = 0.0, reynolds = 0.0;

	double* x = NULL, * y = NULL, * z = NULL, * q = NULL, * ans;

	FILE* fpg = NULL, * fpq = NULL;

	FILE* output_file{};

	// Open files
	if (NULL == (fpg = fopen(grid_path, "rb"))) return 5;

	if (NULL == (fpq = fopen(q_path, "rb"))) return 4;

	nz = read_nz(fpg);
	printf("nz = %d\n", nz);

	ni = (int*)malloc(nz * sizeof(int));
	nj = (int*)malloc(nz * sizeof(int));
	nk = (int*)malloc(nz * sizeof(int));
	ipg = (int*)malloc(nz * sizeof(int));
	zone_size = (int*)malloc(nz * sizeof(int));
	ipq = (int*)malloc(nz * sizeof(int));

	ierr = read_dims(fpg, nz, ni, nj, nk);
	ierr = check_q_nz(fpq, nz);
	ierr = check_q_dims(fpq, nz, ni, nj, nk);

	for (n = 0; n < nz; n++)
	{
		printf("n = %d, ni = %d, nj = %d, nk = %d\n", n + 1, ni[n], nj[n], nk[n]);
		zone_size[n] = ni[n] * nj[n] * nk[n];
		if (n == 0)
		{
			ipg[n] = 0;
			ipq[n] = 0;
		}
		else
		{
			ipg[n] = ipg[n - 1] + zone_size[n - 1];
			ipq[n] = ipg[n - 1] + zone_size[n - 1] * 5;
		}
		grid_size += zone_size[n];
	}

	printf("grid size = %d\n", grid_size);


	x = (double*)malloc(grid_size * sizeof(double));
	y = (double*)malloc(grid_size * sizeof(double));
	z = (double*)malloc(grid_size * sizeof(double));
	q = (double*)malloc(grid_size * 5 * sizeof(double));
	ans = (double*)malloc(grid_size * 5 * sizeof(double));


	for (n = 0; n < nz; n++)
	{
		printf("reading grid zone %d .... ", n);
		read_grid(fpg, zone_size[n], &x[ipg[n]], &y[ipg[n]], &z[ipg[n]]);
	}

	for (n = 0; n < nz; n++)
	{
		printf("reading q zone %d .... ", n);
		read_q(fpq, zone_size[n], &fsmach, &alpha, &time, &reynolds, &q[ipq[n]]);
		printf("fsmach = %lf, alpha = %lf, reynolds = %lf, time = %lf ... ", fsmach, alpha, reynolds, time);
		printf("done \n");
	}
	for (n = 0; n < nz; n++)
	{
		calc_uv(zone_size[n], &q[ipq[n]], &x[ipg[n]], &y[ipg[n]], &z[ipg[n]], &ans[ipq[n]]);
	}
	*input = ans;
	*ni_p = ni[0] + 1;
	*nj_p = nj[0] + 1;
	*nk_p = nk[0] + 1;
	return 0;


}
int read_nz(FILE* grid)
{
	int nz;
	fread(&nz, sizeof(int), 1, grid);
	return nz;
}
int read_dims(FILE* grid, const int nz, int* ni, int* nj, int* nk)
{
	int nir = 0, njr = 0, nkr = 0, n = 0;
	for (n = 0; n < nz; n++)
	{
		fread(&nir, sizeof(int), 1, grid);
		fread(&njr, sizeof(int), 1, grid);
		fread(&nkr, sizeof(int), 1, grid);
		ni[n] = nir;
		nj[n] = njr;
		nk[n] = nkr;
	}
	return 0;
}
int read_grid(FILE* grid, const int zone_size, double* x, double* y, double* z)
{
	fread(x, sizeof(double), zone_size, grid);
	fread(y, sizeof(double), zone_size, grid);
	fread(z, sizeof(double), zone_size, grid);

	printf("done \n");

	return 0;
}
int check_q_nz(FILE* qfile, int nz)
{
	int nzq;
	fread(&nzq, sizeof(int), 1, qfile);
	if (nz == nzq)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}
int check_q_dims(FILE* qfile, const int nz, int* ni, int* nj, int* nk)
{
	int nir = 0, njr = 0, nkr = 0, n = 0;
	for (n = 0; n < nz; n++)
	{
		fread(&nir, sizeof(int), 1, qfile);
		fread(&njr, sizeof(int), 1, qfile);
		fread(&nkr, sizeof(int), 1, qfile);
		if (ni[n] != nir || nj[n] != njr || nk[n] != nkr)
		{
			return 1;
		}
	}
	return 0;
}
int read_q(FILE* qfile, const int zone_size, double* fsmach, double* alpha, double* time, double* reynolds, double* q)
{
	fread(fsmach, sizeof(double), 1, qfile);
	fread(alpha, sizeof(double), 1, qfile);
	fread(reynolds, sizeof(double), 1, qfile);
	fread(time, sizeof(double), 1, qfile);
	fread(q, sizeof(double), zone_size * 5, qfile);

	return 0;
}
int calc_uv(int zone_size, double* q, double* x, double* y, double* z, double* ans)
{
	int i = 0;
	double u, v, rho, momx, momy;
	for (i = 0; i < zone_size; i++)
	{
		rho = q[i];
		momx = q[i + zone_size];
		momy = q[i + 2 * zone_size];
		u = momx / rho;
		v = momy / rho;
		ans[5 * i] = x[i];
		ans[5 * i + 1] = y[i];
		ans[5 * i + 2] = z[i];
		ans[5 * i + 3] = u;
		ans[5 * i + 4] = v;

	}
	return 0;
}


