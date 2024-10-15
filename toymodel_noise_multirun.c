#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <fftw3.h>

#define numx 30 // number of columns. has to be even for pbc.
#define numy 30 // number of rows. has to be even for pbc.
#define N numx*numy // number of particles.

double getDist(double xi, double xj, double L);
double getPBCval(double x, double L);
int* getNeighIdx(int i);

/*
	Toy model of a Hookean spring system with both longitudinal and transverse interaction with noise.
	The particles start from a lattice position with fluctuations. So their nearest neighbors are pre-set for faster simulation.
	It calculates the Fourier-transform of the current density and then the current correlation function. 
*/

int main() {
	clock_t tStart = clock();
	srand((unsigned)time(NULL));

	double R = 0.5; // embryo radius. rescaled to 2R = 1.
	double spacing = 1.0; // spacing scale.
	double r0 = 2 * R * spacing; // perfect crystal lattice spacing
	double Lx = numx * r0; // x length of the system box.
	double Ly = 0.5 * sqrt(3) * numy * r0; // y length of the system box.

	double pi = acos(-1); // define pi
	double u0 = 0.05; // initial displacement amplitude

	double D0 = 0.0001; // noise amplitude
	double sig = 1.0; // std of the Gaussian random variable

	double dt = 0.001; // timestep size
	int n_tot = 100000; // total number of timesteps
	int n_interval = 100; // timestep interval for data recording
	int n_rec = (int)(n_tot / n_interval); // number of timesteps that get recorded

	double ka = 1.0; // transverse spring constant
	double k = 0.5; // longitudinal spring constant

	double dx = 0.5 * spacing; // x grid size for coarse-graining
	double dy = 0.5 * spacing; // y grid size for coarse-graining
	const int xnum = (int)round(Lx / dx); // number of x grid
	const int ynum = (int)round(Ly / dy); // number of y grid

	const int n_j = (int)((n_tot / n_interval) * 0.8); // timesteps that will get analyzed by Fourier transform

	int wnum = (int)floor(n_j / 2) + 1; // number of w grid
	double dkx = 2 * pi / Lx; // resolution in kx
	double dky = 2 * pi / Ly; // resolution in ky
	double dw = 2 * pi / (dt * n_interval * n_j); // resolution in w

	int n_simul = 100; // number of iterations to be done

	FILE* fp_para = fopen("simulation_para_toy.txt", "w");
	fprintf(fp_para, "numx = %i,  numy = %i, N = %i \n", numx, numy, N);
	fprintf(fp_para, "n_tot = %i, n_interval = %i, dt = %f, n_j = %i \n", n_tot, n_interval, dt, n_j);
	fprintf(fp_para, "tmax = n_tot*dt = %f, t_interval = n_interval*dt = %f \n", n_tot * dt, n_interval * dt);
	fprintf(fp_para, "R = %f, spacing = %f, r0 = %f \n", R, spacing, r0);
	fprintf(fp_para, "k = %f, ka = %f, u0 = %f \n", k, ka, u0);
	fprintf(fp_para, "number of simulations = %i \n", n_simul);
	fprintf(fp_para, "Gaussian white noise with D0 = %f \n", D0);
	fclose(fp_para);

	int n_idx, ix, iy;
	double xij, yij, rij, nxij, nyij, xcur0, ycur0;
	int idx_neigh;

	// perfect triangular lattice
	double* xlatt;
	xlatt = (double*)malloc(N * sizeof(double));
	double* ylatt;
	ylatt = (double*)malloc(N * sizeof(double));

	for (int ix1 = 0; ix1 < (int)(numy / 2); ix1++) {
		for (int ix2 = 0; ix2 < numx; ix2++) {
			xlatt[numx * 2 * ix1 + ix2] = ix2 * r0;
			xlatt[numx * (2 * ix1 + 1) + ix2] = (ix2 + 0.5) * r0;
		}
	}

	for (int iy1 = 0; iy1 < numy; iy1++) {
		for (int iy2 = 0; iy2 < numx; iy2++) {
			ylatt[numx * iy1 + iy2] = 0.5 * sqrt(3) * iy1 * r0;
		}
	}

	// kx and ky range
	double* kxrange;
	kxrange = (double*)malloc(xnum * sizeof(double));
	double* kyrange;
	kyrange = (double*)malloc(ynum * sizeof(double));

	for (int i = 0; i < xnum; i++) {
		kxrange[i] = -pi / dx + dkx * i;
	}
	for (int j = 0; j < ynum; j++) {
		kyrange[j] = -pi / dy + dky * j;
	}

	// w range
	double* wrange;
	wrange = (double*)malloc(wnum * sizeof(double));

	for (int i = 0; i < wnum; i++) {
		wrange[i] = i * dw;
	}
	
	int ki_shift, kj_shift;
	double jxre, jxim, jyre, jyim;
	double qxhat, qyhat, qx, qy;
	double jlre, jlim, jtre, jtim;

	double* Cll; // current correlation Jl*Jl
	Cll = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Clt_r; // current correlation Jl*Jt
	Clt_r = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Clt_i; // current correlation Jt*Jl
	Clt_i = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Ctt; // current correlation Jt*Jt
	Ctt = (double*)calloc(xnum * ynum * wnum, sizeof(double));

	// n_simul iterations of the simulation
	for (int nsim = 0; nsim < n_simul; nsim++) {
		// preparing to generate normal random variables using gsl
		const gsl_rng_type* T_gauss;
		gsl_rng* gr;
		gsl_rng_env_setup();
		T_gauss = gsl_rng_default;
		gr = gsl_rng_alloc(T_gauss);
		gsl_rng_set(gr, time(NULL)); // seed based on the current time
		
		// current position
		double* xcur;
		xcur = (double*)malloc(N * sizeof(double));
		double* ycur;
		ycur = (double*)malloc(N * sizeof(double));

		// velocity of each particle
		double* vxcur;
		vxcur = (double*)malloc(N * sizeof(double));
		double* vycur;
		vycur = (double*)malloc(N * sizeof(double));

		// initial condition
		for (int i = 0; i < N; i++) {
			xcur[i] = xlatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
			ycur[i] = ylatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
		}

		// current matrix
		double* jxmat;
		jxmat = (double*)calloc(xnum * ynum * n_j, sizeof(double));
		double* jymat;
		jymat = (double*)calloc(xnum * ynum * n_j, sizeof(double));

		for (int n = 0; n < n_tot; n++) {
			for (int i = 0; i < N; i++) {
				vxcur[i] = 0.0;
				vycur[i] = 0.0;

				int* neighvec = getNeighIdx(i); // array of neighbor indices

				for (int j = 0; j < 6; j++) {
					idx_neigh = neighvec[j];
					xij = getDist(xcur[i], xcur[idx_neigh], Lx);
					yij = getDist(ycur[i], ycur[idx_neigh], Ly);
					rij = sqrt(xij * xij + yij * yij);
					nxij = xij / rij;
					nyij = yij / rij;

					vxcur[i] += -ka * (rij - r0) * nyij - k * (rij - r0) * nxij;
					vycur[i] += ka * (rij - r0) * nxij - k * (rij - r0) * nyij;
				}
				free(neighvec);
			}
			
			// update variables and save
			for (int i = 0; i < N; i++) {
				xcur0 = xcur[i] + dt * vxcur[i] + sqrt(2 * D0 * dt) * gsl_ran_gaussian(gr, sig); // with translational noise
				xcur[i] = getPBCval(xcur0, Lx);
				ycur0 = ycur[i] + dt * vycur[i] + sqrt(2 * D0 * dt) * gsl_ran_gaussian(gr, sig); // with translational noise
				ycur[i] = getPBCval(ycur0, Ly);
			}

			if (n % n_interval == n_interval - 1) {
				n_idx = (int)floor(n / n_interval);

				if (n_idx >= n_rec - n_j) {
					for (int i = 0; i < N; i++) {
						ix = (int)floor(xcur[i] / dx);
						iy = (int)floor(ycur[i] / dy);

						jxmat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vxcur[i];
						jymat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vycur[i];
					}
				}
			}
		}
		free(xcur);
		free(ycur);
		free(vxcur);
		free(vycur);
		gsl_rng_free(gr); // free the random number generator

		fftw_complex* outjx;
		outjx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * xnum * ynum * wnum);
		fftw_plan pjx;
		pjx = fftw_plan_dft_r2c_3d(xnum, ynum, n_j, jxmat, outjx, FFTW_ESTIMATE);
		fftw_execute(pjx);
		fftw_destroy_plan(pjx);
		free(jxmat);

		fftw_complex* outjy;
		outjy = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * xnum * ynum * wnum);
		fftw_plan pjy;
		pjy = fftw_plan_dft_r2c_3d(xnum, ynum, n_j, jymat, outjy, FFTW_ESTIMATE);
		fftw_execute(pjy);
		fftw_destroy_plan(pjy);
		free(jymat);

		for (int ki = 0; ki < xnum; ki++) {
			// shift the order in kx and ky axis because of the artifact of DFT
			if (ki < (int)(xnum / 2)) {
				ki_shift = ki + (int)(xnum / 2);
			}
			else {
				ki_shift = ki - (int)(xnum / 2);
			}

			for (int kj = 0; kj < ynum; kj++) {
				
				qx = kxrange[ki];
				qy = kyrange[kj];

				if (qx == 0.0 && qy == 0.0) {
					qxhat = 0.0;
					qyhat = 0.0;
				}
				else {
					qxhat = qx / sqrt(qx * qx + qy * qy);
					qyhat = qy / sqrt(qx * qx + qy * qy);
				}
				
				// shift the order in kx and ky axis because of the artifact of DFT
				if (kj < (int)(ynum / 2)) {
					kj_shift = kj + (int)(ynum / 2);
				}
				else {
					kj_shift = kj - (int)(ynum / 2);
				}

				for (int ni = 0; ni < wnum; ni++) {
					jxre = outjx[ki_shift * ynum * wnum + kj_shift * wnum + ni][0]; // real part of Jx
					jxim = outjx[ki_shift * ynum * wnum + kj_shift * wnum + ni][1]; // imaginary part of Jx
					jyre = outjy[ki_shift * ynum * wnum + kj_shift * wnum + ni][0];
					jyim = outjy[ki_shift * ynum * wnum + kj_shift * wnum + ni][1];

					jlre = jxre * qxhat + jyre * qyhat; // longitudinal Jl real part
					jlim = jxim * qxhat + jyim * qyhat; // longitudinal Jl imaginary part
					jtre = jyre * qxhat - jxre * qyhat; // transverse Jt real part
					jtim = jyim * qxhat - jxim * qyhat; // transverse Jt imaginary part

					Cll[ki * ynum * wnum + kj * wnum + ni] += (jlre * jlre + jlim * jlim) / n_simul; // longitudinal current correalation function
					Clt_r[ki * ynum * wnum + kj * wnum + ni] += (jlre * jtre + jlim * jtim) / n_simul; // real part of cross current correlation function
					Clt_i[ki * ynum * wnum + kj * wnum + ni] += (jlre * jtim - jlim * jtre) / n_simul; // imaginary part of cross current correlation function
					Ctt[ki * ynum * wnum + kj * wnum + ni] += (jtre * jtre + jtim * jtim) / n_simul; // transverse current correlation function
				}
			}
		}
		fftw_free(outjx);
		fftw_free(outjy);
	}
	free(xlatt);
	free(ylatt);
	free(kxrange);
	free(kyrange);
	free(wrange);

	FILE* fpj1;
	fpj1 = fopen("Cll_toy.dat", "w");
	FILE* fpj2;
	fpj2 = fopen("Clt_r_toy.dat", "w");
	FILE* fpj3;
	fpj3 = fopen("Clt_i_toy.dat", "w");
	FILE* fpj4;
	fpj4 = fopen("Ctt_toy.dat", "w");

	for (int i = 0; i < xnum * ynum * wnum; i++) {
		fwrite(&Cll[i], sizeof(double), 1, fpj1);
		fwrite(&Clt_r[i], sizeof(double), 1, fpj2);
		fwrite(&Clt_i[i], sizeof(double), 1, fpj3);
		fwrite(&Ctt[i], sizeof(double), 1, fpj4);
	}
	fclose(fpj1);
	fclose(fpj2);
	fclose(fpj3);

	free(Cll);
	free(Clt_r);
	free(Clt_i);
	free(Ctt);

	printf("Simulation done. %i iterations. Time taken: %.7fs\n", n_simul, (double)(clock() - tStart) / CLOCKS_PER_SEC);
	return 0;
}

// function to calculate the distance for pbc
double getDist(double xi, double xj, double L) {
	double dist0;
	if (xi > xj) {
		dist0 = xi - xj;
	}
	else {
		dist0 = xj - xi;
	}

	double dist;
	if (2 * dist0 <= L) {
		dist = xi - xj;
	}
	else {
		if (xi < xj) {
			dist = xi - xj + L;
		}
		else {
			dist = xi - xj - L;
		}
	}

	return dist;
}

// function to get the position in pbc
double getPBCval(double x, double L) {
	double x_pbc;

	if (x < 0.0) {
		x_pbc = x + L;
	}
	else if (x > L) {
		x_pbc = x - L;
	}
	else {
		x_pbc = x;
	}
	return x_pbc;
}
// function to get a vector of indices of neighbors
int* getNeighIdx(int i) {
	int* idxvec; // vector whose elements are the indices of the neighbors of particle i
	idxvec = (int*)malloc(6 * sizeof(int));

	int idx_row = i / numx; // index for the row
	int idx_col = i % numx; // index for the column
	int idx1; // index for neighbor 1
	int idx2;
	int idx3;
	int idx4;
	int idx5;
	int idx6;

	if (idx_row == 0) {
		idx3 = i + numx;
		idx5 = i - numx + N;

		if (idx_col == 0) {
			idx1 = i + 1;
			idx2 = i + numx - 1;
			idx4 = N - 1;
			idx6 = i + 2 * numx - 1;
		}
		else if (idx_col == numx - 1) {
			idx1 = i - numx + 1;
			idx2 = i - 1;
			idx4 = N - 2;
			idx6 = i + numx - 1;
		}
		else {
			idx1 = i + 1;
			idx2 = i - 1;
			idx4 = i - numx - 1 + N;
			idx6 = i + numx - 1;
		}
	}
	else if (idx_row == numy - 1) {
		idx4 = i - numx;
		idx6 = i + numx - N;

		if (idx_col == 0) {
			idx1 = i + 1;
			idx2 = N - 1;
			idx3 = 1;
			idx5 = i - numx + 1;
		}
		else if (idx_col == numx - 1) {
			idx1 = numx * (numy - 1);
			idx2 = i - 1;
			idx3 = 0;
			idx5 = numx * (numy - 2);
		}
		else {
			idx1 = i + 1;
			idx2 = i - 1;
			idx3 = i + numx + 1 - N;
			idx5 = i - numx + 1;
		}
	}
	else {
		if (idx_row % 2 == 0) { // even index of row
			idx3 = i + numx;
			idx5 = i - numx;

			if (idx_col == 0) {
				idx1 = i + 1;
				idx2 = i + numx - 1;
				idx4 = i - 1;
				idx6 = i + 2 * numx - 1;
			}
			else if (idx_col == numx - 1) {
				idx1 = i - numx + 1;
				idx2 = i - 1;
				idx4 = i - numx - 1;
				idx6 = i + numx - 1;
			}
			else {
				idx1 = i + 1;
				idx2 = i - 1;
				idx4 = i - numx - 1;
				idx6 = i + numx - 1;
			}
		}
		else { // odd index of row
			idx4 = i - numx;
			idx6 = i + numx;

			if (idx_col == 0) {
				idx1 = i + 1;
				idx2 = i + numx - 1;
				idx3 = i + numx + 1;
				idx5 = i - numx + 1;
			}
			else if (idx_col == numx - 1) {
				idx1 = i - numx + 1;
				idx2 = i - 1;
				idx3 = i + 1;
				idx5 = i - 2 * numx + 1;
			}
			else {
				idx1 = i + 1;
				idx2 = i - 1;
				idx3 = i + numx + 1;
				idx5 = i - numx + 1;
			}
		}
	}

	idxvec[0] = idx1;
	idxvec[1] = idx2;
	idxvec[2] = idx3;
	idxvec[3] = idx4;
	idxvec[4] = idx5;
	idxvec[5] = idx6;

	return idxvec;
}