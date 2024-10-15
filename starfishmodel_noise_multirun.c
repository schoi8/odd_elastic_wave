#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define numx 30 // number of columns. has to be even.
#define numy 30 // number of rows. has to be even.
#define N numx*numy // number of particles.
#define f0 1.0 // amplitude of transverse force

double getDist(double xi, double xj, double L);
double getPBCval(double x, double L);
double drdt_xl(double dx, double dy, double Fsti, double frepi, double Fstj, double frepj);
double drdt_yl(double dx, double dy, double Fsti, double frepi, double Fstj, double frepj);
double nearfield(double dx, double dy);
double drdt_xt(double dx, double dy, double wi, double wj);
double drdt_yt(double dx, double dy, double wi, double wj);
int* getNeighIdx(int i);

/*
	The model of starfish embryos based on the one from Tan et al. Nature 607, 287-293 (2022).
	We added the self-propulsion of the embryos with circular trajectory as well as noise in the amplitude of the self-propulsion.
	It calculates the Fourier-transform of the current density and then the current correlation functions. 
	The initial position of the particles is located at the triangular lattice with some fluctuations, so the nearest-neighbors of the particles 
	do not change. Therefore, the interaction is calculated only between those 6 nearest neighbors.
*/

int main() {
	clock_t tStart = clock();
	srand((unsigned)time(NULL));

	double R = 0.5; // embryo radius. rescaled to 2R = 1.
	double spacing = 1.2; // spacing scale.
	double r0 = 2 * R * spacing; // perfect crystal lattice spacing

	double pi = acos(-1); // define pi
	double u0 = 0.05; // initial displacement amplitude
	double v0 = 0.01; // amplitude of self-propelling velocity
	double ve = 0.1; // amplitude of noise in v0.
	double vi; // self-propelling velocity amplitude of each particle including the noise effect
	double sig = 1.0; // standard deviation for the white noise

	double Fst0 = 53.7;
	double frep0 = 785.1;
	double w0 = 1.0;

	double tcur = 0.0;
	double dt = 0.001;
	int n_tot = 100000; // total number of time steps
	int n_interval = 100; // time step interval for data recording
	int n_rec = (int)(n_tot / n_interval);

	double Lx = numx * r0; // x length of the system box.
	double Ly = 0.5 * sqrt(3) * numy * r0; // y length of the system box.
	double rm = 3.8 * R; // neighbor detecting radius 

	double dx = 0.5 * spacing;
	double dy = 0.5 * spacing;
	const int xnum = (int)round(Lx / dx);
	const int ynum = (int)round(Ly / dy);

	const int n_j = (int)((n_tot / n_interval) * 0.8);

	int wnum = (int)floor(n_j / 2) + 1;

	double dkx = 2 * pi / Lx; // resolution in kx
	double dky = 2 * pi / Ly; // resolution in ky
	double dw = 2 * pi / (dt * n_interval * n_j); // resolution in w

	int n_simul = 100; // number of simulations to run

	FILE* fp_para = fopen("simulation_para_starfish.txt", "w");
	fprintf(fp_para, "numx = %i,  numy = %i, N = %i \n", numx, numy, N);
	fprintf(fp_para, "n_tot = %i, n_interval = %i, dt = %f \n", n_tot, n_interval, dt);
	fprintf(fp_para, "tmax = n_tot*dt = %f, t_interval = n_interval*dt = %f \n", n_tot * dt, n_interval * dt);
	fprintf(fp_para, "R = %f, spacing = %f, r0 = %f, rm = %f \n", R, spacing, r0, rm);
	fprintf(fp_para, "Fst0 = %f, frep0 = %f, w0 = %f, u0 = %f, v0 = %f \n", Fst0, frep0, w0, u0, v0);
	fprintf(fp_para, "transverse force magnitude f0 = %f \n", f0);
	fprintf(fp_para, "v0 noisy. mean = %f, noise amplitude = %f \n", v0, ve);
	fprintf(fp_para, "number of simulations = %i \n", n_simul);
	fclose(fp_para);

	double xij, yij, rij, xcur0, ycur0;
	double thcur0;
	int n_idx, ix, iy;
	int idx_neigh;

	int endcode = 0;

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

		// current velocity of each particle
		double* vxcur;
		vxcur = (double*)malloc(N * sizeof(double));
		double* vycur;
		vycur = (double*)malloc(N * sizeof(double));

		// angle of self-propelling motion of each particle
		double* thcur;
		thcur = (double*)malloc(N * sizeof(double));
		
		// noise part of v0 term
		double* vxnoise;
		vxnoise = (double*)malloc(N * sizeof(double));
		double* vynoise;
		vynoise = (double*)malloc(N * sizeof(double));
		
		// initial condition
		for (int i = 0; i < N; i++) {
			xcur[i] = xlatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
			ycur[i] = ylatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
			thcur0 = 2 * pi * (double)rand() / (double)RAND_MAX;
			thcur[i] = thcur0;
		}
		
		// current matrix
		double* jxmat;
		jxmat = (double*)calloc(xnum * ynum * n_j, sizeof(double));
		double* jymat;
		jymat = (double*)calloc(xnum * ynum * n_j, sizeof(double));

		tcur = 0.0;

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

					if (rij < 2 * R) { // check for the overlap between particles as they have a finite radius.
						printf("overlap happened at time %f which is time step %i \n", tcur, n);
						printf("happend at %i th simulation out of total %i runs intended \n", nsim, n_simul);
						printf("overlap between %d and %d \n", i, j);
						endcode = 1;
						printf("x values are %f and %f \n", xcur[i], xcur[idx_neigh]);
						printf("y values are %f and %f \n", ycur[i], ycur[idx_neigh]);
						printf("calculated xdis %f \n", xij);
						printf("calculated ydis %f \n", yij);
						printf("Time taken: %.7fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
						break;
					}
					else if (rij < rm) { // within the distance in which the interaction happens
						vxcur[i] += drdt_xl(xij, yij, Fst0, frep0, Fst0, frep0) + drdt_xt(xij, yij, w0, w0);
						vycur[i] += drdt_yl(xij, yij, Fst0, frep0, Fst0, frep0) + drdt_yt(xij, yij, w0, w0);
					}
				}
				free(neighvec);
				if (endcode == 1) {
					break;
				}
			}
			if (endcode == 1) {
				break;
			}

			// update variables
			for (int i = 0; i < N; i++) {
				vi = ve * gsl_ran_gaussian(gr, sig) + v0; // noisy self-propulsion

				vxnoise[i] = (vi - v0) * cos(thcur[i]);
				vynoise[i] = (vi - v0) * sin(thcur[i]);

				vxcur[i] += vi * cos(thcur[i]);
				vycur[i] += vi * sin(thcur[i]);

				xcur0 = xcur[i] + dt * vxcur[i];
				xcur[i] = getPBCval(xcur0, Lx);

				ycur0 = ycur[i] + dt * vycur[i];
				ycur[i] = getPBCval(ycur0, Ly);

				thcur0 = thcur[i] + dt * w0;
				thcur[i] = getPBCval(thcur0, 2 * pi);
			}

			// save the variables
			if (n % n_interval == n_interval - 1) {
				n_idx = (int)floor(n / n_interval);

				if (n_idx >= n_rec - n_j) {
					for (int i = 0; i < N; i++) {
						ix = (int)floor(xcur[i] / dx);
						iy = (int)floor(ycur[i] / dy);

						//jxmat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vxcur[i];
						//jymat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vycur[i];
						jxmat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vxcur[i] - vxnoise[i];
						jymat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vycur[i] - vynoise[i];
						/*
							If we calculate the current correlation function with the noise included in the velocity, we need more statistics 
							to get a clear picture of the current correlation function. 
						*/
					}
				}
			}
			tcur += dt;
		}
		free(xcur);
		free(ycur);
		free(vxcur);
		free(vycur);
		free(thcur);
		free(vxnoise);
		free(vynoise);
		gsl_rng_free(gr); // free the random number generator

		if (endcode == 1) {
			free(jxmat);
			free(jymat);
			exit(0);
		}

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
	fpj1 = fopen("Cll_starfish.dat", "w");
	FILE* fpj2;
	fpj2 = fopen("Clt_r_starfish.dat", "w");
	FILE* fpj3;
	fpj3 = fopen("Clt_i_starfish.dat", "w");
	FILE* fpj4;
	fpj4 = fopen("Ctt_starfish.dat", "w");

	for (int i = 0; i < xnum * ynum * wnum; i++) {
		fwrite(&Cll[i], sizeof(double), 1, fpj1);
		fwrite(&Clt_r[i], sizeof(double), 1, fpj2);
		fwrite(&Clt_i[i], sizeof(double), 1, fpj3);
		fwrite(&Ctt[i], sizeof(double), 1, fpj4);
	}
	fclose(fpj1);
	fclose(fpj2);
	fclose(fpj3);
	fclose(fpj4);

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

// function to calculate longitudinal force
double drdt_xl(double dx, double dy, double Fsti, double frepi, double Fstj, double frepj) {
	double pi = acos(-1.0); // define pi

	double rij = sqrt(dx * dx + dy * dy);
	double rijm = sqrt(dx * dx + dy * dy + 1);

	double Fst = 0.5 * (Fsti + Fstj);
	double frep = 0.5 * (frepi + frepj);

	double vstx = -(Fst / (8 * pi)) * dx / pow(rijm, 3);
	double frepx = (3 / pow(2, 10)) * frep * dx / pow(rij, 14);

	return (vstx + frepx);
}

double drdt_yl(double dx, double dy, double Fsti, double frepi, double Fstj, double frepj) {
	double pi = acos(-1.0); // define pi

	double rij = sqrt(dx * dx + dy * dy);
	double rijm = sqrt(dx * dx + dy * dy + 1);

	double Fst = 0.5 * (Fsti + Fstj);
	double frep = 0.5 * (frepi + frepj);

	double vsty = -(Fst / (8 * pi)) * dy / pow(rijm, 3);
	double frepy = (3 / pow(2, 10)) * frep * dy / pow(rij, 14);

	return (vsty + frepy);
}

// function to calculate transverse force
double nearfield(double dx, double dy) {
	double R = 0.5; // radius of embryo
	double lc = 1.0; // ratio between R and dc, the range for transverse force
	double dc = lc * R;

	double rij = sqrt(dx * dx + dy * dy);
	double dij = rij - 1.0;
	double nearf;

	if (dij < dc) {
		nearf = log(0.5 * lc / dij);
	}
	else {
		nearf = 0.0;
	}

	return nearf;
}

double drdt_xt(double dx, double dy, double wi, double wj) {
	//double f0 = 1.0; // coefficient for transverse force
	double nearf = nearfield(dx, dy);
	double rij = sqrt(dx * dx + dy * dy);
	double fnfx = 0.5 * (wi + wj) * f0 * nearf * dy / rij;

	return fnfx;
}

double drdt_yt(double dx, double dy, double wi, double wj) {
	//double f0 = 1.0; // coefficient for transverse force
	double nearf = nearfield(dx, dy);
	double rij = sqrt(dx * dx + dy * dy);
	double fnfy = -0.5 * (wi + wj) * f0 * nearf * dx / rij;

	return fnfy;
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