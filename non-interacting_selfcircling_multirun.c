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

double getPBCval(double x, double L);

/*
	simulation for non-interacting self-circling particles 
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
	double w0 = 1.0; // self-circling frequency
	double v0 = 0.01; // self-circling velocity
	double l0 = v0 / w0;

	double dt = 0.01; // timestep size
	int n_tot = 40000; // total number of timesteps
	int n_interval = 40; // timestep interval for data recording
	int n_rec = (int)(n_tot / n_interval); // number of timesteps that get recorded

	double dx = 0.3 * spacing; // x grid size for coarse-graining
	double dy = 0.3 * spacing; // y grid size for coarse-graining
	const int xnum = (int)round(Lx / dx); // number of x grid
	const int ynum = (int)round(Ly / dy); // number of y grid

	const int n_j = (int)((n_tot / n_interval) * 0.8); // timesteps that will get analyzed by Fourier transform

	int wnum = (int)floor(n_j / 2) + 1; // number of w grid
	double dkx = 2 * pi / Lx; // resolution in kx
	double dky = 2 * pi / Ly; // resolution in ky
	double dw = 2 * pi / (dt * n_interval * n_j); // resolution in w

	int n_simul = 100; // number of iterations to be done

	FILE* fp_para = fopen("simulation_para_nisc.txt", "w");
	fprintf(fp_para, "numx = %i,  numy = %i, N = %i \n", numx, numy, N);
	fprintf(fp_para, "n_tot = %i, n_interval = %i, dt = %f, n_j = %i \n", n_tot, n_interval, dt, n_j);
	fprintf(fp_para, "tmax = n_tot*dt = %f, t_interval = n_interval*dt = %f \n", n_tot * dt, n_interval * dt);
	fprintf(fp_para, "spacing = %f \n", spacing);
	fprintf(fp_para, "w0 = %f, v0 = %f, l0 = %f \n", w0, v0, l0);
	fprintf(fp_para, "number of simulations = %i \n", n_simul);
	fclose(fp_para);

	int n_idx, ix, iy;
	double xij, yij, rij, nxij, nyij, xcur0, ycur0;
	double thcur0;

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

		// orientation of each particle
		double* thcur;
		thcur = (double*)malloc(N * sizeof(double));

		// initial condition
		for (int i = 0; i < N; i++) {
			thcur[i] = 2 * pi * (double)rand() / (double)RAND_MAX; // random angle between 0 to 2pi
			//thcur[i] = 0.0; // synchronized circling
			xcur[i] = xlatt[i] + l0 * sin(thcur[i]);
			ycur[i] = ylatt[i] - l0 * cos(thcur[i]);
		}

		// current matrix
		double* jxmat;
		jxmat = (double*)calloc(xnum * ynum * n_j, sizeof(double));
		double* jymat;
		jymat = (double*)calloc(xnum * ynum * n_j, sizeof(double));

		for (int n = 0; n < n_tot; n++) {
			for (int i = 0; i < N; i++) {
				thcur0 = thcur[i] + w0 * dt;
				thcur[i] = getPBCval(thcur0, 2 * pi);
				vxcur[i] = v0 * cos(thcur[i]);
				vycur[i] = v0 * sin(thcur[i]);
				xcur0 = xcur[i] + dt * vxcur[i];
				xcur[i] = getPBCval(xcur0, Lx);
				ycur0 = ycur[i] + dt * vycur[i];
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
		free(thcur);

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
	fpj1 = fopen("Cll_nisc.dat", "w");
	FILE* fpj2;
	fpj2 = fopen("Clt_r_nisc.dat", "w");
	FILE* fpj3;
	fpj3 = fopen("Clt_i_nisc.dat", "w");
	FILE* fpj4;
	fpj4 = fopen("Ctt_nisc.dat", "w");

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