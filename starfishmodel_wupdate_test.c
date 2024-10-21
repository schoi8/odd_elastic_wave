#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_blas.h>

/*
	Starfish embryo model test for the effect of updating self-spinning frequency w. 
	Pre-set 6 neighbors for faster simulation, as neighbors do not change. Noise is in self-circling.
*/

#define numx 20 // number of columns. has to be even for pbc.
#define numy 20 // number of rows. has to be even for pbc.
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
gsl_vector* getVofRowSum(gsl_matrix* mat);
gsl_matrix* getInverse(gsl_matrix* mat);

int main() {
	clock_t tStart = clock();
	srand((unsigned)time(NULL));

	FILE* fp_xarr = fopen("x_arr.dat", "w");
	FILE* fp_yarr = fopen("y_arr.dat", "w");
	FILE* fp_vxarr = fopen("vx_arr.dat", "w");
	FILE* fp_vyarr = fopen("vy_arr.dat", "w");
	FILE* fp_tharr = fopen("th_arr.dat", "w");
	FILE* fp_warr = fopen("w_arr.dat", "w");
	fclose(fp_xarr);
	fclose(fp_yarr);
	fclose(fp_vxarr);
	fclose(fp_vyarr);
	fclose(fp_tharr);
	fclose(fp_warr);
	FILE* fp_xcur = fopen("xcur.txt", "w");
	FILE* fp_ycur = fopen("ycur.txt", "w");
	FILE* fp_vxcur = fopen("vxcur.txt", "w");
	FILE* fp_vycur = fopen("vycur.txt", "w");
	FILE* fp_thcur = fopen("thcur.txt", "w");
	FILE* fp_wcur = fopen("wcur.txt", "w");
	fclose(fp_xcur);
	fclose(fp_ycur);
	fclose(fp_vxcur);
	fclose(fp_vycur);
	fclose(fp_thcur);
	fclose(fp_wcur);

	double R = 0.5; // embryo radius. rescaled to 2R = 1.
	double spacing = 1.2; // spacing scale.
	double r0 = 2 * R * spacing; // perfect crystal lattice spacing
	double Lx = numx * r0; // x length of the system box.
	double Ly = 0.5 * sqrt(3) * numy * r0; // y length of the system box.
	double rm = 3.8 * R; // neighbor detecting radius 

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

	// current rotational frequency of each particle
	double w0 = 1.0;
	gsl_vector* wcur = gsl_vector_alloc(N);
	gsl_vector_set_all(wcur, w0); // intially no variation in w

	gsl_vector* w0vec = gsl_vector_alloc(N); // a vector of length N whose element is w0. Used later for matrix calculation for w update.
	gsl_vector_set_all(w0vec, w0);

	double pi = acos(-1); // define pi
	double u0 = 0.05; // initial displacement amplitude

	double Fst0 = 53.7;
	double frep0 = 785.1;
	double tau0 = 0.12;

	// rondom variable v0
	double vi; // "v0 value" of each particle i
	double v0 = 0.01; // average amplitude of self-propelling velocity
	//double v0 = 0.0;
	double ve = 0.1; // noise amplitude of v0.
	//double ve = 0.0;
	double* vxnoise;
	vxnoise = (double*)malloc(N * sizeof(double));
	double* vynoise;
	vynoise = (double*)malloc(N * sizeof(double));

	double thcur0;

	// preparing to generate normal random variables using gsl
	const gsl_rng_type* T_gauss;
	gsl_rng* gr;
	gsl_rng_env_setup();
	T_gauss = gsl_rng_default;
	gr = gsl_rng_alloc(T_gauss);
	gsl_rng_set(gr, time(NULL)); // seed based on the current time

	double sig = 1.0;

	// initial condition
	for (int i = 0; i < N; i++) {
		xcur[i] = xlatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
		ycur[i] = ylatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
		thcur0 = 2 * pi * (double)rand() / (double)RAND_MAX;
		thcur[i] = thcur0;
	}

	// simulation
	double tcur = 0.0;
	double dt = 0.001;
	int n_tot = 100000; // total number of time steps
	int n_interval = 100; // time step interval for data recording
	int n_rec = (int)(n_tot / n_interval);

	FILE* fp_para = fopen("simulation_para.txt", "w");
	fprintf(fp_para, "numx = %i,  numy = %i, N = %i \n", numx, numy, N);
	fprintf(fp_para, "n_tot = %i, n_interval = %i, dt = %f \n", n_tot, n_interval, dt);
	fprintf(fp_para, "tmax = n_tot*dt = %f, t_interval = n_interval*dt = %f \n", n_tot * dt, n_interval * dt);
	fprintf(fp_para, "R = %f, spacing = %f, r0 = %f, rm = %f \n", R, spacing, r0, rm);
	fprintf(fp_para, "Fst0 = %f, frep0 = %f, w0 = %f, f0 = %f, u0 = %f, v0 = %f \n", Fst0, frep0, w0, f0, u0, v0);
	fprintf(fp_para, "noisy v0. mean = %f, noise amplitude = %f \n", v0, ve);
	fclose(fp_para);

	// current matrix
	double dx = 0.5 * spacing;
	double dy = 0.5 * spacing;
	const int xnum = (int)round(Lx / dx);
	const int ynum = (int)round(Ly / dy);

	const int n_j = (int)((n_tot / n_interval) * 0.8);
	
	double* jxmat;
	jxmat = (double*)calloc(xnum * ynum * n_j, sizeof(double));
	double* jymat;
	jymat = (double*)calloc(xnum * ynum * n_j, sizeof(double));

	//double xij, yij, rij, xcur0, ycur0, wi, wj, phi;
	double xij, yij, rij, xcur0, ycur0;
	double wi, wj;
	int n_idx, ix, iy;
	int idx_neigh;

	int endcode = 0;

	for (int n = 0; n < n_tot; n++) {
		gsl_matrix* wadj_mat = gsl_matrix_alloc(N, N); // adjacency matrix to use later to update w
		gsl_matrix_set_zero(wadj_mat); // initialize elements to zero

		for (int i = 0; i < N; i++) {
			vxcur[i] = 0.0;
			vycur[i] = 0.0;

			int* neighvec = getNeighIdx(i); // array of neighbor indices

			for (int j = 0; j < 6; j++) {
				idx_neigh = neighvec[j];
				xij = getDist(xcur[i], xcur[idx_neigh], Lx);
				yij = getDist(ycur[i], ycur[idx_neigh], Ly);
				rij = sqrt(xij * xij + yij * yij);

				if (rij < 2 * R) {
					printf("overlap happened at time %f which is time step %i \n", tcur, n);
					printf("overlap between %d and %d \n", i, j);
					endcode = 1;
					printf("x values are %f and %f \n", xcur[i], xcur[idx_neigh]);
					printf("y values are %f and %f \n", ycur[i], ycur[idx_neigh]);
					printf("calculated xdis %f \n", xij);
					printf("calculated ydis %f \n", yij);
					printf("Time taken: %.7fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
					break;
				}
				else if (rij < rm) {
					gsl_matrix_set(wadj_mat, i, idx_neigh, tau0 * nearfield(xij, yij));
					wi = gsl_vector_get(wcur, i);
					wj = gsl_vector_get(wcur, idx_neigh);

					vxcur[i] += drdt_xl(xij, yij, Fst0, frep0, Fst0, frep0) + drdt_xt(xij, yij, wi, wj);
					vycur[i] += drdt_yl(xij, yij, Fst0, frep0, Fst0, frep0) + drdt_yt(xij, yij, wi, wj);
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

		// complete the w adj matrix
		gsl_vector* diagEle = getVofRowSum(wadj_mat);
		for (int i = 0; i < N; i++) {
			gsl_matrix_set(wadj_mat, i, i, gsl_vector_get(diagEle, i) + 1.0);
		}
		gsl_vector_free(diagEle);

		// get inverse of the w adj matrix
		gsl_matrix* wadj_inv = getInverse(wadj_mat);
		gsl_matrix_free(wadj_mat);

		// update w
		gsl_blas_dgemv(CblasNoTrans, 1.0, wadj_inv, w0vec, 0.0, wcur);
		gsl_matrix_free(wadj_inv);

		// update variables
		for (int i = 0; i < N; i++) {
			vi = ve * gsl_ran_gaussian(gr, sig) + v0;

			vxnoise[i] = (vi - v0) * cos(thcur[i]);
			vynoise[i] = (vi - v0) * sin(thcur[i]);
			
			vxcur[i] += vi * cos(thcur[i]);
			vycur[i] += vi * sin(thcur[i]);

			xcur0 = xcur[i] + dt * vxcur[i];
			xcur[i] = getPBCval(xcur0, Lx);
			
			ycur0 = ycur[i] + dt * vycur[i];
			ycur[i] = getPBCval(ycur0, Ly);

			thcur0 = thcur[i] + dt * gsl_vector_get(wcur, i);
			thcur[i] = getPBCval(thcur0, 2 * pi);
		}
		
		// save the variables
		if (n % n_interval == n_interval - 1) {
			fp_xarr = fopen("x_arr.dat", "a");
			fp_yarr = fopen("y_arr.dat", "a");
			fp_vxarr = fopen("vx_arr.dat", "a");
			fp_vyarr = fopen("vy_arr.dat", "a");
			fp_tharr = fopen("th_arr.dat", "a");
			fp_warr = fopen("w_arr.dat", "a");

			for (int i = 0; i < N; i++) {
				if (fp_xarr == NULL) {
					perror("Error in x_arr\n");
					return 1;
				}
				fprintf(fp_xarr, "%f\n", xcur[i]);

				if (fp_yarr == NULL) {
					perror("Error in y_arr\n");
					return 1;
				}
				fprintf(fp_yarr, "%f\n", ycur[i]);

				if (fp_vxarr == NULL) {
					perror("Error in vx_arr\n");
					return 1;
				}
				fprintf(fp_vxarr, "%f\n", vxcur[i]);

				if (fp_vyarr == NULL) {
					perror("Error in vy_arr\n");
					return 1;
				}
				fprintf(fp_vyarr, "%f\n", vycur[i]);

				if (fp_tharr == NULL) {
					perror("Error in th_arr\n");
					return 1;
				}
				fprintf(fp_tharr, "%f\n", thcur[i]);

				if (fp_warr == NULL) {
					perror("Error in w_arr\n");
					return 1;
				}
				fprintf(fp_warr, "%f\n", gsl_vector_get(wcur, i));
			}
			fclose(fp_xarr);
			fclose(fp_yarr);
			fclose(fp_vxarr);
			fclose(fp_vyarr);
			fclose(fp_tharr);
			fclose(fp_warr);

			n_idx = (int)floor(n / n_interval);

			if (n_idx >= n_rec - n_j) {
				for (int i = 0; i < N; i++) {
					ix = (int)floor(xcur[i] / dx);
					iy = (int)floor(ycur[i] / dy);

					//jxmat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vxcur[i];
					//jymat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vycur[i];
					jxmat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vxcur[i] - vxnoise[i];
					jymat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vycur[i] - vynoise[i];
				}
			}
		}
		else if (n % n_interval == 0) {
			fp_xcur = fopen("xcur.txt", "w");
			fp_ycur = fopen("ycur.txt", "w");
			fp_vxcur = fopen("vxcur.txt", "w");
			fp_vycur = fopen("vycur.txt", "w");
			fp_thcur = fopen("thcur.txt", "w");
			fp_wcur = fopen("wcur.txt", "w");

			for (int i = 0; i < N; i++) {
				if (fp_xcur == NULL) {
					perror("Error in xcur\n");
					return 1;
				}
				fprintf(fp_xcur, "%f\n", xcur[i]);

				if (fp_ycur == NULL) {
					perror("Error in ycur\n");
					return 1;
				}
				fprintf(fp_ycur, "%f\n", ycur[i]);

				if (fp_vxcur == NULL) {
					perror("Error in vxcur\n");
					return 1;
				}
				fprintf(fp_vxcur, "%f\n", vxcur[i]);

				if (fp_vycur == NULL) {
					perror("Error in vycur\n");
					return 1;
				}
				fprintf(fp_vycur, "%f\n", vycur[i]);

				if (fp_thcur == NULL) {
					perror("Error in thcur\n");
					return 1;
				}
				fprintf(fp_thcur, "%f\n", thcur[i]);

				if (fp_wcur == NULL) {
					perror("Error in wcur\n");
					return 1;
				}
				fprintf(fp_wcur, "%f\n", gsl_vector_get(wcur, i));
			}
			fclose(fp_xcur);
			fclose(fp_ycur);
			fclose(fp_vxcur);
			fclose(fp_vycur);
			fclose(fp_thcur);
			fclose(fp_wcur);
		}
		else { // save the variables during the time interval that is not recorded, in case of possible crash
			fp_xcur = fopen("xcur.txt", "a");
			fp_ycur = fopen("ycur.txt", "a");
			fp_vxcur = fopen("vxcur.txt", "a");
			fp_vycur = fopen("vycur.txt", "a");
			fp_thcur = fopen("thcur.txt", "a");
			fp_wcur = fopen("wcur.txt", "a");
			
			for (int i = 0; i < N; i++) {
				if (fp_xcur == NULL) {
					perror("Error in xcur\n");
					return 1;
				}
				fprintf(fp_xcur, "%f\n", xcur[i]);

				if (fp_ycur == NULL) {
					perror("Error in ycur\n");
					return 1;
				}
				fprintf(fp_ycur, "%f\n", ycur[i]);

				if (fp_vxcur == NULL) {
					perror("Error in vxcur\n");
					return 1;
				}
				fprintf(fp_vxcur, "%f\n", vxcur[i]);

				if (fp_vycur == NULL) {
					perror("Error in vycur\n");
					return 1;
				}
				fprintf(fp_vycur, "%f\n", vycur[i]);

				if (fp_thcur == NULL) {
					perror("Error in thcur\n");
					return 1;
				}
				fprintf(fp_thcur, "%f\n", thcur[i]);

				if (fp_wcur == NULL) {
					perror("Error in wcur\n");
					return 1;
				}
				fprintf(fp_wcur, "%f\n", gsl_vector_get(wcur, i));
			}
			fclose(fp_xcur);
			fclose(fp_ycur);
			fclose(fp_vxcur);
			fclose(fp_vycur);
			fclose(fp_thcur);
			fclose(fp_wcur);
		}
		tcur += dt;
	}
	free(xlatt);
	free(ylatt);
	free(xcur);
	free(ycur);
	free(vxcur);
	free(vycur);
	free(thcur);
	gsl_vector_free(wcur);
	gsl_vector_free(w0vec);
	gsl_rng_free(gr); // free the random number generator
	free(vxnoise);
	free(vynoise);

	if (endcode == 1) {
		free(jxmat);
		free(jymat);
		exit(0);
	}

	int wnum = (int)floor(n_j / 2) + 1;

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

	double dkx = 2 * pi / Lx; // resolution in kx
	double dky = 2 * pi / Ly; // resolution in ky

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

	double* jxlre; // longitudinal Jx real part
	jxlre = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* jxlim; // longitudinal Jx imaginary part
	jxlim = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* jxtre; // transverse Jx real part
	jxtre = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* jxtim; // transverse Jx imaginary part
	jxtim = (double*)calloc(xnum * ynum * wnum, sizeof(double));

	double* jylre; // longitudinal Jy real part
	jylre = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* jylim; // longitudinal Jy imaginary part
	jylim = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* jytre; // transverse Jy real part
	jytre = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* jytim; // transverse Jy imaginary part
	jytim = (double*)calloc(xnum * ynum * wnum, sizeof(double));

	int ki_shift, kj_shift;
	double qxhat, qyhat, qx, qy, jxre, jxim, jyre, jyim;

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

				jxlre[ki * ynum * wnum + kj * wnum + ni] = (jxre * qxhat + jyre * qyhat) * qxhat; // longitudinal Jx real part
				jxlim[ki * ynum * wnum + kj * wnum + ni] = (jxim * qxhat + jyim * qyhat) * qxhat; // longitudinal Jx imaginary part
				jxtre[ki * ynum * wnum + kj * wnum + ni] = jxre - (jxre * qxhat + jyre * qyhat) * qxhat; // transverse Jx real part
				jxtim[ki * ynum * wnum + kj * wnum + ni] = jxim - (jxim * qxhat + jyim * qyhat) * qxhat; // transverse Jx imaginary part

				jylre[ki * ynum * wnum + kj * wnum + ni] = (jxre * qxhat + jyre * qyhat) * qyhat; // longitudinal Jy real part
				jylim[ki * ynum * wnum + kj * wnum + ni] = (jxim * qxhat + jyim * qyhat) * qyhat; // longitudinal Jy imaginary part
				jytre[ki * ynum * wnum + kj * wnum + ni] = jyre - (jxre * qxhat + jyre * qyhat) * qyhat; // transverse Jy real part
				jytim[ki * ynum * wnum + kj * wnum + ni] = jyim - (jxim * qxhat + jyim * qyhat) * qyhat; // transverse Jy imaginary part
			}
		}
	}
	fftw_free(outjx);
	fftw_free(outjy);
	free(kxrange);
	free(kyrange);

	FILE* fpj1;
	fpj1 = fopen("jxlre.dat", "w");
	FILE* fpj2;
	fpj2 = fopen("jxlim.dat", "w");
	FILE* fpj3;
	fpj3 = fopen("jxtre.dat", "w");
	FILE* fpj4;
	fpj4 = fopen("jxtim.dat", "w");
	FILE* fpj5;
	fpj5 = fopen("jylre.dat", "w");
	FILE* fpj6;
	fpj6 = fopen("jylim.dat", "w");
	FILE* fpj7;
	fpj7 = fopen("jytre.dat", "w");
	FILE* fpj8;
	fpj8 = fopen("jytim.dat", "w");

	for (int i = 0; i < xnum * ynum * wnum; i++) {
		fwrite(&jxlre[i], sizeof(double), 1, fpj1);
		fwrite(&jxlim[i], sizeof(double), 1, fpj2);
		fwrite(&jxtre[i], sizeof(double), 1, fpj3);
		fwrite(&jxtim[i], sizeof(double), 1, fpj4);
		fwrite(&jylre[i], sizeof(double), 1, fpj5);
		fwrite(&jylim[i], sizeof(double), 1, fpj6);
		fwrite(&jytre[i], sizeof(double), 1, fpj7);
		fwrite(&jytim[i], sizeof(double), 1, fpj8);
	}
	fclose(fpj1);
	fclose(fpj2);
	fclose(fpj3);
	fclose(fpj4);
	fclose(fpj5);
	fclose(fpj6);
	fclose(fpj7);
	fclose(fpj8);

	free(jxlre);
	free(jxlim);
	free(jxtre);
	free(jxtim);
	free(jylre);
	free(jylim);
	free(jytre);
	free(jytim);

	printf("Simulation done. Time taken: %.7fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
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

gsl_vector* getVofRowSum(gsl_matrix* mat) {
	gsl_vector* diagEle = gsl_vector_alloc(N);
	gsl_vector* rowVec = gsl_vector_alloc(N);

	for (int i = 0; i < N; i++) {
		double rowSum = 0.0;
		gsl_matrix_get_row(rowVec, mat, i);

		for (int j = 0; j < N; j++) {
			rowSum += gsl_vector_get(rowVec, j);
		}
		gsl_vector_set(diagEle, i, rowSum);
	}
	gsl_vector_free(rowVec);

	return diagEle;
}

gsl_matrix* getInverse(gsl_matrix* mat) {
	gsl_permutation* p = gsl_permutation_alloc(N);
	int s;
	// Compute the LU decomposition of this matrix
	gsl_linalg_LU_decomp(mat, p, &s);
	// Compute the inverse of the LU decomposition
	gsl_matrix* inv = gsl_matrix_alloc(N, N);
	gsl_linalg_LU_invert(mat, p, inv);
	gsl_permutation_free(p);
	return inv;
}