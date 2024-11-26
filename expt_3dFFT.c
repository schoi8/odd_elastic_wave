#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

/*
	3d Fourier transform of the experimental data of the trajectories of the starfish embryos.
	We use the embryo trajectory longer than 10 frames and in the later half of the measurement time.
	The trajectory data contains the detected coordinates of the embryos. 
	When the embryos are not detected, the entry is 0, which is excluded in the Fourier transform in this code.
	The pixel to millimeter ratio and the timestep for each time frame are from Tan et al. Nature 607, 287-293 (2022)
*/

int main() {
	clock_t tStart = clock();

	const int N = 1987; // number of trajectories
	const int n_tot = 400; // number of time frames 

	double* x_arr; // x coordinates of the embryo trajectories
	x_arr = (double*)malloc(N * n_tot * sizeof(double));
	double* y_arr; // y coordinates of the embryo trajectories
	y_arr = (double*)malloc(N * n_tot * sizeof(double));

	FILE* fp;
	fp = fopen("x_longtraj_laterhalf_v3.dat", "rb");
	if (fp == NULL) { // file does not exist
		printf("Error reading file for x\n");
		exit(0);
	}
	for (int i = 0; i < N * n_tot; i++) {
		fscanf(fp, "%lf", &x_arr[i]);
	}
	fclose(fp);

	fp = fopen("y_longtraj_laterhalf_v3.dat", "rb");
	if (fp == NULL) {
		printf("Error reading file for y\n");
		exit(0);
	}
	for (int i = 0; i < N * n_tot; i++) {
		fscanf(fp, "%lf", &y_arr[i]);
	}
	fclose(fp);

	// save real space velocity of each particle
	FILE* fp_vx = fopen("vx_arr.txt", "w");
	fclose(fp_vx);
	FILE* fp_vy = fopen("vy_arr.txt", "w");
	fclose(fp_vy);

	double Lx = 2000; // box size twice that of the cluster in pixels
	double Ly = 2000; // box size twice that of the cluster in pixels
	int xnum = 200;
	int ynum = 200;
	double pixtomm = 7.8 / 900; // convert from pixel to mm
	double dx = pixtomm * Lx / xnum;
	double dy = pixtomm * Ly / ynum;
	double x0 = -400.0 * pixtomm; // beginning of the box
	double y0 = -400.0 * pixtomm;

	double dt = 10.0; // timestep for each time frame

	// current matrix on the space
	double* jxmat; // current in x-direction
	jxmat = (double*)calloc(xnum * ynum * (n_tot - 1), sizeof(double)); // used calloc to initialize to 0
	double* jymat; // current in y-direction
	jymat = (double*)calloc(xnum * ynum * (n_tot - 1), sizeof(double));

	double xcur, ycur, xnext, ynext, vx, vy;
	int ix, iy;

	for (int i = 0; i < N; i++) {
		for (int it = 0; it < n_tot - 1; it++) {
			xcur = x_arr[n_tot * i + it] * pixtomm; // x value of the particle i at the current timestep
			xnext = x_arr[n_tot * i + it + 1] * pixtomm; // x value of the particle i at the next timestep
			ycur = y_arr[n_tot * i + it] * pixtomm; // y value of the particle i at the current timestep
			ynext = y_arr[n_tot * i + it + 1] * pixtomm; // y value of the particle i at the next timestep

			//if (xnext > 0) {
			if (xcur > 0 && xnext > 0) {
				vx = (xnext - xcur) / dt; // vx value of particle i
				vy = (ynext - ycur) / dt; // vy value of particle i

				ix = (int)floor((xcur - x0) / dx); // index corresponding to the x value on the spatial grid
				iy = (int)floor((ycur - y0) / dy); // index corresponding to the y value on the spatial grid

				jxmat[ix * ynum * (n_tot - 1) + iy * (n_tot - 1) + it] += vx;
				jymat[ix * ynum * (n_tot - 1) + iy * (n_tot - 1) + it] += vy;

				fp_vx = fopen("vx_arr.txt", "a");
				if (fp_vx == NULL) {
					perror("Error in vx\n");
					return 1;
				}
				fprintf(fp_vx, "%lf\n", vx);
				fclose(fp_vx);

				fp_vy = fopen("vy_arr.txt", "a");
				if (fp_vy == NULL) {
					perror("Error in vy\n");
					return 1;
				}
				fprintf(fp_vy, "%lf\n", vy);
				fclose(fp_vy);
			}
		}
	}
	free(x_arr);
	free(y_arr);

	// save jxmat and jymat
	FILE* fp_jx;
	fp_jx = fopen("jxmat.dat", "w");
	FILE* fp_jy;
	fp_jy = fopen("jymat.dat", "w");

	for (int i = 0; i < xnum * ynum * (n_tot - 1); i++) {
		fwrite(&jxmat[i], sizeof(double), 1, fp_jx);
		fwrite(&jymat[i], sizeof(double), 1, fp_jy);
	}
	fclose(fp_jx);
	fclose(fp_jy);

	int wnum = (int)floor((n_tot - 1) / 2) + 1;

	fftw_complex* outjx;
	outjx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * xnum * ynum * wnum);
	fftw_plan pjx;
	pjx = fftw_plan_dft_r2c_3d(xnum, ynum, n_tot - 1, jxmat, outjx, FFTW_ESTIMATE);
	fftw_execute(pjx);
	fftw_destroy_plan(pjx);
	free(jxmat);

	fftw_complex* outjy;
	outjy = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * xnum * ynum * wnum);
	fftw_plan pjy;
	pjy = fftw_plan_dft_r2c_3d(xnum, ynum, n_tot - 1, jymat, outjy, FFTW_ESTIMATE);
	fftw_execute(pjy);
	fftw_destroy_plan(pjy);
	free(jymat);

	double pi = acos(-1.0); // define pi
	double dkx = 2 * pi / (Lx * pixtomm); // resolution in kx
	double dky = 2 * pi / (Ly * pixtomm); // resolution in ky

	double* kxrange; // range of kx
	kxrange = (double*)malloc(xnum * sizeof(double));
	double* kyrange; // range of ky
	kyrange = (double*)malloc(ynum * sizeof(double));

	for (int i = 0; i < xnum; i++) {
		kxrange[i] = -pi / dx + dkx * i;
	}
	for (int j = 0; j < ynum; j++) {
		kyrange[j] = -pi / dy + dky * j;
	}

	int ki_shift; // index in kx dimension after shift so that 0 is at the center
	int kj_shift; // index in ky dimension after shift so that 0 is at the center
	double qxhat, qyhat, qx, qy, jxre, jxim, jyre, jyim;
	double jlre, jlim, jtre, jtim;

	double* Cll; // current correlation Jl*Jl
	Cll = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Clt_r; // current correlation Jl*Jt's real part
	Clt_r = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Clt_i; // current correlation Jl*Jt's imaginary part
	Clt_i = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Ctt; // current correlation Jt*Jt
	Ctt = (double*)calloc(xnum * ynum * wnum, sizeof(double));

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
				jyre = outjy[ki_shift * ynum * wnum + kj_shift * wnum + ni][0]; // real part of Jy
				jyim = outjy[ki_shift * ynum * wnum + kj_shift * wnum + ni][1]; // imaginary part of Jy

				jlre = jxre * qxhat + jyre * qyhat; // longitudinal Jl real part
				jlim = jxim * qxhat + jyim * qyhat; // longitudinal Jl imaginary part
				jtre = jyre * qxhat - jxre * qyhat; // transverse Jt real part
				jtim = jyim * qxhat - jxim * qyhat; // transverse Jt imaginary part

				Cll[ki * ynum * wnum + kj * wnum + ni] += jlre * jlre + jlim * jlim; // longitudinal current correalation function
				Clt_r[ki * ynum * wnum + kj * wnum + ni] += jlre * jtre + jlim * jtim; // real part of cross current correlation function
				Clt_i[ki * ynum * wnum + kj * wnum + ni] += jlre * jtim - jlim * jtre; // imaginary part of cross current correlation function
				Ctt[ki * ynum * wnum + kj * wnum + ni] += jtre * jtre + jtim * jtim; // transverse current correlation function
			}
		}
	}
	fftw_free(outjx);
	fftw_free(outjy);
	free(kxrange);
	free(kyrange);

	FILE* fpj1;
	fpj1 = fopen("Cll.dat", "w");
	FILE* fpj2;
	fpj2 = fopen("Cltr.dat", "w");
	FILE* fpj3;
	fpj3 = fopen("Clti.dat", "w");
	FILE* fpj4;
	fpj4 = fopen("Ctt.dat", "w");

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

	printf("Analysis done. Time taken: %.7fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	return 0;
}