#include <fftw3.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define dim		1
#define TPB		1024  // TPB = number of threads per block

void real2complex(fftw_complex *c, double *a, int n);
void complex2real_scaled(double *a, fftw_complex *c, double scale, int n);
void solve_poisson(fftw_complex *c, double *K, int n);
void gaussian(double *bin, const double *R, const int n);
void importData(double *R, double *data, const char *file, const int n);
void mirror(double *dataM, const double *data, const int n, const int reverse);
void exportData(const char *file, const double *R, const double *data, const int n);
void fixBC(double *data, const double *R, const double delta, const int n);	// Boundary condition
void getRho2(double *result, const double *data, const double delta, const int n);
void getError(const double *rho, const double *rho2, const int n);


int main(){
	///////////////////////////// INITIZALIZATION ////////////////////////////
	printf("Phase 1: Set Up The Environment for Testing\n");
	int N = 400;
	double range, deltaR, deltaK; 	
	double PI = atan(1) * 4;
	
	char *starFile = (char *)"Star_WENO_Density_1.dat";
	char *phiFile = (char *)"phi_data.dat";
	char *rhoFile = (char *)"rho_data.dat";
	char *rho2File = (char *)"rho2_data.dat";
	
	double *RData = (double *)malloc(sizeof(double) * N);
	double *R = (double *)malloc(sizeof(double) * 2 * N);
	double *K = (double *)malloc(sizeof(double) * 2 * N);
	double *rhoData = (double *)malloc(sizeof(double) * N);
	double *rho = (double *)malloc(sizeof(double) * 2 * N);
	double *rho2 = (double *)malloc(sizeof(double) * 2 * N);
	double *phi = (double *)malloc(sizeof(double) * 2 * N);
	fftw_complex *rho_complex = (fftw_complex *)malloc(sizeof(fftw_complex) * 2 * N);
	
	importData(RData, rhoData, starFile, N);
	mirror(R, RData, N, -1);
	mirror(rho, rhoData, N, 1);
	range = RData[N-1] - RData[0];
	deltaR = range / (N - 1);
	
	int m = 0;
	/*range = 100;
	deltaR = range / N;
	for(int i = -N; i < N; i++){
		R[m] = i * deltaR;
		m += 1;
	}
	m = 0;
	gaussian(rho, R, 2 * N);*/
	
	printf("Range: %.3lf\n", range);
	printf("Delta: %.3lf\n", deltaR);
	deltaK = 1.0 / (2 * range);
	for(int i = 0; i < N; i++){
		if(m < N){
			K[m] = i * deltaK;
			K[m+N] = (double)(i - N) * deltaK;
		}
		m += 1;
	}
	for (int i = 0; i < 2 * N; i++){
        phi[i] = 0.0;
		rho2[i] = 0.0;
	}
	
	printf("Phase 1 ended\n\n");
	//////////////////////////////////////////////////////////////////////////
	
	printf("Phase 2: Evaluation\n");

	printf("Start to solve the Poisson equation...");
	fflush(stdout);
	clock_t startTime2 = clock();

	const double PI2 = 4 * PI * PI;
	double scale = 1.0 / (2 * N * PI2);
	real2complex(rho_complex, rho, 2 * N);
	fftw_plan planF = fftw_plan_dft_1d(2 * N, rho_complex, rho_complex, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(planF);
	solve_poisson(rho_complex, K, 2 * N);
	fftw_plan planB = fftw_plan_dft_1d(2 * N, rho_complex, rho_complex, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(planB);
	complex2real_scaled(phi, rho_complex, scale, 2 * N);

	printf("done!\n");
	fflush(stdout);

	fixBC(phi, R, deltaR, 2 * N);
	clock_t endTime2 = clock();
	
	printf("Phase 2 ended\n");

	double totalTime2 = (double)(endTime2 - startTime2) / CLOCKS_PER_SEC;
	
	printf("Time spent on calculation: %.3lf sec\n", totalTime2);
	
	printf("Phase 3: Data Exportation and Error Evaluation\n");
	
	getRho2(rho2, phi, deltaR, 2 * N);
	getError(rho, rho2, 2 * N);
	exportData(phiFile, R, phi, 2 * N);
	exportData(rhoFile, R, rho, 2 * N);
	exportData(rho2File, R, rho2, 2 * N);
	printf("Finish!\n");
	printf("Phase 3 ended\n\n");

    // Destroy plan and clean up memory on device
	free(K);
	free(RData);
	free(R);
	free(rhoData);
	free(rho);
	free(rho2);
	free(rho_complex);
	free(phi);
	// fftw_destory_plan(planF);
	// fftw_destory_plan(planB);
	
	return 0;
	
}

void real2complex(fftw_complex *c, double *a, int n){

	for(int i=0; i<n; i++){
		c[i][0] = a[i];
		c[i][1] = 0.0;
	}
	
}

void complex2real_scaled(double *a, fftw_complex *c, double scale, int n){

	for(int i=0; i<n; i++){
			a[i] = scale * c[i][0];
	}
	
}

void solve_poisson(fftw_complex *c, double *K, int n){

	double scale;
	
	for(int i=0; i<n; i++){
		if(i == 0){
			scale = 0.0;
		}else{
			scale = -1.0 / (K[i] * K[i]);
		}

		c[i][0] *= scale;
		c[i][1] *= scale;
	}
	
}

void gaussian(double *bin, const double *R, const int n){

	const double PI = 4 * atan(1);
	double r;
	double scale;		// Normalization factor
	
	// Generate required function
	printf("Generating density distribution...");
	fflush(stdout);
	
	scale = 10.0 / sqrt(2 * PI);
	for(int i = 0; i < n; i++){
		bin[i] = scale * exp(-(R[i] * R[i]) / 2); 
	}
	
	printf("done!\n");
	fflush(stdout);

}

void importData(double *R, double *data, const char *file, const int n){

	FILE *dataFile = fopen(file, "r");
	long double time, R_tmp, data_tmp = 0.0;
	
	printf("Importing data from \"%s\"\n", file);
	fflush(stdout);
	
	if(dataFile != NULL){
		fscanf(dataFile, " \"Time =    %Lf", &time);
		printf("Time = %lf\n", time);
		for(int i = 0; i < n; i++){
			fscanf(dataFile, "%Le %Le", &R_tmp, &data_tmp);
			R[i] = (double) R_tmp;
			data[i] = (double) data_tmp;
		}
		fclose(dataFile);
	}else{
		printf("File not found!\n");
	}

}

void mirror(double *dataM, const double *data, const int n, const int reverse){

	for(int i = 0; i < n; i++){
		dataM[i] = reverse * data[n-1-i];
		dataM[i+n] = data[i];
	}

}

void exportData(const char *file, const double *R, const double *data, const int n){
	
	FILE *dataFile = fopen(file, "w");
	
	printf("Exporting data to \"%s\"...", file);
	fflush(stdout);
	
	if(dataFile != NULL){
		for(int i = 0; i < n; i++){ 
			fprintf(dataFile, "%le\t%.18le\n", R[i], data[i]);
		}
		printf("done!\n");
		printf("All data have been stored in \"%s\".\n", file);
		fflush(stdout);
		fclose(dataFile);
	}else{
		printf("File not found!");
	}
	
}

void fixBC(double *data, const double *R, const double delta, const int n){

	double a, b;			// Solution of laplace equation: ar + b
	
	printf("Handling boundary condition...");
	fflush(stdout);
	
	a = (double)(data[2] - data[0]) / (delta * 2);
	b = data[1] - a * abs(R[1]);
	
	for(int i=0; i<n; i++){
		data[i] -= a * R[i] + b;
	}
	
	printf("done!\n");
	fflush(stdout);

}

void getRho2(double *result, const double *data, const double delta, const int n){

	const double iDelta2 = 1.0 / (delta * delta);
	const double scale = iDelta2;
	// Fix boundary to be zero
	for(int i = 0; i < n; i++){
		result[0] = 0.0;
		result[n-1] = 0.0;
	}
	// Finite Difference
	for(int i = 1; i < n - 1; i++){
		result[i] = scale * (data[i-1] + data[i+1] - 2 * data[i]);
	}
	double c = result[1];
	for(int i = 1; i < n - 1; i++){
		result[i] -= c;
	}

}

void getError(const double *rho, const double *rho2, const int n){

	double error = 0.0;
	double totalError = 0.0;
	double averageError = 0.0;
	double maxError = 0.0;
	
	int count = 0;
	for(int i = 0; i < n; i++){
		if (abs(rho2[i])>0 || abs(rho[i])>0){
			error = (double) abs(rho2[i] - rho[i]);
			totalError += error; 
			if(error > maxError){
				maxError = error;
			}
			count += 1;
		}
	}
	printf("Max error: %le\n", maxError);
	if(count > 0){
		averageError = (double) totalError / count;
		printf("Average error = %le\n", averageError);
	}else{
		printf("No error!\n");
	}

}