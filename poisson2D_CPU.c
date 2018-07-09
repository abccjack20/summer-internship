#include <fftw3.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define dim		2


void real2complex(fftw_complex *c, double *a, int n);
void complex2real_scaled(double *a, fftw_complex *c, double scale, int n);
void solve_poisson(fftw_complex *c, double *kx, double *ky, int n);
void exportData(const char *file, const double *X, const double *Y, const double *Z, const int n);
void gaussian(double *bin, const double *X, const double *Y, const double *sPos, const double *var, const int sNum, const int n);
void fixBC(double *result, const double *X, const double *Y, const double *sPos, const int sNum, const double delta, const int n);	// Boundary condition


int main(){
	///////////////////////////// INITIZALIZATION ////////////////////////////
	int N, R, sNum; 	
	printf("Phase 1: Set Up The Environment for Testing\n");
	printf("Input the range of x and y: ");		// the range of x and y will be from -R to R
	scanf("%d", &R);
	printf("Input the number of samples: ");	// the number of samples will be N * N
	scanf("%d", &N); 
	printf("Allocating memory...");
	fflush(stdout);
	clock_t startTime11 = clock();
	
	char *uFile = (char *)"u_data.dat";
	char *rFile = (char *)"r_data.dat";
	
	double *X = (double *)malloc(sizeof(double) * N);
	double *Y = (double *)malloc(sizeof(double) * N);
	double *kx = (double *)malloc(sizeof(double) * N);
	double *ky = (double *)malloc(sizeof(double) * N);
	double *r = (double *)malloc(sizeof(double) * N * N);
	double *u = (double *)malloc(sizeof(double) * N * N);
	fftw_complex *r_complex = (fftw_complex *)malloc(sizeof(fftw_complex) * N * N);
	
	const double EPSILON = 8.85418782 * pow(10, -12); // Permitivity of free space
	const double PI = 4 * atan(1);
	
	int m = 0;
	double deltaX = (double)R / (N / 2);
	double deltaK = 1.0 / (2 * R);
	for(int i = N/-2; i < N/2; i++){
		if(m < N){
			X[m] = i * deltaX; 
			Y[m] = i * deltaX; 
		}
		m += 1;
	}
	m = 0;
	for(int i = 0; i < N/2; i++){
		if(m < N/2){
			kx[m] = i * deltaK;
			kx[m+N/2] = (double)(i - N / 2) * deltaK;
			ky[m] = i * deltaK;
			ky[m+N/2] = (double)(i - N / 2) * deltaK;
		}
		m += 1;
	}
	clock_t endTime11 = clock();
	
	printf("done!\n");
	fflush(stdout);
	
	clock_t startTime12 = clock();
	printf("Number of signal: ");
	scanf("%d", &sNum);
	double *sPos = (double *)malloc(sizeof(double) * dim * sNum);	// Position of signal
	double *var = (double *)malloc(sizeof(double) * sNum);			// Variances
	for(int s = 0; s < sNum; s++){
		printf("Position of signal %d(e.g. 1.2 -3): ", s+1);
		scanf("%lf %lf", &sPos[0+s*dim], &sPos[1+s*dim]);
		printf("Value of variance %d: ", s+1);
		scanf("%lf", &var[s]);
	}
	for(int s = 0; s < sNum; s++){
		printf("Position %d = (%lf,%lf); Variance %d = %lf\n", s+1, sPos[0+s*dim],
			   sPos[1+s*dim], s+1, sNum, var[s]);
	}
	gaussian(r, X, Y, sPos, var, sNum, N);	// Generate a Gaussian Distribution for r
	gaussian(u, X, Y, sPos, var, sNum, N);
	clock_t endTime12 = clock();

	double totalTime11 = (double)(endTime11 - startTime11) / CLOCKS_PER_SEC;
	double totalTime12 = (double)(endTime12 - startTime12) / CLOCKS_PER_SEC;
	
	printf("Phase 1 ended\n");
	printf("Time spent on allocating memory: %lf sec\n", totalTime11);
	printf("Time spent on generating function: %lf sec\n\n", totalTime12);
	//////////////////////////////////////////////////////////////////////////
	
	printf("Phase 2: Evaluation\n");

	const double PI2 = 4 * PI * PI;
	double scale = 1.0 / (N * N * PI2);
	clock_t startTime21 = clock();
	real2complex(r_complex, r, N);
	fftw_plan planF = fftw_plan_dft_2d(N, N, r_complex, r_complex, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(planF);
	solve_poisson(r_complex, kx, ky, N);
	fftw_plan planB = fftw_plan_dft_2d(N, N, r_complex, r_complex, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(planB);
	complex2real_scaled(u, r_complex, scale, N);
	fixBC(u, X, Y, sPos, sNum, deltaX, N);
	clock_t endTime21 = clock();
	
	printf("Phase 2 ended\n");

	double totalTime21 = (double)(endTime21 - startTime21) / CLOCKS_PER_SEC;
	
	printf("Time spent on calculation: %lf sec\n", totalTime21);
	
	printf("Phase 3: Data Exportation\n");
	
	clock_t startTime31 = clock();
	exportData(rFile, X, Y, r, N);
	exportData(uFile, X, Y, u, N);
	clock_t endTime31 = clock();
	printf("Finish!\n");
	printf("Phase 3 ended\n");
	
	double totalTime31 = (double)(endTime31 - startTime31) / CLOCKS_PER_SEC;
	printf("Time spent on exporting files: %lf sec\n", totalTime31);

    // Destroy plan and clean up memory on device
	free(kx);
	free(ky);
	free(X);
	free(Y);
	free(r);
	free(r_complex);
	free(u);
	// fftw_destory_plan(planF);
	// fftw_destory_plan(planB);
	
	return 0;
	
}

void real2complex(fftw_complex *c, double *a, int n){
	
	for(int j=0; j<n; j++){
		for(int i=0; i<n; i++){
			c[i+j*n][0] = a[i+j*n];
			c[i+j*n][1] = 0.0;
		}
	}
	
}

void complex2real_scaled(double *a, fftw_complex *c, double scale, int n){
	
	for(int j=0; j<n; j++){
		for(int i=0; i<n; i++){
			a[i+j*n] = scale * c[i+j*n][0];
		}
	}	
	
}

void solve_poisson(fftw_complex *c, double *kx, double *ky, int n){
	
	double scale;
	for(int j=0; j<n; j++){
		for(int i=0; i<n; i++){
			if(i == 0 && j == 0){
				scale = 0.0;
			}else{
				scale = -1.0 / (kx[i] * kx[i] + ky[j] * ky[j]);
			}
			
			c[i+j*n][0] *= scale;
			c[i+j*n][1] *= scale;
		}
	}
	
}

void exportData(const char *file, const double *X, const double *Y, const double *Z, const int n){
	
	FILE *dataFile = fopen(file, "w");
	
	printf("Exporting data to \"%s\"...", file);
	fflush(stdout);
	
	if(dataFile != NULL){
		for(int j = 0; j < n ; j++){
			for(int i = 0; i < n; i++){ 
				fprintf(dataFile, "%lf\t%lf\t%lf\n", X[i], Y[j], Z[i+j*n]);
			}
		}
		printf("done!\n");
		printf("All data have been stored in \"%s\".\n", file);
		fflush(stdout);
		fclose(dataFile);
	}else{
		printf("File not found!");
	}
	
}

void gaussian(double *bin, const double *X, const double *Y, const double *sPos, const double *var, const int sNum, const int n){

	const double PI = 4 * atan(1);
	double x, y;
	
	double *scale = (double *)malloc(sizeof(double) * sNum);		// Normalization factor
	
	// Generate required function
	printf("Generating density distribution...");
	fflush(stdout);
	
	for(int s = 0; s < sNum; s++){
		scale[s] = 10.0 / sqrt(2 * PI * var[s]);
	}
	for(int j = 0; j < n; j++){
        for(int i = 0; i < n; i++){
			bin[i+j*n] = 0;
			for(int s = 0; s < sNum; s++){
				x = X[i] - sPos[0+s*dim];
				y = Y[j] - sPos[1+s*dim];
				bin[i+j*n] += scale[s] * exp(-(x * x + y * y)/(2 * var[s]));
			} 
		}
	}
	// Fix boundary
	for(int i = 0; i < n; i++){
		bin[i+(n-1)*n] = bin[i];
		bin[(n-1)+i*n] = bin[i*n];
	}
	
	printf("done!\n");
	fflush(stdout);

}

void fixBC(double *result, const double *X, const double *Y, const double *sPos, const int sNum, const double delta, const int n){

	double a, b, c;			// Solution of laplace equation: ax + by + c
	
	printf("Handling boundary condition...");
	fflush(stdout);
	
	a = (double)(result[2+1*n] - result[0+1*n]) / (delta * 2);
	b = (double)(result[1+2*n] - result[1+0*n]) / (delta * 2);
	c = result[1+1*n] - a * X[1] - b * Y[1];
	
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			result[i+j*n] -= a * X[i] + b * Y[j] + c;
		}
	}
	
	printf("done!\n");
	fflush(stdout);

}