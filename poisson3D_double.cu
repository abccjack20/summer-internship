#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define dim		3
#define TPBx	16  // TPBx * TPBy = number of threads per block
#define TPBy 	8
#define TPBz	8

__global__ void real2complex(cufftDoubleComplex *c, double *a, int n);
__global__ void complex2real_scaled(double *a, cufftDoubleComplex *c, double scale, int n);
__global__ void solve_poisson(cufftDoubleComplex *c, double *kx, double *ky, double *kz, int n);
void exportData(const char *file, const double *X, const double *Y, const double *Z, const int n);
void exportData2D(const char *xfile, const char *yfile, const char *zfile, const double *X, const double *Y, const double *Z, const double *data, const int n);
void gaussian(double *bin, const double *X, const double *Y, const double *Z, const double *sPos, const double *var, const int sNum, const int n);
void fixBC(double *result, const double *X, const double *Y, const double *Z, const double delta, const int n);	// Boundary condition
void getR(double *result, const double *data, const double delta, const int n);
void getError(const double *data, const double *result, const int n);


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
	
	char *uXFName = (char *)"uX_data.dat";
	char *uYFName = (char *)"uY_data.dat";
	char *uZFName = (char *)"uZ_data.dat";
	char *rXFName = (char *)"rX_data.dat";
	char *rYFName = (char *)"rY_data.dat";
	char *rZFName = (char *)"rZ_data.dat";
	char *RXFName = (char *)"RX_data.dat";
	char *RYFName = (char *)"RY_data.dat";
	char *RZFName = (char *)"RZ_data.dat";
	
	double *X = (double *)malloc(sizeof(double) * N);
	double *Y = (double *)malloc(sizeof(double) * N);
	double *Z = (double *)malloc(sizeof(double) * N);
	double *kx = (double *)malloc(sizeof(double) * N);
	double *ky = (double *)malloc(sizeof(double) * N);
	double *kz = (double *)malloc(sizeof(double) * N);
	double *r = (double *)malloc(sizeof(double) * N * N * N);
	double *r2 = (double *)malloc(sizeof(double) * N * N * N);
	double *u = (double *)malloc(sizeof(double) * N * N * N);
	
	const double EPSILON = 8.85418782 * pow(10, -12); // Permitivity of free space
	const double PI = 4 * atan(1);
	
	double *kx_d, *ky_d, *kz_d, *r_d;
	cufftDoubleComplex *r_complex_d;
	cudaMalloc((void **)&kx_d, sizeof(double) * N);
	cudaMalloc((void **)&ky_d, sizeof(double) * N);
	cudaMalloc((void **)&kz_d, sizeof(double) * N);
	cudaMalloc((void **)&r_d, sizeof(double) * N * N * N);
	cudaMalloc((void **)&r_complex_d, sizeof(cufftDoubleComplex) * N * N * N);
	
	int m = 0;
	double deltaX = (double)R / (N / 2);
	double deltaK = 1.0 / (2 * R);
	for(int i = N/-2; i < N/2; i++){
		if(m < N){
			X[m] = i * deltaX; 
			Y[m] = i * deltaX; 
			Z[m] = i * deltaX; 
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
			kz[m] = i * deltaK;
			kz[m+N/2] = (double)(i - N / 2) * deltaK;
		}
		m += 1;
	}
	clock_t endTime11 = clock();
	printf("done!\n");
	fflush(stdout);
	
	// Ask for essential parameters for generating a charge density distribution
	printf("Number of signal: ");
	scanf("%d", &sNum);
	double *sPos = (double *)malloc(sizeof(double) * dim * sNum);	// Position of signal
	double *var = (double *)malloc(sizeof(double) * sNum);			// Variances
	for(int s = 0; s < sNum; s++){
		printf("Position of signal %d(e.g. 1.2 -3 0): ", s+1);
		scanf("%lf %lf %lf", &sPos[0+s*dim], &sPos[1+s*dim], &sPos[2+s*dim]);
		printf("Value of variance %d: ", s+1);
		scanf("%lf", &var[s]);
	}
	for(int s = 0; s < sNum; s++){
		printf("Position %d = (%lf,%lf,%lf); Variance %d = %lf\n", s+1, sPos[0+s*dim],
			   sPos[1+s*dim], sPos[2+s*dim], s+1, sNum, var[s]);
	}
	
	clock_t startTime12 = clock();
	gaussian(r, X, Y, Z, sPos, var, sNum, N);	// Generate a Gaussian Distribution for r
	clock_t endTime12 = clock();
	
	for (int i = 0; i < N * N * N; i++){
        u[i] = 0.0;
	}

	double totalTime11 = (double)(endTime11 - startTime11) / CLOCKS_PER_SEC;
	double totalTime12 = (double)(endTime12 - startTime12) / CLOCKS_PER_SEC;
	
	printf("Phase 1 ended\n");
	printf("Time spent on allocating memory: %lf sec\n", totalTime11);
	printf("Time spent on generating function: %lf sec\n\n", totalTime12);
	//////////////////////////////////////////////////////////////////////////
	
	printf("Phase 2: Evaluation\n");
	printf("Copying data from the host to the device...");
	fflush(stdout);
	clock_t startTime21 = clock();
	
	cudaMemcpy(kx_d, kx, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(ky_d, ky, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(kz_d, kz, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(r_d, r, sizeof(double) * N * N * N, cudaMemcpyHostToDevice);

	cufftHandle plan;
	if(cufftPlan3d(&plan, N, N, N, CUFFT_Z2Z) != CUFFT_SUCCESS){
		printf("\nCUFFT error: Plan creation failed!\n");
	}

    // Compute the execution configuration
	dim3 dimBlock(TPBx, TPBy, TPBz);
	dim3 dimGrid(N / dimBlock.x, N / dimBlock.y, N / dimBlock.z);
	
    // Handle N not multiple of TPBx or TPBy
	if(N % TPBx != 0){
		dimGrid.x += 1;
	}
	if(N % TPBy != 0){
		dimGrid.y += 1;
	}
	if(N % TPBz != 0){
		dimGrid.z += 1;
	}
	
	clock_t endTime21 = clock();
	printf("done!\n");
	printf("Start to solve the Poisson equation...");
	fflush(stdout);
	clock_t startTime22 = clock();

	const double PI2 = 4 * PI * PI;
	double scale = 1.0 / (N * N * N * PI2);
	real2complex<<<dimGrid, dimBlock>>>(r_complex_d, r_d, N);
	if(cufftExecZ2Z(plan, r_complex_d, r_complex_d, CUFFT_FORWARD) != CUFFT_SUCCESS){
		printf("\nCUFFT error: ExecZ2Z Forward failed!\n");
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
		printf("\nCuda error: Failed to synchronize\n");	
	}
	solve_poisson<<<dimGrid, dimBlock>>>(r_complex_d, kx_d, ky_d, kz_d, N);
	if(cufftExecZ2Z(plan, r_complex_d, r_complex_d, CUFFT_INVERSE) != CUFFT_SUCCESS){
		printf("\nCUFFT error: ExecZ2Z Backward failed!\n");
	}
	if(cudaDeviceSynchronize() != cudaSuccess){
		printf("\nCuda error: Failed to synchronize\n");	
	}
	complex2real_scaled<<<dimGrid, dimBlock>>>(r_d, r_complex_d, scale, N);
	clock_t endTime22 = clock();
	
	clock_t startTime23 = clock();
	cudaMemcpy(u, r_d, sizeof(double) * N * N * N, cudaMemcpyDeviceToHost);
	clock_t endTime23 = clock();
	printf("done!\n");
	fflush(stdout);
	
	clock_t startTime24 = clock();
	fixBC(u, X, Y, Z, deltaX, N);
	clock_t endTime24 = clock();
	
	printf("Phase 2 ended\n");

	double totalTime21 = (double)(endTime22 + endTime24 - startTime22 - startTime24) / CLOCKS_PER_SEC;
	double totalTime22 = (double)(endTime21 + endTime23 - startTime21 - endTime23) / CLOCKS_PER_SEC;
	
	printf("Time spent on calculation: %lf sec\n", totalTime21);
	printf("Time spent on data transfer: %lf sec\n\n", totalTime22);
	
	// Evaluate error
	printf("Phase 3: Error Evaluation And Data Exportation\n");
	
	clock_t startTime41 = clock();
	printf("delta = %lf\n", deltaX);
	printf("Evaluating the average error...\n");
	fflush(stdout);
	getR(r2, u, deltaX, N);
	getError(r, r2, N);
	clock_t endTime41 = clock();
	printf("done!\n");
	fflush(stdout);
	
	printf("Exporting data...\n");
	fflush(stdout);
	clock_t startTime42 = clock();
	exportData2D(uXFName, uYFName, uZFName, X, Y, Z, u, N);
	exportData2D(rXFName, rYFName, rZFName, X, Y, Z, r, N);
	exportData2D(RXFName, RYFName, RZFName, X, Y, Z, r2, N);
	clock_t endTime42 = clock();
	printf("done!\n");
	fflush(stdout);
	printf("Phase 4 ended\n");
	
	double totalTime41 = (double)(endTime41 - startTime41) / CLOCKS_PER_SEC;
	double totalTime42 = (double)(endTime42 - startTime42) / CLOCKS_PER_SEC;
	printf("Time spent on evaluating error: %lf sec\n", totalTime41);
	printf("Time spent on data exportation: %lf sec\n\n", totalTime42);

    // Destroy plan and clean up memory on device
	free(kx);
	free(ky);
	free(kz);
	free(X);
	free(Y);
	free(Z);
	free(r);
	free(u);
	free(sPos);
	free(var);
	cufftDestroy(plan);
	cudaFree(r_d);
	cudaFree(r_complex_d);
	cudaFree(kx_d);
	cudaFree(ky_d);
	cudaFree(kz_d);
	
	return 0;
	
}

__global__ void real2complex(cufftDoubleComplex *c, double *a, int n){
    /* compute idx and idy, the location of the element in the original NxN array */
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	int idxZ = blockIdx.z * blockDim.z + threadIdx.z;
	
	if(idxX < n && idxY < n && idxZ < n){
		int idx = idxX + idxY * n + idxZ * n * n;
		c[idx].x = a[idx];
		c[idx].y = 0.0;
	}
	
}

__global__ void complex2real_scaled(double *a, cufftDoubleComplex *c, double scale, int n){
    /* Compute index X and index Y, the location of the element in the original NxN array */
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	int idxZ = blockIdx.z * blockDim.z + threadIdx.z;
	
	if(idxX < n && idxY < n && idxZ < n){
		int idx = idxX + idxY * n + idxZ * n * n;
		a[idx] = scale * c[idx].x;
	}
	
}

__global__ void solve_poisson(cufftDoubleComplex *c, double *kx, double *ky, double *kz, int n){
    /* compute idxX and idxY, the location of the element in the original NxN array */
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	int idxZ = blockIdx.z * blockDim.z + threadIdx.z;
	
	if(idxX < n && idxY < n && idxZ < n){
        int idx = idxX + idxY * n + idxZ * n * n;
		double scale;
		
        if(idxX == 0 && idxY == 0 && idxZ == 0){
			scale = 0.0;
		}else{
			scale = -1.0 / (kx[idxX] * kx[idxX] + ky[idxY] * ky[idxY] + kz[idxZ] * kz[idxZ]);
		}
		
        c[idx].x *= scale;
        c[idx].y *= scale;
	}
	
}

void exportData(const char *file, const double *X, const double *Y, const double *Z, const double *data, const int n){
	
	FILE *dataFile = fopen(file, "w");
	
	printf("Exporting data to \"%s\"...", file);
	fflush(stdout);
	
	if(dataFile != NULL){
		for(int k=0; k < n; k++){
			for(int j = 0; j < n; j++){
				for(int i = 0; i < n; i++){ 
					fprintf(dataFile, "%lf\t%lf\t%lf\n", X[i], Y[j], Z[k], data[i+j*n+k*n*n]);
				}
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

void exportData2D(const char *xfile, const char *yfile, const char *zfile, const double *X, const double *Y, const double *Z, const double *data, const int n){

	FILE *xFile = fopen(xfile, "w");
	
	if(xFile != NULL){
		for(int i = 0; i < n; i++){ 
			fprintf(xFile, "%lf\t%lf\n", X[i], data[i+(n/2)*n+(n/2)*n*n]);
		}
		printf("All data have been stored in \"%s\".\n", xfile);
		fclose(xFile);
	}else{
		printf("xFile not found!");
	}
	
	FILE *yFile = fopen(yfile, "w");
	
	if(yFile != NULL){
		for(int j = 0; j < n; j++){ 
			fprintf(yFile, "%lf\t%lf\n", Y[j], data[(n/2)+j*n+(n/2)*n*n]);
		}
		printf("All data have been stored in \"%s\".\n", yfile);
		fclose(yFile);
	}else{
		printf("yFile not found!");
	}
	
	FILE *zFile = fopen(zfile, "w");
	
	if(zFile != NULL){
		for(int k = 0; k < n; k++){ 
			fprintf(zFile, "%lf\t%lf\n", Z[k], data[(n/2)+(n/2)*n+k*n*n]);
		}
		printf("All data have been stored in \"%s\".\n", zfile);
		fclose(zFile);
	}else{
		printf("zFile not found!");
	}

}

void gaussian(double *bin, const double *X, const double *Y, const double *Z, const double *sPos, const double *var, const int sNum, const int n){

	const double PI = 4 * atan(1);
	double x, y, z;
	
	double *scale = (double *)malloc(sizeof(double) * sNum);		// Normalization factor
	
	// Generate required function
	printf("Generating density distribution...");
	fflush(stdout);
	
	for(int s = 0; s < sNum; s++){
		scale[s] = 10.0 / sqrt(2 * PI * var[s]);
	}
	for(int k=0; k < n ; k++){
		for(int j = 0; j < n; j++){
			for(int i = 0; i < n; i++){
				bin[i+j*n+k*n*n] = 0;
				for(int s = 0; s < sNum; s++){
					x = X[i] - sPos[0+s*dim];
					y = Y[j] - sPos[1+s*dim];
					z = Z[k] - sPos[2+s*dim];
					bin[i+j*n+k*n*n] += scale[s] * exp(-(x * x + y * y + z * z)/(2 * var[s]));
				} 
			}
		}
	}
	
	printf("done!\n");
	fflush(stdout);

}

void fixBC(double *result, const double *X, const double *Y, const double *Z, const double delta, const int n){

	double a, b, c, d;			// Solution of laplace equation: ax + by + cz + d
	
	printf("Handling boundary condition...");
	fflush(stdout);
	
	a = (double)(result[2+1*n+1*n*n] - result[0+1*n+1*n*n]) / (delta * 2);
	b = (double)(result[1+2*n+1*n*n] - result[1+0*n+1*n*n]) / (delta * 2);
	c = (double)(result[1+1*n+2*n*n] - result[1+1*n+0*n*n]) / (delta * 2);
	d = result[1+1*n+1*n*n] - a * X[1] - b * Y[1] - c * Z[1];
	
	for(int k = 0; k < n ; k++){
		for(int j = 0; j < n; j++){
			for(int i = 0; i<n; i++){
				result[i+j*n+k*n*n] -= a * X[i] + b * Y[j] + c * Z[k] + d;
			}
		}
	}
	
	printf("done!\n");
	fflush(stdout);

}

void getR(double *result, const double *data, const double delta, const int n){

	const double iDelta2 = 1.0 / (delta * delta);
	const double scale = iDelta2;
	
	// Fix boundary to be zero
	for(int i = 0; i < n; i++){
		result[i+0*n+0*n*n] = 0.0;
		result[i+(n-1)*n+0*n*n] = 0.0;
		result[i+0*n+(n-1)*n*n] = 0.0;
		result[i+(n-1)*n+(n-1)*n*n] = 0.0;

		result[0+i*n+0*n*n] = 0.0;
		result[(n-1)+i*n+0*n*n] = 0.0;
		result[0+i*n+(n-1)*n*n] = 0.0;
		result[(n-1)+i*n+(n-1)*n*n] = 0.0;

		result[0+0*n+i*n*n] = 0.0;
		result[(n-1)+0*n+i*n*n] = 0.0;
		result[0+(n-1)*n+i*n*n] = 0.0;
		result[(n-1)+(n-1)*n+i*n*n] = 0.0;
	}
	
	// Finite Difference
	for(int k = 1; k < n - 1; k++){
		for(int j = 1; j < n - 1; j++){
			for(int i = 1; i < n - 1; i++){
				result[i+j*n+k*n*n] = scale * (data[(i-1)+j*n+k*n*n] + data[(i+1)+j*n+k*n*n]
										 + data[i+(j-1)*n+k*n*n] + data[i+(j+1)*n+k*n*n]
										 + data[i+j*n+(k-1)*n*n] + data[i+j*n+(k+1)*n*n]
										 - 6 * data[i+j*n+k*n*n]);
			}
		}
	}

}

void getError(const double *data, const double *result, const int n){

	double error = 0.0;
	double totalError = 0.0;
	double averageError = 0.0;
	double maxError = 0.0;
	
	int count = 0;
	for(int k=0; k < n ; k++){
		for(int j = 0; j < n; j++){
			for(int i = 0; i < n; i++){
				if (abs(result[i+j*n+k*n*n])>0 && abs(data[i+j*n+k*n*n])>0){
					error = (double) abs(result[i+j*n+k*n*n] - data[i+j*n+k*n*n]);
					totalError += error; 
					if(error > maxError){
						maxError = error;
					}
					count += 1;
				}
			}
		}
	}
	printf("Max error: %lf\n", maxError);
	averageError = (double) totalError / count;
	printf("Average error = %lf\n", averageError);

}