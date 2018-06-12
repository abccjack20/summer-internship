#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define TPBx	32  // TPBx * TPBy = number of threads per block
#define TPBy 	32 

__global__ void real2complex(cufftComplex *c, float *a, int n);
__global__ void complex2real_scaled(float *a, cufftComplex *c, float scale, int n);
__global__ void solve_poisson(cufftComplex *c, float *kx, float *ky, int n);
void exportData(const char *file, const float *X, const float *Y, const float *Z, const int n);
void gaussian(float *bin, const float *X, const float *Y, const int n);


int main(){
	///////////////////////////// INITIZALIZATION ////////////////////////////
	int N, R; 	
	printf("Phase 1: Set Up The Environment for Testing\n");
	printf("Input the range of x and y: ");		// the range of x and y will be from -R to R
	scanf("%d", &R);
	printf("Input the number of samples: ");	// the number of samples will be N * N
	scanf("%d", &N); 
	printf("Allocating memory...\n");
	clock_t startTime11 = clock();
	
    char *uFile = (char *)"u_data.dat";
	char *rFile = (char *)"r_data.dat";
	
	float *X = (float *)malloc(sizeof(float) * N);
    float *Y = (float *)malloc(sizeof(float) * N);
	float *kx = (float *)malloc(sizeof(float) * N);
    float *ky = (float *)malloc(sizeof(float) * N);
    float *r = (float *)malloc(sizeof(float) * N * N);
	float *u = (float *)malloc(sizeof(float) * N * N);
	
	const float EPSILON = 8.85418782 * pow(10, -12); // Permitivity of free space
	const float PI = 4 * atan(1);
	
	float *kx_d, *ky_d, *r_d;
    cufftComplex *r_complex_d;
    cudaMalloc((void **)&kx_d, sizeof(float) * N);
    cudaMalloc((void **)&ky_d, sizeof(float) * N);
    cudaMalloc((void **)&r_d, sizeof(float) * N * N);
    cudaMalloc((void **)&r_complex_d, sizeof(cufftComplex) * N * N);
	
    int m = 0;
	for(int i = N/-2; i < N/2; i++){
		if(m < N){
			X[m] = i * (float)R / (N / 2); 
			Y[m] = -1 * i * (float)R / (N / 2); 
			kx[m] = i * PI * N / (float)R;		// Centers kx values to be at the origin
			ky[m] = - i * PI * N / (float)R;	// Centers ky values to be at the origin
		}
		m += 1;
	}
	clock_t endTime11 = clock();
	
	clock_t startTime12 = clock();
	gaussian(r, X, Y, N);	// Generate a Gaussian Distribution for r
	clock_t endTime12 = clock();
	
    for (int i = 0; i < N * N; i++){
        u[i] = 0.f;
	}

	double totalTime11 = (double)(endTime11 - startTime11) / CLOCKS_PER_SEC;
	double totalTime12 = (double)(endTime12 - startTime12) / CLOCKS_PER_SEC;
	
	printf("Phase 1 ended\n");
	printf("Time spent on allocating memory: %f sec\n", totalTime11);
	printf("Time spent on generating function: %f sec\n\n", totalTime12);
	//////////////////////////////////////////////////////////////////////////
	
	printf("Phase 2: Evaluation\n");
	printf("Copying data from the host to the device...\n");
	clock_t startTime21 = clock();
	
    cudaMemcpy(kx_d, kx, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(ky_d, ky, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(r_d, r, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_C2C);

    // Compute the execution configuration
    dim3 dimBlock(TPBx, TPBy);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
	
    // Handle N not multiple of TPBx or TPBy
    if(N % TPBx != 0){
		dimGrid.x += 1;
	}
    if(N % TPBy != 0){
		dimGrid.y += 1;
	}
	
	clock_t endTime21 = clock();
	printf("Start to solve the Poisson equation...\n");
	clock_t startTime22 = clock();

    real2complex<<<dimGrid, dimBlock>>>(r_complex_d, r_d, N);
    cufftExecC2C(plan, r_complex_d, r_complex_d, CUFFT_FORWARD);
    solve_poisson<<<dimGrid, dimBlock>>>(r_complex_d, kx_d, ky_d, N);
    cufftExecC2C(plan, r_complex_d, r_complex_d, CUFFT_INVERSE);
	float scale = 1.f / (EPSILON * N * N);
	complex2real_scaled<<<dimGrid, dimBlock>>>(r_d, r_complex_d, scale, N);
	
	clock_t endTime22 = clock();
	clock_t startTime23 = clock();

    cudaMemcpy(u, r_d, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
	
	clock_t endTime23 = clock();
	printf("Phase 2 ended\n");

	double totalTime21 = (double)(endTime22 - startTime22) / CLOCKS_PER_SEC;
	double totalTime22 = (double)(endTime21 + endTime23 - startTime21 - endTime23) / CLOCKS_PER_SEC;
	
	printf("Time spent on calculation: %f sec\n", totalTime21);
	printf("Data spent on data transfer: %f sec\n\n", totalTime22);
	
	printf("Phase 3: Data Exportation");
	printf("Exporting data...\n");
	
	clock_t startTime31 = clock();
	exportData(rFile, X, Y, r, N);
	exportData(uFile, X, Y, u, N);
	clock_t endTime31 = clock();
	printf("Finish!\n");
	printf("Phase 3 ended\n");
	
	double totalTime31 = (double)(endTime31 - startTime31) / CLOCKS_PER_SEC;
	printf("Time spent on exporting files: %f sec\n", totalTime31);

    // Destroy plan and clean up memory on device
    free(kx);
    free(ky);
	free(X);
	free(Y);
    free(r);
    free(u);
    cufftDestroy(plan);
    cudaFree(r_complex_d);
    cudaFree(kx_d);
	cudaFree(ky_d);
	
	return 0;
	
}

__global__ void real2complex(cufftComplex *c, float *a, int n){
    /* compute idx and idy, the location of the element in the original NxN array */
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idxX < n && idxY < n){
		int idx = idxX + idxY * n;
		c[idx].x = a[idx];
		c[idx].y = 0.0f;
	}
	
}

__global__ void complex2real_scaled(float *a, cufftComplex *c, float scale, int n){
    /* Compute index X and index Y, the location of the element in the original NxN array */
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idxX < n && idxY < n){
		int idx = idxX + idxY * n;
		a[idx] = scale * c[idx].x;
	}
	
}

__global__ void solve_poisson(cufftComplex *c, float *kx, float *ky, int n){
    /* compute idx and idy, the location of the element in the original NxN array */
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	
    if (idxX < n && idxY < n){
        int idx = idxX + idxY * n;
        float scale = -(kx[idxX] * kx[idxX] + ky[idxY] * ky[idxY]);
		
        if(idxX == n/2 && idxY == n/2){
			scale = 1.0f;
		}
		
        scale = 1.0f / scale;
        c[idx].x *= scale;
        c[idx].y *= scale;
    }
	
}

void exportData(const char *file, const float *X, const float *Y, const float *Z, const int n){
	
	FILE *dataFile = fopen(file, "w");
	
	if(dataFile != NULL){
		for(int j = 0; j < n ; j++){
			for(int i = 0; i < n; i++){ 
				fprintf(dataFile, "%f\t%f\t%f\n", X[i], Y[j], Z[i+j*n]);
			}
		}
		printf("All data have been stored in \"%s\".\n", file);
		fclose(dataFile);
	}else{
		printf("File not found!");
	}
	
}

void gaussian(float *bin, const float *X, const float *Y, const int n){

	int sNum;		// Number of signal
	int dim = 2;
	const float PI = 4 * atan(1);
	float x, y;
	
	// Ask for essential parameters
	printf("Number of signal: ");
	scanf("%d", &sNum);
	
	float *sPos = (float *)malloc(sizeof(float) * dim * sNum);	// Position of signal
	float *scale = (float *)malloc(sizeof(float) * sNum);		// Normalization factor
	float *var = (float *)malloc(sizeof(float) * sNum);			// Variances
	for(int s = 0; s < sNum; s++){
		printf("Position of signal %d(e.g. 1.2 -3): ", s+1);
		scanf("%f %f", &sPos[0+s*dim], &sPos[1+s*dim]);
		printf("Value of variance %d: ", s+1);
		scanf("%f", &var[s]);
	}
	
	// Generate required function
	printf("Generating density distribution...");
	for(int s = 0; s < sNum; s++){
		scale[s] = 1.0f / sqrt(2 * PI * var[s]);
	}
	for(int j = 0; j < n-1; j++){
        for(int i = 0; i < n-1; i++){
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
	
	// Clean up
	free(sPos);
	free(scale);
	free(var);

}