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


int main(){
	///////////////////////////// INITIZALIZATION ////////////////////////////
	int N, R; 	
	printf("Input the range of x and y: ");		// the range of x and y will be from -R to R
	scanf("%d", &R);
	printf("Input the number of samples: ");	// the number of samples will be N * N
	scanf("%d", &N); 
	
    char *uFile = (char *)"u_data.dat";
	char *rFile = (char *)"r_data.dat";
	
	float *X = (float *)malloc(sizeof(float) * N);
    float *Y = (float *)malloc(sizeof(float) * N);
	float *kx = (float *)malloc(sizeof(float) * N);
    float *ky = (float *)malloc(sizeof(float) * N);
    float *r = (float *)malloc(sizeof(float) * N * N);
	float *u = (float *)malloc(sizeof(float) * N * N);
	
	const float EPSILON = 8.85418782 * pow(10, -12); // Permitivity of free space
	
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
			kx[m] = (float)i;	// centers kx values to be at the origin
			ky[m] = (float)i;	// centers ky values to be at the origin
		}
		m += 1;
	}
	
	for(int j = 0; j < N; j++){
        for (int i = 0; i < N; i++){
			r[i+j*N] = exp(-(X[i]*X[i] + Y[j]*Y[j]));
			// (float)(1.0 / (X[x]*X[x] + Y[y]*Y[y] + 0.1));
					// exp(-(X[x]*X[x] + Y[y]*Y[y]));
		}
	}	
    for (int i = 0; i < N * N; i++){
        u[i] = 0.f;
	}
	//////////////////////////////////////////////////////////////////////////
	
	clock_t startTime = clock();
	printf("Let's start...\n");
	printf("N = %d\n", N);
	
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

    real2complex<<<dimGrid, dimBlock>>>(r_complex_d, r_d, N);
    cufftExecC2C(plan, r_complex_d, r_complex_d, CUFFT_FORWARD);
    solve_poisson<<<dimGrid, dimBlock>>>(r_complex_d, kx_d, ky_d, N);
    cufftExecC2C(plan, r_complex_d, r_complex_d, CUFFT_INVERSE);
	float scale = 1.f / (EPSILON * N * N);
	complex2real_scaled<<<dimGrid, dimBlock>>>(r_d, r_complex_d, scale, N);

    cudaMemcpy(u, r_d, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
	
	clock_t endTime = clock();
	double totalTime = ((double)endTime - (double)startTime) / ((double)CLOCKS_PER_SEC);
	
	printf("Time used: %f sec\n", totalTime);
	
	printf("Exporting data...\n");
	exportData(rFile, X, Y, r, N);
	exportData(uFile, X, Y, u, N);
	printf("Finish!\n");

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
    /* compute idx and idy, the location of the element in the original NxN array */
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