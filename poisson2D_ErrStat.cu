#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define TPBx	32  // TPBx * TPBy = number of threads per block
#define TPBy 	32 

__global__ void real2complex(cufftDoubleComplex *c, double *a, int n);
__global__ void complex2real_scaled(double *a, cufftDoubleComplex *c, double scale, int n);
__global__ void solve_poisson(cufftDoubleComplex *c, double *kx, double *ky, int n);
void core(double *error, double *maxError, const int n, const int range);
void exportData(const char *file, const double *X, const double *Y, const double *Z, const int m, const int n);
void gaussian(double *bin, const double *X, const double *Y, const int n);
void getError(double *error, double *maxError, const double *data, const double *result, const int n);
void getR2(double *result, const double *data, const double delta, const int n);


int main(){
	///////////////////////////// INITIZALIZATION ////////////////////////////
	int minN, maxN, dN, minR, maxR, dR;
	char *errorFile = (char *)"error.dat";
	char *maxErrorFile = (char *)"maxError.dat";
	
	printf("Range of sample size: ");
	scanf("%d:%d", &minN, &maxN);
	printf("Range of sample range: ");
	scanf("%d:%d", &minR, &maxR);
	printf("Change of sample size: ");
	scanf("%d", &dN);
	printf("Change of sample range: ");
	scanf("%d", &dR);
	printf("Sample size: %d:%d; Range: %d:%d; dN: %d; dR: %d\n", minN, maxN, minR, maxR, dN, dR);
	
	int numN = (maxN - minN) / dN + 1;
	int numR = (maxR - minR) / dR + 1;
	double *error = (double *)malloc(sizeof(double));
	double *maxError = (double *)malloc(sizeof(double));
	double *errorList = (double *)malloc(sizeof(double) * numN * numR);
	double *maxErrorList = (double *)malloc(sizeof(double) * numN * numR);
	double *NList = (double *)malloc(sizeof(double) * numN);
	double *RList = (double *)malloc(sizeof(double) * numR);
	
	int currentN = minN;
	int currentR = minR;
	int r = 0;
	int n = 0;
	int p = 0;
	int pnew = 0;
	printf("Start!\n");
	printf("Progress: %d%%", p);
	fflush(stdout);
	while(r < numR){
		while(n < numN){
			NList[n] = currentN;
			core(error, maxError, currentN, currentR);
			errorList[n+r*numN] = *error;
			maxErrorList[n+r*numN] = *maxError;
			n += 1;
			currentN += dN;
			pnew = (int) (100 * (n + r * numN)) / (numN * numR);
			if(pnew > p){
				p = pnew;
				if(p <= 10){
					printf("\b\b%d%%", p);
					fflush(stdout);
				}else{
					printf("\b\b\b%d%%", p);
					fflush(stdout);
				}
			}
		}
		RList[r] = currentR;
		r += 1;
		currentR += dR;
		n = 0;
		currentN = minN;
	}
	printf("\nDone!\n");
	
	printf("Exporting Data...");
	exportData(errorFile, NList, RList, errorList, numN, numR);
	exportData(maxErrorFile, NList, RList, maxErrorList, numN, numR);
	printf("The data is %d x %d.\n", numN, numR);
	free(error);
	free(maxError);
	
	return 0;
	
}

__global__ void real2complex(cufftDoubleComplex *c, double *a, int n){
    /* compute idx and idy, the location of the element in the original NxN array */
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idxX < n && idxY < n){
		int idx = idxX + idxY * n;
		c[idx].x = a[idx];
		c[idx].y = 0.0;
	}
	
}

__global__ void complex2real_scaled(double *a, cufftDoubleComplex *c, double scale, int n){
    /* Compute index X and index Y, the location of the element in the original NxN array */
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idxX < n && idxY < n){
		int idx = idxX + idxY * n;
		a[idx] = scale * c[idx].x;
	}
	
}

__global__ void solve_poisson(cufftDoubleComplex *c, double *kx, double *ky, int n){
    /* compute idxX and idxY, the location of the element in the original NxN array */
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (idxX < n && idxY < n){
        int idx = idxX + idxY * n;
        double scale = -(kx[idxX] * kx[idxX] + ky[idxY] * ky[idxY]);
		
        if(idxX == 0 && idxY == 0){
			scale = 1.0;
		}
		
        scale = 1.0 / scale;
        c[idx].x *= scale;
        c[idx].y *= scale;
	}
	
}

void core(double *error, double *maxError, const int n, const int range){

	double *X = (double *)malloc(sizeof(double) * n);
	double *Y = (double *)malloc(sizeof(double) * n);
	double *kx = (double *)malloc(sizeof(double) * n);
	double *ky = (double *)malloc(sizeof(double) * n);
	double *r = (double *)malloc(sizeof(double) * n * n);
	double *r2 = (double *)malloc(sizeof(double) * n * n);
	double *u = (double *)malloc(sizeof(double) * n * n);
	
	const double EPSILON = 8.85418782 * pow(10, -12); // Permitivity of free space
	const double PI = 4 * atan(1);
	
	double *kx_d, *ky_d, *r_d;
	cufftDoubleComplex *r_complex_d;
	cudaMalloc((void **)&kx_d, sizeof(double) * n);
	cudaMalloc((void **)&ky_d, sizeof(double) * n);
	cudaMalloc((void **)&r_d, sizeof(double) * n * n);
	cudaMalloc((void **)&r_complex_d, sizeof(cufftDoubleComplex) * n * n);
	
	int m = 0;
	double deltaX = (double)range / (n / 2);
	double deltaK = 1.0 / (2 * range);
	for(int i = n/-2; i < n/2; i++){
		if(m < n){
			X[m] = i * deltaX; 
			Y[m] = i * deltaX; 
		}
		m += 1;
	}
	m = 0;
	for(int i = 0; i < n/2; i++){
		if(m < n/2){
			kx[m] = i * deltaK;
			kx[m+n/2] = (double)(i - n / 2) * deltaK;
			ky[m] = i * deltaK;
			ky[m+n/2] = (double)(i - n / 2) * deltaK;
		}
		m += 1;
	}
	
	gaussian(r, X, Y, n);	// Generate a Gaussian Distribution for r
	
	for (int i = 0; i < n * n; i++){
        u[i] = 0.0;
	}
	
	cudaMemcpy(kx_d, kx, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(ky_d, ky, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(r_d, r, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftPlan2d(&plan, n, n, CUFFT_Z2Z);

    // Compute the execution configuration
	dim3 dimBlock(TPBx, TPBy);
	dim3 dimGrid(n / dimBlock.x, n / dimBlock.y);
	
    // Handle N not multiple of TPBx or TPBy
	if(n % TPBx != 0){
		dimGrid.x += 1;
	}
	if(n % TPBy != 0){
		dimGrid.y += 1;
	}
	
	const double PI2 = 4 * PI * PI;
	double scale = 1.0 / (n * n * PI2);
	real2complex<<<dimGrid, dimBlock>>>(r_complex_d, r_d, n);
	cufftExecZ2Z(plan, r_complex_d, r_complex_d, CUFFT_FORWARD);
	solve_poisson<<<dimGrid, dimBlock>>>(r_complex_d, kx_d, ky_d, n);
	cufftExecZ2Z(plan, r_complex_d, r_complex_d, CUFFT_INVERSE);
	complex2real_scaled<<<dimGrid, dimBlock>>>(r_d, r_complex_d, scale, n);

	cudaMemcpy(u, r_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
	
	getR2(r2, u, deltaX, n);
	getError(error, maxError, r, r2, n);

    // Destroy plan and clean up memory on device
	free(kx);
	free(ky);
	free(X);
	free(Y);
	free(r);
	free(r2);
	free(u);
	cufftDestroy(plan);
	cudaFree(r_d);
	cudaFree(r_complex_d);
	cudaFree(kx_d);
	cudaFree(ky_d);

}

void exportData(const char *file, const double *X, const double *Y, const double *Z, const int m, const int n){
	
	FILE *dataFile = fopen(file, "w");
	
	if(dataFile != NULL){
		for(int j = 0; j < n ; j++){
			for(int i = 0; i < m; i++){ 
				fprintf(dataFile, "%lf\t%lf\t%lf\n", X[i], Y[j], Z[i+j*m]);
			}
		}
		printf("All data have been stored in \"%s\".\n", file);
		fclose(dataFile);
	}else{
		printf("File not found!");
	}
	
}

void gaussian(double *bin, const double *X, const double *Y, const int n){

	const double PI = 4 * atan(1);
	double x, y;
	
	// Generate required function
	double scale = 10.0 / sqrt(2 * PI);
	for(int j = 0; j < n; j++){
        for(int i = 0; i < n; i++){
			x = X[i] * X[i];
			y = Y[j] * Y[j];
			bin[i+j*n] = scale * exp(-(x + y) / 2);
		}
	}
	// Fix boundary
	for(int i = 0; i < n; i++){
		bin[i+(n-1)*n] = bin[i];
		bin[(n-1)+i*n] = bin[i*n];
	}

}

void getError(double *error, double *maxError, const double *data, const double *result, const int n){

	double difference = 0.0;
	double totalError = 0.0;
	double mError = 0.0;
	
	int count = 0;
	for(int j = 0; j < n; j++){
		for(int i = 0; i < n; i++){
			if (abs(result[i+j*n])>0 && abs(data[i+j*n])>0){
				difference = (double) abs(result[i+j*n] - data[i+j*n]);
				totalError += difference; 
				if(difference > mError){
					mError = difference;
				}
				count += 1;
			}
		}
	}
	*error = (double) totalError / count;
	*maxError = mError;

}

void getR2(double *result, const double *data, const double delta, const int n){

	const double iDelta2 = 1.0 / (delta * delta);
	const double scale = iDelta2;
	// Fix boundary to be zero
	for(int i = 0; i < n; i++){
		result[i+0*n] = 0.0;
		result[i+(n-1)*n] = 0.0;
		result[0+i*n] = 0.0;
		result[(n-1)+i*n] = 0.0;
	}
	// Finite Difference
	for(int j = 1; j < n - 1; j++){
		for(int i = 1; i < n - 1; i++){
			result[i+j*n] = scale * (data[(i-1)+j*n] + data[(i+1)+j*n]
									 + data[i+(j-1)*n] + data[i+(j+1)*n]
									 - 4 * data[i+j*n]);
		}
	}

}