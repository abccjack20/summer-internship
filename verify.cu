#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <unistd.h>

void importData(const char *file, float *X, float *Y, float *data, const int n);
float getDelta(const float *data, const int n);
void getR(float *result, const float *data, const float delta, const int n);
float getError(const float *data, const float *result, const int n);
void exportData(const char *file, const float *X, const float *Y, const float *Z, const int n);


int main(void){
	
	// Declaration
	int N = 2048;
	float delta, error;
	char *rFName = (char *)"r_data.dat";
	char *uFName = (char *)"u_data.dat";
	char *RFName = (char *)"R_data.dat";
		
	// Initialization
	printf("Input the sample size:\n");
	scanf("%d", &N); 
	printf("Allocating memory for the data...\n");
	float *rData = (float *)malloc(sizeof(float) * N * N);
	float *uData = (float *)malloc(sizeof(float) * N * N);
	float *r = (float *)malloc(sizeof(float) * N * N);
	float *X = (float *)malloc(sizeof(float) * N);
	float *Y = (float *)malloc(sizeof(float) * N);
	printf("Finish!\n");
	
	// Retrive Data
	printf("Retriving data from \"%s\"...\n", rFName);
	importData(rFName, X, Y, rData, N);
	printf("Finish!\n");
		
	printf("Retriving data from \"%s\"...\n", uFName);
	importData(uFName, X, Y, uData, N);
	printf("Finish!\n");
	
	// Evaluate error
	printf("Evaluating the average error...\n");
	delta = getDelta(X, N);
	printf("delta = %f\n", delta);
	getR(r, uData, delta, N);
	error = getError(rData, r, N);
	printf("Finish!\n");
	printf("Average error = %f\n", error);
	
	printf("Exporting R data...\n");
	exportData(RFName, X, Y, r, N);
	
	// Clean-up
	free(rData);
	free(uData);
	free(r);
	free(X);
	free(Y);
	
	return 0;

}

void importData(const char *file, float *X, float *Y, float *data, const int n){

	FILE *dataFile = fopen(file, "r");
	float *Xtmp = (float *)malloc(sizeof(float) * n * n);
	float *Ytmp = (float *)malloc(sizeof(float) * n * n);
	
	if(dataFile != NULL){
		for(int j = 0; j < n; j++){
			for(int i = 0; i < n; i++){
				fscanf(dataFile, "%f\t%f\t%f", &Xtmp[i+j*n], &Ytmp[i+j*n], &data[i+j*n]);
			}
		}
		printf("All data from \"%s\" have been retrieved.\n", file);
		fclose(dataFile);
	}else{
		printf("File not found!\n");
	}
	
	for(int i = 0; i < n; i++){
		X[i] = Xtmp[i];
		Y[i] = Ytmp[i*n];
	}
	
	free(Xtmp);
	free(Ytmp);
	
}

float getDelta(const float *X, const int n){
	
	float delta = 0.0f;
	for(int i = 1; i < n; i++){
		delta += abs(X[i] - X[i-1]);
	}
	delta /= (float)(n-1);
	return delta;
	
}

void getR(float *result, const float *data, const float delta, const int n){

	float iDelta2 = 1.0f / (delta * delta);
	const float EPSILON = 8.85418782 * pow(10, -12);
	// Fix boundary to be zero
	for(int i = 0; i < n; i++){
		result[i] = 0.0f;
		result[i+(n-1)*n] = 0.0f;
		result[0+i*n] = 0.0f;
		result[(n-1)+i*n] = 0.0f;
	}
	// Finite Difference
	for(int j = 1; j < n - 1; j++){
		for(int i = 1; i < n - 1; i++){
			result[i+j*n] = EPSILON * iDelta2 * (data[(i-1)+j*n] + data[(i+1)+j*n]
												 + data[i+(j-1)*n] + data[i+(j+1)*n]
									   			 - 4 * data[i+j*n]);
			result[i+j*n] *= n * n;
			/* if(result[i+j*n] != 0){
				printf("%d:%f ", i+j*n, result[i+j*n]);
			} */
		}
	}

}

float getError(const float *data, const float *result, const int n){

	float totalError = 0.0f;
	float averageError = 0.0f;
	
	for(int j = 0; j < n; j++){
		for(int i = 0; i < n; i++){
			totalError += abs(data[i+j*n] - result[i+j*n]);
		}
	}
	averageError = totalError / (float)(n * n);
	return averageError;

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