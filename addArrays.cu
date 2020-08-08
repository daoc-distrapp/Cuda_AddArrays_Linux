#include <stdio.h>

__global__
void addArrays(const int* A, const int* B, int* C) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	C[i] = A[i] + B[i];
}

int main(void) {
	// Crea los buffer en el host, para los datos de entrada y salida
	int* inputA = (int*)malloc(sizeof(int) * 1024);
	int* inputB = (int*)malloc(sizeof(int) * 1024);
	int* outputC = (int*)malloc(sizeof(int) * 1024);

	// Crea los buffer en la GPU, para los datos de entrada y salida
	int *gpuBuffer_A, *gpuBuffer_B, *gpuBuffer_C;
	cudaMalloc(&gpuBuffer_A, sizeof(int)*1024);
	cudaMalloc(&gpuBuffer_B, sizeof(int)*1024);
	cudaMalloc(&gpuBuffer_C, sizeof(int) * 1024);

	// Inicializa los buffer del host con los valores de entrada
	for (int i = 0; i < 1024; i++) {
		inputA[i] = i; //0,1,2,...,1023
		inputB[i] = 1023 - i; //1023,1022,...,0
	}

	// Copia los valores de entrada desde los buffer del host a los buffer de la GPU
	cudaMemcpy(gpuBuffer_A, inputA, sizeof(int) * 1024, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuBuffer_B, inputB, sizeof(int) * 1024, cudaMemcpyHostToDevice);

	// Ejecuta la kernel en la GPU (4 bloques * 256 hilos = 1024 elementos calculados)
	addArrays<<<4, 256>>>(gpuBuffer_A, gpuBuffer_B, gpuBuffer_C);

	// Recupera el resultado desde la GPU y lo pone en un buffer del host
	cudaMemcpy(outputC, gpuBuffer_C, sizeof(int) * 1024, cudaMemcpyDeviceToHost);

	// Presenta el resultado
	for (int i = 0; i < 1024; i++) {
		printf("Resultados %d: (%d + %d = %d)\n", i, inputA[i], inputB[i], outputC[i]);
	}

	// Libera los recursos
	cudaFree(gpuBuffer_A);
	cudaFree(gpuBuffer_B);
	cudaFree(gpuBuffer_C);
	free(inputA);
	free(inputB);
	free(outputC);
}
