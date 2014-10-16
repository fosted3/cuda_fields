#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

struct settings
{
	unsigned int iterations;
	unsigned int n;
	unsigned int block_dim;
};

__global__ void iterate(float *from, float *to, bool *read_only, unsigned int N)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (read_only[x+(y*N)] == false && x != 0 && y != 0 && x != (N - 1) && y != (N - 1))
	{
		to[x+(y*N)] = (from[(x+1)+((y+1)*N)] + from[(x+1)+((y-1)*N)] + from[(x-1)+((y+1)*N)] + from[(x-1)+((y-1)*N)]) / 4;
	}
	else
	{
		to[x+(y*N)] = from[x+(y*N)];
	}
}

void read_config(char *settings_file, settings &config)
{
	FILE *handle = fopen(settings_file, "r");
	fseek(handle, 0L, SEEK_END);
	unsigned long size = ftell(handle);
	fseek(handle, 0L, SEEK_SET);
	char *buf = (char*) malloc((size + 1) * sizeof(char));
	memset(buf, 0, size + 1);
	fread(buf, size, 1, handle);
	const char *delimiter = " \n";
	char *token = strtok(buf, delimiter);
	while (token != NULL)
	{
		if (strcmp(token, "iterations") == 0)
		{
			token = strtok(NULL, delimiter);
			if (token == NULL) { break; }
			else { config.iterations = atoi(token); }
		}
		else if (strcmp(token, "n") == 0)
		{
			token = strtok(NULL, delimiter);
			if (token == NULL) { break; }
			else { config.n = atoi(token); }
		}
		else if (strcmp(token, "block_dim") == 0)
		{
			token = strtok(NULL, delimiter);
			if (token == NULL) { break; }
			else { config.block_dim = atoi(token); }
		}
		token = strtok(NULL, delimiter);
	}
	free(buf);
	fclose(handle);
}

void init(char *settings_file, settings &config, float *z, bool *read_only)
{
	FILE *handle = fopen(settings_file, "r");
	fseek(handle, 0L, SEEK_END);
	unsigned long size = ftell(handle);
	fseek(handle, 0L, SEEK_SET);
	char *buf = (char*) malloc((size + 1) * sizeof(char));
	for(unsigned long i = 0; i < config.n * config.n; i++)
	{
		z[i] = 0.0f;
		read_only[i] = false;
	}
	fread(buf, size, 1, handle);
	const char *delimiter = " \n";
	char *token = strtok(buf, delimiter);
	unsigned int a;
	unsigned int b;
	unsigned int c;
	unsigned int d;
	float f;
	unsigned int x;
	unsigned int y;
	while (token != NULL)
	{
		if (strcmp(token, "yline") == 0) //line in y direction, syntax: yline [xpos] [ystart] [yend] [value]
		{
			token = strtok(NULL, delimiter);
			if (token != NULL) { a = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { b = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { c = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { f = atof(token); }
			else { break; }
			x = a;
			//printf("yline %u %u %u %f\n", a, b, c, f);
			for (y = b; y <= c; y++)
			{
				//printf("(%u, %u)\n", x, y);
				assert(y < config.n);
				read_only[x+(y*config.n)] = true;
				z[x+(y*config.n)] = f;
			}
		}
		else if (strcmp(token, "xline") == 0) //line in x direction, syntax: xline [ypos] [xstart] [xend] [value]
		{
			token = strtok(NULL, delimiter);
			if (token != NULL) { a = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { b = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { c = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { f = atof(token); }
			else { break; }
			y = a;
			//printf("xline %u %u %u %f\n", a, b, c, f);
			for (x = b; x <= c; x++)
			{
				//printf("(%u, %u)\n", x, y);
				assert(x < config.n);
				read_only[x+(y*config.n)] = true;
				z[x+(y*config.n)] = f;
			}
		}
		else if (strcmp(token, "rectangle") == 0) //rectangle from (x0, y0) to (x1, y1), syntax: square [x0] [y0] [x1] [y1] [value]
		{
			token = strtok(NULL, delimiter);
			if (token != NULL) { a = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { b = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { c = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { d = atoi(token); }
			else { break; }
			token = strtok(NULL, delimiter);
			if (token != NULL) { f = atof(token); }
			else { break; }
			//printf("rectangle %u %u %u %u %f\n", a, b, c, d, f);
			assert(a < c);
			assert(b < d);
			for (x = a; x <= c; x++)
			{
				for (y = b; y <= d; y++)
				{
					//printf("(%u, %u)\n", x, y);
					assert(x < config.n);
					assert(y < config.n);
					read_only[x+(y*config.n)] = true;
					z[x+(y*config.n)] = f;
				}
			}
		}
		token = strtok(NULL, delimiter);
	}
	free(buf);
	fclose(handle);
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		printf("Usage: %s [config file] [output]\n", argv[0]);
		exit(1);
	}
	settings config;
	config.iterations = 0;
	config.n = 0;
	config.block_dim = 0;
	char *settings_file = argv[1];
	read_config(settings_file, config);
	//printf("iterations: %u, n: %u, block_dim: %u\n", config.iterations, config.n, config.block_dim);
	//exit(0);
	dim3 block(config.block_dim, config.block_dim);
	dim3 grid((int)ceil(config.n/block.x),(int)ceil(config.n/block.y));
	unsigned int float_mem = config.n * config.n * sizeof(float);
	unsigned int bool_mem = config.n * config.n * sizeof(bool);
	float *z = (float *)malloc(float_mem);
	bool *read_only = (bool *)malloc(bool_mem);
	init(settings_file, config, z, read_only);
	//set_initial(z, config.n);
	//set_reserved(read_only, config.n);
	/*for (unsigned int y = 0; y < config.n; y++)
	{
		for (unsigned int x = 0; x < config.n; x++)
		{
			printf("%06.2f ", z[x+(y*config.n)]);
			if (read_only[x+(y*config.n)]) { printf("t"); }
			else { printf("f"); }
			if (x == config.n - 1) { printf("\n"); }
			else { printf(", "); }
		}
	}
	exit(1);*/
	float *d_z_1;
	float *d_z_2;
	bool *d_read_only;
	assert(cudaSuccess == cudaMalloc((void**) &d_z_1, float_mem));
	assert(cudaSuccess == cudaMalloc((void**) &d_z_2, float_mem));
	assert(cudaSuccess == cudaMalloc((void**) &d_read_only, bool_mem));
	assert(cudaSuccess == cudaMemcpy(d_z_1, z, float_mem, cudaMemcpyHostToDevice));
	assert(cudaSuccess == cudaMemcpy(d_read_only, read_only, bool_mem, cudaMemcpyHostToDevice));
	for (unsigned int i = 0; i < config.iterations; i++)
	{
		iterate<<<grid, block>>>(d_z_1, d_z_2, d_read_only, config.n);
		iterate<<<grid, block>>>(d_z_2, d_z_1, d_read_only, config.n);
		if (i % 500 == 0) { printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bIterations: %u", i); }
	}
	printf("\n");
	assert(cudaSuccess == cudaMemcpy(z, d_z_1, float_mem, cudaMemcpyDeviceToHost));
	FILE *handle = fopen(argv[2], "w");
	assert(handle != NULL);
	for (unsigned int y = 0; y < config.n; y++)
	{
		for (unsigned int x = 0; x < config.n; x++)
		{
			fprintf(handle, "%06.2f", z[x+(y*config.n)]);
			if (x == config.n - 1) { fprintf(handle, "\n"); }
			else { fprintf(handle, ", "); }
		}
	}
	fclose(handle);
	assert(cudaSuccess == cudaFree(d_z_1));
	assert(cudaSuccess == cudaFree(d_z_2));
	assert(cudaSuccess == cudaFree(d_read_only));
	free(z);
	free(read_only);
	return 0;	
}
