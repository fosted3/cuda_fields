#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdarg.h>

struct settings
{
	unsigned int iterations;
	unsigned int x;
	unsigned int y;
	unsigned int z;
	unsigned int n;
	bool dim_3d;
	unsigned int threads;
	unsigned long read_only;
};

bool read(char *token, const char *delim, int argc, ...)
{
	va_list argv;
	va_start(argv, argc);
	unsigned int *temp_int;
	float *temp_float;
	unsigned int i;
	for (i = 0; i < argc; i++)
	{
		token = strtok(NULL, delim);
		if (i == argc - 1)
		{
			temp_float = (float *) va_arg(argv, float *);
			if (token != NULL) { *temp_float = atof(token); }
			else { return false; }
		}
		else
		{
			temp_int = (unsigned int *) va_arg(argv, unsigned int*);
			if (token != NULL) { *temp_int = atoi(token); }
			else { return false; }
		}
	}
	return true;
}

unsigned int next_pow(unsigned int x)
{
	unsigned int n = 1;
	while (n < x) { n <<= 1; }
	return n;
}

unsigned int __max(unsigned int a, unsigned int b)
{
	if (a > b) { return a; }
	return b;
}

__global__ void iterate2d(float *from, float *to, bool *read_only, unsigned int N)
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

__global__ void iterate3d(float *from, float *to, bool *read_only, unsigned int N, unsigned int NN)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (read_only[x+(y*N)+(z*NN)] == false && x != 0 && y != 0 && z != 0 && x != (N - 1) && y != (N - 1) && z != (N - 1))
	{
		to[x+(y*N)+(z*NN)] = (from[(x+1)+((y+1)*N)+((z+1)*NN)] + from[(x+1)+((y+1)*N)+((z-1)*NN)] + from[(x+1)+((y-1)*N)+((z+1)*NN)] + from[(x+1)+((y-1)*N)+((z-1)*NN)] + from[(x-1)+((y+1)*N)+((z+1)*NN)] + from[(x-1)+((y+1)*N)+((z-1)*NN)] + from[(x-1)+((y-1)*N)+((z+1)*NN)] + from[(x-1)+((y-1)*N)+((z-1)*NN)]) / 8;
	}
	else
	{
		to[x+(y*N)+(z*NN)] = from[x+(y*N)+(z*NN)];
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
		else if (strcmp(token, "x") == 0)
		{
			token = strtok(NULL, delimiter);
			if (token == NULL) { break; }
			else { config.x = atoi(token); }
		}
		else if (strcmp(token, "y") == 0)
		{
			token = strtok(NULL, delimiter);
			if (token == NULL) { break; }
			else { config.y = atoi(token); }
		}
		else if (strcmp(token, "z") == 0)
		{
			token = strtok(NULL, delimiter);
			if (token == NULL) { break; }
			else { config.z = atoi(token); }
		}
		else if (strcmp(token, "threads") == 0)
		{
			token = strtok(NULL, delimiter);
			if (token == NULL) { break; }
			else { config.threads = atoi(token); }
		}
		else if (strcmp(token, "2d") == 0)
		{
			config.dim_3d = false;
		}
		else if (strcmp(token, "3d") == 0)
		{
			config.dim_3d = true;
		}
		token = strtok(NULL, delimiter);
	}
	free(buf);
	fclose(handle);
}

void init2d(char *settings_file, settings &config, float *data, bool *read_only)
{
	FILE *handle = fopen(settings_file, "r");
	fseek(handle, 0L, SEEK_END);
	unsigned long size = ftell(handle);
	fseek(handle, 0L, SEEK_SET);
	char *buf = (char*) malloc((size + 1) * sizeof(char));
	for(unsigned long i = 0; i < config.n * config.n; i++)
	{
		data[i] = 0.0f;
		read_only[i] = false;
	}
	config.read_only = 0;
	fread(buf, size, 1, handle);
	const char *delimiter = " \n";
	char *token = strtok(buf, delimiter);
	unsigned int x0;
	unsigned int x1;
	unsigned int y0;
	unsigned int y1;
	float f;
	unsigned int x;
	unsigned int y;
	while (token != NULL)
	{
		if (strcmp(token, "point") == 0) //point located at (x,y), syntax: point [x] [y] [value]
		{
			assert(read(token, delimiter, 3, &x, &y, &f));
			assert(x < config.n);
			assert(y < config.n);
			if (read_only[x+(y*config.n)] == false)
			{
				read_only[x+(y*config.n)] = true;
				config.read_only ++;
			}
			data[x+(y*config.n)] = f;
		}
		if (strcmp(token, "yline") == 0) //line in y direction, syntax: yline [xpos] [ystart] [yend] [value]
		{
			assert(read(token, delimiter, 4, &x, &y0, &y1, &f));
			assert(y0 < y1);
			assert(x < config.n);
			assert(y1 < config.n);
			for (y = y0; y <= y1; y++)
			{
				if (read_only[x+(y*config.n)] == false)
				{
					read_only[x+(y*config.n)] = true;
					config.read_only ++;
				}
				data[x+(y*config.n)] = f;
			}
		}
		else if (strcmp(token, "xline") == 0) //line in x direction, syntax: xline [ypos] [xstart] [xend] [value]
		{
			assert(read(token, delimiter, 4, &y, &x0, &x1, &f));
			assert(x0 < x1);
			assert(y < config.n);
			assert(x1 < config.n);
			for (x = x0; x <= x1; x++)
			{
				assert(x < config.n);
				if (read_only[x+(y*config.n)] == false)
				{
					read_only[x+(y*config.n)] = true;
					config.read_only ++;
				}
				data[x+(y*config.n)] = f;
			}
		}
		else if (strcmp(token, "rectangle") == 0) //rectangle from (x0, y0) to (x1, y1), syntax: square [x0] [y0] [x1] [y1] [value]
		{
			assert(read(token, delimiter, 5, &x0, &y0, &x1, &y1, &f));
			assert(x0 < x1);
			assert(y0 < y1);
			assert(x1 < config.n);
			assert(y1 < config.n);
			for (x = x0; x <= x1; x++)
			{
				for (y = y0; y <= y1; y++)
				{
					if (read_only[x+(y*config.n)] == false)
					{
						read_only[x+(y*config.n)] = true;
						config.read_only ++;
					}
					data[x+(y*config.n)] = f;
				}
			}
		}
		token = strtok(NULL, delimiter);
	}
	for (x = config.x; x < config.n; x++)
	{
		for (y = config.y; y < config.n; y++)
		{
			if (read_only[x+(y*config.n)] == false)
			{
				read_only[x+(y*config.n)] = true;
				config.read_only ++;
			}
		}
	}
	free(buf);
	fclose(handle);
}

void init3d(char *settings_file, settings &config, float *data, bool *read_only)
{
	FILE *handle = fopen(settings_file, "r");
	fseek(handle, 0L, SEEK_END);
	unsigned long size = ftell(handle);
	fseek(handle, 0L, SEEK_SET);
	char *buf = (char*) malloc((size + 1) * sizeof(char));
	for(unsigned long i = 0; i < config.n * config.n * config.n; i++)
	{
		data[i] = 0.0f;
		read_only[i] = false;
	}
	config.read_only = 0;
	fread(buf, size, 1, handle);
	const char *delimiter = " \n";
	char *token = strtok(buf, delimiter);
	unsigned int x0;
	unsigned int x1;
	unsigned int y0;
	unsigned int y1;
	unsigned int z0;
	unsigned int z1;
	float f;
	unsigned int x;
	unsigned int y;
	unsigned int z;
	while (token != NULL)
	{
		if (strcmp(token, "point") == 0) //point located at (x,y,z), syntax: point [x] [y] [z] [value]
		{
			assert(read(token, delimiter, 4, &x, &y, &z, &f));
			assert(x < config.n);
			assert(y < config.n);
			assert(z < config.n);
			if (read_only[x+(y*config.n)+(z*config.n*config.n)] == false)
			{
				config.read_only ++;
				read_only[x+(y*config.n)+(z*config.n*config.n)] = true;
			}
			data[x+(y*config.n)+(z*config.n*config.n)] = f;
			
		}
		if (strcmp(token, "zline") == 0) //line in z direction, syntax: zline [xpos] [ypos] [zstart] [zend] [value]
		{
			assert(read(token, delimiter, 5, &x, &y, &z0, &z1, &f));
			assert(z0 < z1);
			assert(x < config.n);
			assert(y < config.n);
			assert(z1 < config.n);
			for (z = z0; z <= z1; z++)
			{
				if (read_only[x+(y*config.n)+(z*config.n*config.n)] == false)
				{
					config.read_only ++;
					read_only[x+(y*config.n)+(z*config.n*config.n)] = true;
				}
				data[x+(y*config.n)+(z*config.n*config.n)] = f;
			}
		}
		else if (strcmp(token, "yline") == 0) //line in y direction, syntax: yline [xpos] [zpos] [ystart] [yend] [value]
		{
			assert(read(token, delimiter, 5, &x, &z, &y0, &y1, &f));
			assert(y0 < y1);
			assert(x < config.n);
			assert(y1 < config.n);
			assert(z < config.n);
			for (y = y0; y <= y1; y++)
			{
				assert(y < config.n);
				if (read_only[x+(y*config.n)+(z*config.n*config.n)] == false)
				{
					config.read_only ++;
					read_only[x+(y*config.n)+(z*config.n*config.n)] = true;
				}
				data[x+(y*config.n)+(z*config.n*config.n)] = f;
			}
		}
		else if (strcmp(token, "xline") == 0) //line in x direction, syntax: xline [ypos] [zpos] [xstart] [xend] [value]
		{
			assert(read(token, delimiter, 5, &y, &z, &x0, &x1, &f));
			assert(x0 < x1);
			assert(x1 < config.n);
			assert(y < config.n);
			assert(z < config.n);
			for (x = x0; x <= x1; x++)
			{
				if (read_only[x+(y*config.n)+(z*config.n*config.n)] == false)
				{
					config.read_only ++;
					read_only[x+(y*config.n)+(z*config.n*config.n)] = true;
				}
				data[x+(y*config.n)+(z*config.n*config.n)] = f;
			}
		}
		else if (strcmp(token, "zrectangle") == 0) //rectangle from (x0, y0) to (x1, y1) in z plane, syntax: zrectangle [x0] [y0] [x1] [y1] [z] [value]
		{
			assert(read(token, delimiter, 6, &x0, &y0, &x1, &y1, &z, &f));
			assert(x0 < x1);
			assert(y0 < y1);
			assert(x1 < config.n);
			assert(y1 < config.n);
			assert(z < config.n);
			for (x = x0; x <= x1; x++)
			{
				for (y = y0; y <= y1; y++)
				{
					if (read_only[x+(y*config.n)+(z*config.n*config.n)] == false)
					{
						config.read_only ++;
						read_only[x+(y*config.n)+(z*config.n*config.n)] = true;
					}
					data[x+(y*config.n)+(z*config.n*config.n)] = f;
				}
			}
		}
		else if (strcmp(token, "yrectangle") == 0) //rectangle from (x0, z0) to (x1, z1) in y plane, syntax yrectangle [x0] [z0] [x1] [z1] [y] [value]
		{
			assert(read(token, delimiter, 6, &x0, &z0, &x1, &z1, &y, &f));
			assert(x0 < x1);
			assert(z0 < z1);
			assert(x1 < config.n);
			assert(y < config.n);
			assert(z1 < config.n);
			for (x = x0; x <= x1; x++)
			{
				for (z = z0; z <= z1; z++)
				{
					if (read_only[x+(y*config.n)+(z*config.n*config.n)] == false)
					{
						config.read_only ++;
						read_only[x+(y*config.n)+(z*config.n*config.n)] = true;
					}
					data[x+(y*config.n)+(z*config.n*config.n)] = f;
				}
			}
		}
		else if (strcmp(token, "xrectangle") == 0) ////rectangle from (y0, z0) to (y1, z1) in x plane, syntax xrectangle [y0] [z0] [y1] [z1] [x] [value]
		{
			assert(read(token, delimiter, 6, &y0, &z0, &y1, &z1, &x, &f));
			assert(y0 < y1);
			assert(z0 < z1);
			assert(x < config.n);
			assert(y1 < config.n);
			assert(z1 < config.n);	
			for (y = y0; y <= y1; y++)
			{
				for (z = z0; z <= z1; z++)
				{
					if (read_only[x+(y*config.n)+(z*config.n*config.n)] == false)
					{
						config.read_only ++;
						read_only[x+(y*config.n)+(z*config.n*config.n)] = true;
					}
					data[x+(y*config.n)+(z*config.n*config.n)] = f;
				}
			}
		}
		else if (strcmp(token, "prism") == 0) //prism from (x0, y0, z0) to (x1, y1, z1), syntax prism [x0] [y0] [z0] [x1] [y1] [z1] [value]
		{
			assert(read(token, delimiter, 7, &x0, &y0, &z0, &x1, &y1, &z1, &f));
			assert(x0 < x1);
			assert(y0 < y1);
			assert(z0 < z1);
			assert(x1 < config.n);
			assert(y1 < config.n);
			assert(z1 < config.n);
			for (x = x0; x <= x1; x++)
			{
				for (y = y0; y <= y1; y++)
				{
					for (z = z0; z <= z1; z++)
					{
						if (read_only[x+(y*config.n)+(z*config.n*config.n)] == false)
						{
							config.read_only ++;
							read_only[x+(y*config.n)+(z*config.n*config.n)] = true;
						}
						data[x+(y*config.n)+(z*config.n*config.n)] = f;
					}
				}
			}
		}
		token = strtok(NULL, delimiter);
	}
	for (x = config.x; x < config.n; x++)
	{
		for (y = config.y; y < config.n; y++)
		{
			for (z = config.z; z < config.n; z++)
			{
				if (read_only[x+(y*config.n)+(z*config.n*config.n)] == false)
				{
					read_only[x+(y*config.n)+(z*config.n*config.n)] = true;
					config.read_only ++;
				}
			}
		}
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
	dim3 block(1,1,1);
	dim3 grid(1,1,1);
	unsigned long float_mem;
	unsigned long bool_mem;
	config.iterations = 0;
	config.n = 0;
	config.threads = 512;
	config.x = 0;
	config.y = 0;
	config.z = 0;
	config.dim_3d = false;
	char *settings_file = argv[1];
	read_config(settings_file, config);
	switch (config.threads)
	{
		case 2048:
			block.x = 32; block.y = 32; block.z = 2;
		case 1024:
			block.x = 32; block.y = 32; block.z = 1;
		case 512:
			block.x = 16; block.y = 16; block.z = 2;
		case 256:
			block.x = 16; block.y = 16; block.z = 1;
		case 128:
			block.x =  8; block.y =  8; block.z = 2;
		case 64:
			block.x =  8; block.y =  8; block.z = 1;
		case 32:
			block.x =  4; block.y =  4; block.z = 2;
	}
	if (!config.dim_3d)
	{
		block.z = 1;
	}
	assert(config.n % block.x == 0);
	assert(config.n % block.y == 0);
	assert(config.n % block.z == 0);
	if (config.dim_3d)
	{
		config.n = next_pow(__max(__max(config.x, config.y), config.z));
		grid.x = config.n / block.x;
		grid.y = config.n / block.y;
		grid.z = config.n / block.z;
		float_mem = config.n * config.n * config.n * sizeof(float);
		bool_mem = config.n * config.n * config.n * sizeof(bool);
	}
	else
	{
		config.n = next_pow(__max(config.x, config.y));
		grid.x = config.n / block.x;
		grid.y = config.n / block.y;
		float_mem = config.n * config.n * sizeof(float);
		bool_mem = config.n * config.n * sizeof(bool);
	}
	float *data = (float *)malloc(float_mem);
	bool *read_only = (bool *)malloc(bool_mem);
	if (config.dim_3d) { init3d(settings_file, config, data, read_only); }
	else { init2d(settings_file, config, data, read_only); }
	float *d_z_1;
	float *d_z_2;
	bool *d_read_only;
	cudaEvent_t start;
	cudaEvent_t stop;
	float compute_time;
	double gflops;
	assert(cudaSuccess == cudaEventCreate(&start));
	assert(cudaSuccess == cudaEventCreate(&stop));
	assert(cudaSuccess == cudaMalloc((void**) &d_z_1, float_mem));
	assert(cudaSuccess == cudaMalloc((void**) &d_z_2, float_mem));
	assert(cudaSuccess == cudaMalloc((void**) &d_read_only, bool_mem));
	assert(cudaSuccess == cudaMemcpy(d_z_1, data, float_mem, cudaMemcpyHostToDevice));
	assert(cudaSuccess == cudaMemcpy(d_read_only, read_only, bool_mem, cudaMemcpyHostToDevice));
	assert(cudaSuccess == cudaEventRecord(start, 0));
	for (unsigned int i = 0; i < config.iterations; i++)
	{
		if (config.dim_3d)
		{
			iterate3d<<<grid, block>>>(d_z_1, d_z_2, d_read_only, config.n, config.n * config.n);
			cudaThreadSynchronize();
			iterate3d<<<grid, block>>>(d_z_2, d_z_1, d_read_only, config.n, config.n * config.n);
			cudaThreadSynchronize();
			printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bIterations: %u", i);
			fflush(stdout);
		}
		else
		{
			iterate2d<<<grid, block>>>(d_z_1, d_z_2, d_read_only, config.n);
			cudaThreadSynchronize();
			iterate2d<<<grid, block>>>(d_z_2, d_z_1, d_read_only, config.n);
			cudaThreadSynchronize();
			if (i % 500 == 0) { printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bIterations: %u", i); }
		}
	}
	assert(cudaSuccess == cudaEventRecord(stop, 0));
	assert(cudaSuccess == cudaEventSynchronize(stop));
	assert(cudaSuccess == cudaEventElapsedTime(&compute_time, start, stop));
	printf("\n");
	printf("Compute time: %fms\n", compute_time);
	if (config.dim_3d) { gflops = ((config.n * config.n * config.n) - config.read_only) * 16.0 * config.iterations / (compute_time * 1000000.0); }
	else { 	gflops = ((config.n * config.n) - config.read_only) * 8.0 * config.iterations / (compute_time * 1000000.0); }
	printf("Compute speed: %f GFLOPS\n", gflops);
	assert(cudaSuccess == cudaMemcpy(data, d_z_1, float_mem, cudaMemcpyDeviceToHost));
	FILE *handle = fopen(argv[2], "w");
	assert(handle != NULL);
	if (config.dim_3d)
	{
		for (unsigned int x = 0; x < config.x; x++)
		{
			for (unsigned int y = 0; y < config.y; y++)
			{
				for (unsigned int z = 0; z < config.z; z++)
				{
					fprintf(handle, "%u, %u, %u, %f\n", x, y, z, data[x+(y*config.n)+(z*config.n*config.n)]);
				}
			}
		}
	}
	else
	{
		for (unsigned int y = 0; y < config.n; y++)
		{
			for (unsigned int x = 0; x < config.n; x++)
			{
				fprintf(handle, "%06.2f", data[x+(y*config.n)]);
				if (x == config.n - 1) { fprintf(handle, "\n"); }
				else { fprintf(handle, ", "); }
			}
		}
	}
	fclose(handle);
	assert(cudaSuccess == cudaFree(d_z_1));
	assert(cudaSuccess == cudaFree(d_z_2));
	assert(cudaSuccess == cudaFree(d_read_only));
	free(data);
	free(read_only);
	return 0;	
}
