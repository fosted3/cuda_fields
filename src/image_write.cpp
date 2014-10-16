#include <fstream>
#include <FreeImage.h>
#include <iostream>
#include <vector>
#include <stdlib.h>

void read_data(const char *infile, std::vector<std::vector<float> > &data)
{
	std::ifstream handle;
	handle.open(infile);
	if (handle.is_open())
	{
		unsigned int x = 0;
		float temp;
		while (handle.good())
		{
			char last = ',';
			data.push_back(std::vector<float>(0));
			while(last == ',' && handle.good())
			{
				handle >> temp;
				handle >> last;
				data[x].push_back(temp);
			}
			if (handle.good()) { handle.seekg(-sizeof(char)-1, std::ios::cur); }
			x++;
		}
		handle.close();
	}
}

float clamp(float a, float x, float b)
{
	if (x < a) { return a; }
	else if (x > b) { return b; }
	return x;
}

void write_image(std::vector<std::vector<float> > &data, const char *outfile, float scale)
{
	FreeImage_Initialise();
	unsigned int img_w = data.size();
	unsigned int img_h = data[0].size();
	FIBITMAP *image = FreeImage_Allocate(img_w, img_h, 24);
	RGBQUAD pixel; //Color variable
	if (!image)
	{
		std::cerr << "Can't allocate memory for image. Exiting." << std::endl;
		exit(1);
	}
	for (unsigned int x = 0; x < img_w; x++)
	{
		for (unsigned int y = 0; y < img_h; y++)
		{
			pixel.rgbRed = (int) clamp(0, data[x][y] * scale, 255);
			pixel.rgbBlue = (int) clamp(0, data[x][y] * scale, 255);
			pixel.rgbGreen = (int) clamp(0, data[x][y] * scale, 255);
			FreeImage_SetPixelColor(image, x, y, &pixel);
		}
	}
	if (!FreeImage_Save(FIF_PNG, image, outfile, 0)) //Make sure the image is saved
	{
		std::cerr << "Cannot save " << outfile << ". Exiting." << std::endl;
		exit(1);
	}
	FreeImage_Unload(image);
	FreeImage_DeInitialise();
}

int main(int argc, char** argv)
{
	if (argc != 3 && argc != 4)
	{
		std::cout << "Usage: " << argv[0] << " [input data] [output image] [[scale]]" << std::endl;
		exit(1);
	}
	std::vector<std::vector<float> > data;
	read_data(argv[1], data);
	if (data.size() == 0)
	{
		std::cout << "Can't open " << argv[1] << " or file is empty." << std::endl;
		exit(1);
	}
	if (argc == 4)
	{
		write_image(data, argv[2], atof(argv[3]));
	}
	else
	{
		write_image(data, argv[2], 1);
	}
	return 0;
}
