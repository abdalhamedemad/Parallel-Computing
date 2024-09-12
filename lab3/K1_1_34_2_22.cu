// %%writefile k1.cu
#define STB_IMAGE_IMPLEMENTATION
#define K 3
#include <stdio.h>

#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <windows.h>
// #include <filesystem>
#include <string>
#include <vector>
#include <iostream>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define MASK_DIM 3

// __constant__ float mask[MASK_DIM * MASK_DIM];

// __global__
// void convolution_3D_batch_kernel(unsigned char *in, unsigned char *out, int w, int h, int comp, int batch_size) {
//     int Col = blockIdx.x * blockDim.x + threadIdx.x;
//     int Row = blockIdx.y * blockDim.y + threadIdx.y;
//     int image_idx = blockIdx.z;

//     if (Col < w && Row < h) {
//         int pixVal_R = 0;
//         int pixVal_G = 0;
//         int pixVal_B = 0;
//         int N_start_col = Col - (MASK_DIM/2);
//         int N_start_row = Row - (MASK_DIM/2);

//             if (image_idx < batch_size) {
//             int index_base = image_idx * (w * h * comp);
//             for(int j = 0; j < MASK_DIM; ++j) {
//                 for(int k = 0; k < MASK_DIM; ++k) {
//                     int curRow = N_start_row + j;
//                     int curCol = N_start_col + k;
//                     if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
//                         int index = index_base + ((curRow * w + curCol) * comp);
//                         float mask_value = mask[j * MASK_DIM + k];
//                         pixVal_R += in[index] * mask_value;
//                         pixVal_G += in[index + 1] * mask_value;
//                         pixVal_B += in[index + 2] * mask_value;
//                     }
//                 }
//             }
//         }

//         int index = ((image_idx * w * h) + (Row * w + Col));
//         out[index] = ((unsigned char)(pixVal_R) + (unsigned char)(pixVal_G) + (unsigned char)(pixVal_B))/3;
//     }
// }

__global__
void convolution_3D_batch_kernelDA(unsigned char *in, unsigned char *out, int w, int h, int comp, int batch_size, float *mask,int mask_dim) {
    extern __shared__ float shared_mask[];
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int image_idx = blockIdx.z;
    // if (threadIdx.x < mask_dim && threadIdx.y < mask_dim) {
    //     shared_mask[threadIdx.y * mask_dim + threadIdx.x] = mask[threadIdx.y * mask_dim + threadIdx.x];
    // }
    // the mask size could exceed the block size
    for (int i = threadIdx.y; i < mask_dim; i += blockDim.y) {
        for (int j = threadIdx.x; j < mask_dim; j += blockDim.x) {
            if ( i < mask_dim && j < mask_dim) {
                shared_mask[i * mask_dim + j] = mask[i * mask_dim + j];
            }
        }
    }
    __syncthreads();

    if (Col < w && Row < h) {
        float pixVal_R = 0;
        float pixVal_G = 0;
        float pixVal_B = 0;
        int N_start_col = Col - (mask_dim/2);
        int N_start_row = Row - (mask_dim/2);

        if (image_idx < batch_size) {
            int index_base = image_idx * (w * h * comp);
            for(int j = 0; j < mask_dim; ++j) {
                for(int k = 0; k < mask_dim; ++k) {
                    int curRow = N_start_row + j;
                    int curCol = N_start_col + k;
                    if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                        int index = index_base + ((curRow * w + curCol) * comp);
                        float mask_value = shared_mask[j * mask_dim + k];
                        pixVal_R += in[index] * mask_value;
                        pixVal_G += in[index + 1] * mask_value;
                        pixVal_B += in[index + 2] * mask_value;
                    }
                }
            }
        }

        int index = ((image_idx * w * h) + (Row * w + Col));
        // out[index] = ((unsigned char)(pixVal_R) + (unsigned char)(pixVal_G) + (unsigned char)(pixVal_B));
        // max between 0 and min between 255 and the value
        out[index] = (unsigned char) max(0, min(255, (unsigned char) (pixVal_R + pixVal_G + pixVal_B)));
    }
}

// std::vector<std::string> listFiles(const std::string &folderPath) {
//     std::vector<std::string> files;

//     try {
//         for (const auto &entry : std::filesystem::directory_iterator(folderPath)) {
//             if (!entry.is_directory()) {
//                 files.push_back(entry.path().filename().string());
//             }
//         }
//     } catch (const std::filesystem::filesystem_error &e) {
//         std::cerr << "Error reading directory: " << e.what() << std::endl;
//     }

//     return files;
// }

std::vector<std::string> listFiles(const std::string &folderPath) {
  std::vector<std::string> files;

      WIN32_FIND_DATA findFileData;
      HANDLE hFind = FindFirstFile((folderPath + "\\*").c_str(), &findFileData);
      if (hFind == INVALID_HANDLE_VALUE) {
          std::cerr << "Error opening directory! Error code: " << GetLastError() << std::endl;
          return files;
      }

      do {
          // Exclude directories "." and ".."
          if (strcmp(findFileData.cFileName, ".") != 0 && strcmp(findFileData.cFileName, "..") != 0) {
              // Add file name to vector
              files.push_back(findFileData.cFileName);
          }
      } while (FindNextFile(hFind, &findFileData) != 0);

      DWORD dwError = GetLastError();
      if (dwError != ERROR_NO_MORE_FILES) {
          std::cerr << "Error reading directory! Error code: " << dwError << std::endl;
      }

      FindClose(hFind);

      return files;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s input_folder_path output_folder_path batch_size mask.txt\n", argv[0]);
        return 1;
    }

    FILE *mask_file = fopen(argv[4], "r");
    if (!mask_file) {
        printf("Error opening mask file!\n");
        return 1;
    }

    int mask_dim;
    fscanf(mask_file, "%d", &mask_dim);
    float *host_mask = new float[mask_dim * mask_dim];
    for (int i = 0; i < mask_dim; ++i) {
        for (int j = 0; j < mask_dim; ++j) {
            fscanf(mask_file, "%f", &host_mask[i * mask_dim + j]);
        }
    }
    fclose(mask_file);
    float *d_mask;
    cudaMalloc(&d_mask, mask_dim * mask_dim * sizeof(float));
    cudaMemcpy(d_mask, host_mask, mask_dim * mask_dim * sizeof(float), cudaMemcpyHostToDevice);

    printf("Reading image...\n");

    std::vector<std::string> files = listFiles(argv[1]);
    int batch_size = atoi(argv[3]);
    int width, height, comp;
    unsigned char *d_in, *d_out;

    std::vector<unsigned char> imageData;
    int totalSize = 0;
    for (int b = 0; b < batch_size; ++b) {
        char input_image_path[100];
        sprintf(input_image_path, "%s/%s", argv[1], files[b].c_str());
        unsigned char *data = stbi_load(input_image_path, &width, &height, &comp, 0);
        int size = width * height * comp;
        totalSize += size;
        std::cout << "width = " << width << ", height = " << height << ", comp = " << comp << std::endl;
        imageData.insert(imageData.end(), data, data + size);
        stbi_image_free(data);
    }

    cudaMalloc(&d_in, batch_size * width * height * comp);
    cudaMalloc(&d_out, batch_size * width * height);
    std::cout << "Size of the vector: " << imageData.size() << std::endl;

    cudaMemcpy(d_in, imageData.data(), totalSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // convolution_3D_batch_kernel<<<dim3((width + 16 - 1) / 16, (height + 16 - 1) / 16, batch_size), dim3(16, 16)>>>(d_in, d_out, width, height, comp, batch_size);
        convolution_3D_batch_kernelDA<<<dim3((width + 16 - 1) / 16, (height + 16 - 1) / 16, batch_size), dim3(16, 16), mask_dim * mask_dim * sizeof(float)>>>(d_in, d_out, width, height, comp, batch_size, d_mask,mask_dim);
        unsigned char *out = (unsigned char *)malloc(batch_size * width * height);
        cudaMemcpy(out, d_out, batch_size * width * height, cudaMemcpyDeviceToHost);

        for (int b = 0; b < batch_size; ++b) {
            char output_image_path[100];
            sprintf(output_image_path, "%s/image_%d.jpg", argv[2], b);
            stbi_write_jpg(output_image_path, width, height, 1, out + b * (width * height), 100);
        }

        free(out);
        cudaFree(d_in);
        cudaFree(d_out);

        return 0;
}
