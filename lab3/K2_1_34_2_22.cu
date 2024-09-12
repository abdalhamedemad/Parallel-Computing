// %%writefile k3.cu
//input tile
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
// #define MASK_DIM 3
#define THREADS_PER_BLOCK_X 18
#define THREADS_PER_BLOCK_Y 18
#define MASK_WIDTH 3
#define MASK_RADIUS 1

#define  O_TILE_WIDTH 16
#define  BLOCK_WIDTH 16


__global__ 
void convolution_3D_input_kernelwds(unsigned char * in, unsigned char * out, int w, int h, int comp, int batch_size,int maskWidth, float * mask2,int output_tile_dim) {
    // int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int image_idx = blockIdx.z;
    int threads_per_block_x = blockDim.x;
    int threads_per_block_y = blockDim.y;
    /////////////////
    // int shared_mem_size = (threads_per_block_y ) * (threads_per_block_x) * 3 * sizeof(unsigned char);

    // Dynamically allocate shared memory for tiles
    extern __shared__ unsigned char shared_mem[];
    // each thread loads one element from the input list into the shared memory
    unsigned char* tile_R = shared_mem;
    unsigned char* tile_G = tile_R + (threads_per_block_y ) * (threads_per_block_x );
    unsigned char* tile_B = tile_G + (threads_per_block_y ) * (threads_per_block_x );

    float *mask3 = (float*)(tile_B + (threads_per_block_y ) * (threads_per_block_x ));
    // if (threadIdx.x < maskWidth && threadIdx.y < maskWidth) {
    //     mask3[threadIdx.y * maskWidth + threadIdx.x] = mask2[threadIdx.y * maskWidth + threadIdx.x];
    // }
    for (int i = threadIdx.y; i < maskWidth; i += blockDim.y) {
        for (int j = threadIdx.x; j < maskWidth; j += blockDim.x) {
            if ( i < maskWidth && j < maskWidth) {
                mask3[i * maskWidth + j] = mask2[i * maskWidth + j];
            }
        }
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // output tile indeces
    int row_o = blockIdx.y * output_tile_dim + ty;
    int col_o = blockIdx.x * output_tile_dim + tx;
    ///////////////////////////
    // input tile indeces
    int row_i = row_o - maskWidth/2;
    int col_i = col_o - maskWidth/2;
    ////////////////////
    int tile_idx = (ty) * (threads_per_block_x) + (tx);
    if (image_idx < batch_size){

    if (row_i >= 0 && row_i < h && col_i >= 0 && col_i < w) {
        tile_R[tile_idx] = in[(image_idx * w * h * comp) + (row_i * w + col_i) * comp];
        tile_G[tile_idx] = in[(image_idx * w * h * comp) + (row_i * w + col_i) * comp + 1];
        tile_B[tile_idx] = in[(image_idx * w * h * comp) + (row_i * w + col_i) * comp + 2];
    } else {
        tile_R[tile_idx] = 0;
        tile_G[tile_idx] = 0;
        tile_B[tile_idx] = 0;
    }
    __syncthreads();
    // printf("gg");
    if (row_o < h && col_o < w && ty <  output_tile_dim && tx < output_tile_dim){
        float sum = 0;
        float maskValue;
        for (int i = 0; i < maskWidth; i++)
        {
            for (int j = 0; j < maskWidth; j++)
            {
               // int tile_idx = (ty + i) * (threads_per_block_x) + (tx + j);
                maskValue= mask3[i * maskWidth + j];
                sum += tile_R[(ty + i) * (threads_per_block_x) + (tx + j)] * maskValue + tile_G[(ty + i) * (threads_per_block_x) + (tx + j)] *maskValue + tile_B[(ty + i) * (threads_per_block_x) + (tx + j)] * maskValue;
            }
        }

        //int index = ((image_idx * w * h) + (row_o * w + col_o));
        // out[((image_idx * w * h) + (row_o * w + col_o))] = (sum);
        out[((image_idx * w * h) + (row_o * w + col_o))] = (unsigned char) max(0, min(255, (unsigned char) (sum)));
    }
}
}

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
    float *host_mask = (float *)malloc(mask_dim * mask_dim * sizeof(float));

    for (int i = 0; i < mask_dim; ++i) {
        for (int j = 0; j < mask_dim; ++j) {
            fscanf(mask_file, "%f", &host_mask[i * mask_dim + j]);
        }
    }
    fclose(mask_file);


    printf("Reading image...\n");

    std::vector<std::string> files = listFiles(argv[1]);
    int batch_size = atoi(argv[3]);
    int width, height, comp;
    unsigned char *d_in, *d_out;


    std::vector<unsigned char> imageData;
    int totalSize = 0;
    // int padded_width, padded_height;
    for (int b = 0; b < batch_size; ++b) {
        char input_image_path[100];
        sprintf(input_image_path, "%s/%s", argv[1], files[b].c_str());
        // int width1, height1, comp1;
        unsigned char *data = stbi_load(input_image_path, &width, &height, &comp, 0);
        int size = width * height * comp;
        totalSize += size;
        // print width, height, comp
        std::cout << "width = " << width << ", height = " << height << ", comp = " << comp << std::endl;
        imageData.insert(imageData.end(), data, data + size);
        stbi_image_free(data);
    }

    cudaMalloc(&d_in, batch_size * width * height * comp);
    cudaMalloc(&d_out, batch_size * width * height);
    // print the vector size
    std::cout << "Size of the vector: " << imageData.size() << std::endl;

    // Copy the concatenated image data to the GPU
    cudaMemcpy(d_in, imageData.data(), totalSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // convolution_3D_input_kernelw<<<dim3((width + 16 - 1) / 16, (height + 16 - 1) / 16, batch_size), dim3(16, 16)>>>(d_in, d_out, width, height, comp, batch_size);
        // int BLOCK_WIDTH = O_TILE_WIDTH + mask_dim - 1;
        int output_tile_dim = BLOCK_WIDTH - mask_dim - 1;
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
        dim3 dimGrid((width - 1)/output_tile_dim +1, (height - 1)/output_tile_dim +1, batch_size);

      float *d_mask;
      cudaMalloc(&d_mask, mask_dim * mask_dim * sizeof(float));
      cudaMemcpy(d_mask, host_mask, mask_dim * mask_dim * sizeof(float), cudaMemcpyHostToDevice);

    //
        int shared_mem_size = 3 * (BLOCK_WIDTH) * (BLOCK_WIDTH) * sizeof(unsigned char) + mask_dim * mask_dim * sizeof(float);
        // convolution_3D_input_kernelw<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, comp, batch_size);
        convolution_3D_input_kernelwds<<<dimGrid, dimBlock,shared_mem_size>>>(d_in, d_out, width, height, comp, batch_size,mask_dim,d_mask,output_tile_dim);


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

