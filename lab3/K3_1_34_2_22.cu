// %%writefile k2.cu

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
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define MASK_WIDTH 3
#define MASK_RADIUS 1

__global__ 
void convolution_3D_out_kernelwdsf2(unsigned char * in, unsigned char * out, int w, int h,int maskWidth, int comp, int batch_size, float *mask2) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int image_idx = blockIdx.z;
    int threads_per_block_x = blockDim.x;
    int threads_per_block_y = blockDim.y;
    // Calculate the size of shared memory needed for tiles
    //int shared_mem_size = (threads_per_block_y + maskWidth - 1) * (threads_per_block_x + maskWidth - 1) * 3 * sizeof(unsigned char);

    // Dynamically allocate shared memory for tiles
    extern __shared__ unsigned char shared_mem[];

    // Linearize the shared memory for each color channel
    unsigned char* tile_R = shared_mem;
    unsigned char* tile_G = tile_R + (threads_per_block_y + maskWidth - 1) * (threads_per_block_x + maskWidth - 1);
    unsigned char* tile_B = tile_G + (threads_per_block_y + maskWidth - 1) * (threads_per_block_x + maskWidth - 1);
    float *mask3 = (float*)(tile_B + (threads_per_block_y + maskWidth - 1) * (threads_per_block_x + maskWidth - 1));
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = blockIdx.y *threads_per_block_y;
    int col_start =  blockIdx.x *threads_per_block_y;
    int index_base = image_idx * (w * h * comp);

if (Col < w && Row < h && image_idx < batch_size) {
    
//    if (threadIdx.x < maskWidth && threadIdx.y < maskWidth) {
//         mask3[threadIdx.y * maskWidth + threadIdx.x] = mask2[threadIdx.y * maskWidth + threadIdx.x];
//     }
    for (int i = threadIdx.y; i < maskWidth; i += blockDim.y) {
        for (int j = threadIdx.x; j < maskWidth; j += blockDim.x) {
            if ( i < maskWidth && j < maskWidth) {
                mask3[i * maskWidth + j] = mask2[i * maskWidth + j];
            }
        }
    }
    __syncthreads();

    for (int i = ty; i < threads_per_block_x + maskWidth - 1; i += blockDim.y) {
        for (int j = tx; j < threads_per_block_y + maskWidth - 1; j += blockDim.x) {
            int x_index = col_start - (maskWidth / 2) + j;
            int y_index = row_start - (maskWidth / 2) + i;

            // Calculate the linear index for accessing shared memory
            int tile_idx = ( i) * (threads_per_block_x + maskWidth - 1) + ( j);

            // Check bounds and load data into tiles
            if (x_index >= 0 && x_index < w && y_index >= 0 && y_index < h) {
                tile_R[tile_idx] = in[(y_index * w + x_index) * 3 + index_base];
                tile_G[tile_idx] = in[(y_index * w + x_index) * 3 + 1 + index_base];
                tile_B[tile_idx] = in[(y_index * w + x_index) * 3 + 2 + index_base];
            } else {
                tile_R[tile_idx] = 0;
                tile_G[tile_idx] = 0;
                tile_B[tile_idx] = 0;
            }
        }
    }
    
    __syncthreads();
        float sum = 0;
        float mask_val ;
        int tile_idx;
        for(int j = 0; j < maskWidth; ++j) {
            for(int k = 0; k < maskWidth; ++k) {
                // int curRow = Row - (maskWidth / 2) + j;
                // int curCol = Col - (maskWidth / 2) + k;
                // if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    // Linearize the indexing for accessing tile elements
                    tile_idx = (ty + j) * (threads_per_block_x + maskWidth - 1) + (tx + k);
                    mask_val = mask3[j * maskWidth + k];
                    sum += tile_R[tile_idx] * mask_val+ tile_G[tile_idx] * mask_val+ tile_B[tile_idx] * mask_val;
                    // pixVal_G += tile_G[tile_idx] * mask3[j * maskWidth + k];
                    // pixVal_B += tile_B[tile_idx] * mask3[j * maskWidth + k];
                // }
            }
        }
        __syncthreads();
        int index = (Row * w + Col) + image_idx*w*h;
        // out[index] = ((unsigned char)(sum));
        out[index] = (unsigned char) max(0, min(255, (unsigned char) (sum)));
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
   float *host_mask = (float *)malloc(mask_dim * mask_dim * sizeof(float));
    for (int i = 0; i < mask_dim; ++i) {
        for (int j = 0; j < mask_dim; ++j) {
            fscanf(mask_file, "%f", &host_mask[i * mask_dim + j]);
        }
    }
    fclose(mask_file);

    // cudaMemcpyToSymbol(mask, host_mask, mask_dim * mask_dim * sizeof(float));

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

    ///
    float *d_mask;
    cudaMalloc(&d_mask, mask_dim * mask_dim * sizeof(float));
    cudaMemcpy(d_mask, host_mask, mask_dim * mask_dim * sizeof(float), cudaMemcpyHostToDevice);

    //
    // convolution_3D_out_kernelw<<<dim3((width + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X, (height + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X, batch_size), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_X)>>>(d_in, d_out, width, height, comp, batch_size);
    int shared_mem_size =  3 * (THREADS_PER_BLOCK_X + mask_dim -1) * (THREADS_PER_BLOCK_X + mask_dim -1) * sizeof(unsigned char) +  mask_dim * mask_dim * sizeof(float);
    convolution_3D_out_kernelwdsf2<<<dim3((width + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X, (height + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X, batch_size), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_X),shared_mem_size>>>(d_in, d_out, width, height,mask_dim, comp, batch_size,d_mask);
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
