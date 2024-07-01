// This is not used at the moment.
__global__ void sumChannelsKernel(const unsigned char* d_img, int width, int height, int channels, int step, double* d_sumR, double* d_sumG, double* d_sumB) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * step + x * channels;
        atomicAdd(d_sumR, d_img[idx + 2]); // R channel
        atomicAdd(d_sumG, d_img[idx + 1]); // G channel
        atomicAdd(d_sumB, d_img[idx + 0]); // B channel
    }
}