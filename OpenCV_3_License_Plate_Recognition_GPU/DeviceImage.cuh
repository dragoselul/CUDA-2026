struct DeviceImage {
    unsigned char* data;
    int width;
    int height;
    int channels;
    size_t pitch; // Use pitch for optimized GPU memory alignment

    // This is a "shallow" struct used to pass data to kernels
};