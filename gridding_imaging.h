/* 
Start date: 01/12/2022 
Expected End date: 15/11/2022 
Actual end date: 01/12/2022

Header file for both gridding + cuFFT, implemented in cuda 
- gridding_imaging_cuda

References: 
https://stackoverflow.com/questions/17489017/can-we-declare-a-variable-of-type-cufftcomplex-in-side-a-kernel

*/

// In order to include the gridding_gpu.cu 
#include <cuda.h> 
#include "cuda_runtime.h"
// So that it recognises: blockIdx
#include "device_launch_parameters.h"


// Cuda kernal: gridding and cuFFT 
__global__ void gridding_imaging_cuda(float *u_cuda, float *v_cuda, 
                                      double wavelength_cuda, int image_size_cuda, double delta_u_cuda, double delta_v_cuda, 
                                      int n_pixels_cuda, int center_x_cuda, int center_y_cuda, int is_odd_x_cuda, int is_odd_y_cuda,
                                      float *vis_real_cuda, float *vis_imag_cuda, 
                                      float *uv_grid_counter_cuda, float *uv_grid_real_cuda, float *uv_grid_imag_cuda, double min_uv_cuda, 
                                      cufftComplex *m_in_buffer_cuda)
{   
    // Calculating the required id 
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Getting corresponding real and imag visibilities 
    double re = vis_real_cuda[i]; 
    double im = vis_imag_cuda[i]; 

    // Checking for NaN values 
    if( !isnan(re) && !isnan(im) )
    {
        // Operation 1: uv_lambda()
        double u_lambda = (u_cuda[i])/(wavelength_cuda); 
        double v_lambda = (v_cuda[i])/(wavelength_cuda); 

        // Calculating distance between the two antennas 
        double uv_distance = sqrt(u_lambda*u_lambda + v_lambda*v_lambda);

        if( uv_distance > min_uv_cuda )
        {
            // (For all the rows of the Correlation Matrix)
            // Operation 2: uv_index()
            double u_pix, v_pix;
            u_pix = round(u_lambda/delta_u_cuda); 
            v_pix = round(v_lambda/delta_v_cuda);
            int u_index = u_pix + (n_pixels_cuda/2); 
            int v_index = v_pix + (n_pixels_cuda/2);

            // Operation 3: x_grid, y_grid (With FFT-UNSHIFT)
            int x_grid = 0;
            int y_grid = 0; 

            if( u_index < center_x_cuda )
                x_grid = u_index + center_x_cuda + is_odd_x_cuda;
            else
                x_grid = u_index - center_x_cuda;       
            if( v_index < center_y_cuda )
                y_grid = v_index + center_y_cuda + is_odd_y_cuda;
            else
                y_grid = v_index - center_y_cuda; 

            // Operation 4: Assignment of (re,im)vis to uv_grid
            // Position for assignment 
            int pos = (n_pixels_cuda*y_grid) + x_grid; 

            if(pos>0 && pos<image_size_cuda)
            {
                // Allocating in uv_grid
                atomicAdd(&uv_grid_real_cuda[pos],re);
                atomicAdd(&uv_grid_imag_cuda[pos],im);
                atomicAdd(&uv_grid_counter_cuda[pos],1);

                // Allocating inside m_in_buffer as well 
                atomicAdd(&m_in_buffer_cuda[pos].x,re);
                atomicAdd(&m_in_buffer_cuda[pos].y,im);

            }   

            // (For all the corresponding columns of the Correlation Matrix)
            // Operation 2: uv_index()
            int u_index2 = -u_pix + (n_pixels_cuda/2); 
            int v_index2 = -v_pix + (n_pixels_cuda/2);

            // Operation 3: x_grid, y_grid (With FFT-UNSHIFT)
            int x_grid2 = 0; 
            int y_grid2 = 0; 

            if( u_index2 < center_x_cuda )
                x_grid2 = u_index2 + center_x_cuda + is_odd_x_cuda;
            else
                x_grid2 = u_index2 - center_x_cuda;       
            if( v_index2 < center_y_cuda )
                y_grid2 = v_index2+ center_y_cuda + is_odd_y_cuda;
            else
                y_grid2 = v_index2 - center_y_cuda; 

            // Operation 4: Assignment of (re,im)vis to uv_grid
            // Position for assignment 
            int pos2 = (n_pixels_cuda*y_grid2) + x_grid2; 

            if(pos2>0 && pos2<image_size_cuda)
            {
                // Allocating in uv_grid
                atomicAdd(&uv_grid_real_cuda[pos2],re);
                atomicAdd(&uv_grid_imag_cuda[pos2],-im);
                atomicAdd(&uv_grid_counter_cuda[pos2],1);

                // Allocating inside m_in_buffer as well 
                atomicAdd(&m_in_buffer_cuda[pos2].x,re);
                atomicAdd(&m_in_buffer_cuda[pos2].y,-im);
            }   

        }
        
    }
    
}
