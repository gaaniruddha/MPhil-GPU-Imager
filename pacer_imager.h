#ifndef _PACER_IMAGER_H__
#define _PACER_IMAGER_H__

#include <string>
using namespace std;

// FFTW, math etc :
// #include <fftw3.h

// for all cuda related stuff 
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cufftw.h"
#include "cufft.h"

class CPacerImager
{
// simple functions required for some internal calculations : date/time/filename etc :
protected : 
   // memory buffers for internal processing 
   // FFTW / cuFFT memory :
   cufftComplex* m_in_buffer_gpu; 
   cufftComplex* m_out_buffer_gpu; 

   int m_in_size;
   int m_out_size;

   // Additional GPU Input variables 
   // GPU Input variables 
   float *u_gpu; 
   float *v_gpu;
   float *vis_real_gpu; 
   float *vis_imag_gpu; 
  
   // GPU Output variables 
   float *uv_grid_real_gpu;
   float *uv_grid_imag_gpu;
   float *uv_grid_counter_gpu; 

   // ... Remaining code removed
 
protected :
   // Allocating GPU Memory 
   void AllocGPUMemory(int XYSIZE, int IMAGE_SIZE); 

   // Clean GPU Memory 
   void CleanGPUMemory(); 

public :
   void gridding_dirty_image( CBgFits& fits_vis_real, CBgFits& fits_vis_imag, CBgFits& fits_vis_u, CBgFits& fits_vis_v, CBgFits& fits_vis_w,
               CBgFits& uv_grid_real, CBgFits& uv_grid_imag, CBgFits& uv_grid_counter, double delta_u, double delta_v, 
               double frequency_mhz, 
               int n_pixels,
               double min_uv /*=-1000*/,
               const char* weighting /*="" weighting : U for uniform (others not implemented) */,
               /*CBgFits& uv_grid_real_param, CBgFits& uv_grid_imag_param, CBgFits& uv_grid_counter, ALREADY PASSED */ 
               bool bSaveIntermediate /*=false*/ , const char* szBaseOutFitsName /*=NULL*/, 
               bool bSaveImaginary /*=true*/ , bool bFFTUnShift /*=true*/); 
   
                   ing=true, 
                    const char* weighting="", // weighting : U for uniform (others not implemented)
                    const char* szBaseOutFitsName=NULL,
                    bool bCornerTurnRequired=true // changes indexing of data "corner-turn" from xGPU structure to continues (FITS image-like)
                  );

   // ... Remaining code removed
};

#endif 
