/*
gridding + imaging in cuda
Start date: 01/12/2022
End date: 01/12/2022

Notes: 
- This has the basic version of the entire GPU Imager, which includes both cuFFT + gridding() 
- The starting version of the code was provided to me by my supervisor. Hence I have removed the sections coded by him, and only kept the code segments written by me. 
*/ 

// Header files 
#include <stdio.h> 
#include <iostream>
using namespace std; 
#include <string.h> 
#include "math.h"
#include <cuda.h> 
#include "cuda_runtime.h"
#include <time.h>

#include "pacer_imager.h"
#include "pacer_common.h"
#include <bg_fits.h>

// FFTW (for the CPU version of imager) 
// #include <fftw3.h>

#include <math.h>

// local defines :
#include "pacer_imager_defs.h"

// msfitslib library :
#include <myfile.h>

#ifdef _PACER_PROFILER_ON_
#include <mydate.h>
#endif

// Gridding kernel implemented in CUDA has been defined here 
#include "gridding_imaging.h"

void __cuda_check_error(cudaError_t err, const char *file, int line){
	if(err != cudaSuccess){
        fprintf(stderr, "CUDA error (%s:%d): %s\n", file, line, cudaGetErrorString(err));
        exit(1);
    }
}

#define CUDA_CHECK_ERROR(X)({\
	__cuda_check_error((X), __FILE__, __LINE__);\
})

// Constructor definition
CPacerImager::CPacerImager()
: m_bInitialised(false), m_Baselines(0), m_pSkyImageReal(NULL), m_pSkyImageImag(NULL), m_bLocalAllocation(false), m_SkyImageCounter(0),
  m_in_buffer_gpu(NULL), m_out_buffer_gpu(NULL), m_in_size(-1), m_out_size(-1), m_bIncludeAutos(false), 
  u_gpu(NULL), v_gpu(NULL), vis_real_gpu(NULL), vis_imag_gpu(NULL), uv_grid_real_gpu(NULL), uv_grid_imag_gpu(NULL), uv_grid_counter_gpu(NULL)
{
   m_PixscaleAtZenith = 0.70312500; // deg for ch=204 (159.375 MHz) EDA2
   m_in_buffer_gpu = NULL;
   m_out_buffer_gpu = NULL;
}

// Destructor definition 
CPacerImager::~CPacerImager()
{
   CleanLocalAllocations();
   CleanGPUMemory(); 
}

// Allocating GPU Memory 
void CPacerImager::AllocGPUMemory(int XYSIZE, int IMAGE_SIZE)
{
   // Memory for GPU input variables: 
   if( !vis_real_gpu )
   {
      CUDA_CHECK_ERROR(cudaMalloc((float**)&vis_real_gpu, XYSIZE*sizeof(float)));
      CUDA_CHECK_ERROR(cudaMalloc((float**)&vis_imag_gpu, XYSIZE*sizeof(float)));
      CUDA_CHECK_ERROR(cudaMalloc((float**)&u_gpu, XYSIZE*sizeof(float)));
      CUDA_CHECK_ERROR(cudaMalloc((float**)&v_gpu, XYSIZE*sizeof(float)));
      // m_in_buffer 
      CUDA_CHECK_ERROR(cudaMalloc((void**) &m_in_buffer_gpu, sizeof(cufftComplex) * IMAGE_SIZE));
 
      // Memory for GPU output variables:  
      CUDA_CHECK_ERROR(cudaMalloc((float**)&uv_grid_real_gpu, IMAGE_SIZE*sizeof(float)));
      CUDA_CHECK_ERROR(cudaMalloc((float**)&uv_grid_imag_gpu, IMAGE_SIZE*sizeof(float)));
      CUDA_CHECK_ERROR(cudaMalloc((float**)&uv_grid_counter_gpu, IMAGE_SIZE*sizeof(float)));
      // m_out_buffer 
      CUDA_CHECK_ERROR(cudaMalloc((void**) &m_out_buffer_gpu, sizeof(cufftComplex) * IMAGE_SIZE)); 
   }
}

// Clean GPU Memory 
void CPacerImager::CleanGPUMemory()
{
   if( vis_real_gpu )
   {
      CUDA_CHECK_ERROR(cudaFree( vis_real_gpu)); 
   }
  
   if( vis_imag_gpu )
   {
      CUDA_CHECK_ERROR(cudaFree( vis_imag_gpu)); 
   }

   if( u_gpu )
   {
      CUDA_CHECK_ERROR(cudaFree( u_gpu)); 
   }

   if( v_gpu )
   {
      CUDA_CHECK_ERROR(cudaFree( v_gpu)); 
   }

   if( uv_grid_real_gpu )
   {
      CUDA_CHECK_ERROR(cudaFree( uv_grid_real_gpu)); 
   }

   if( uv_grid_imag_gpu )
   {
      CUDA_CHECK_ERROR(cudaFree( uv_grid_imag_gpu)); 
   }

   if( uv_grid_counter_gpu )
   {
      CUDA_CHECK_ERROR(cudaFree( uv_grid_counter_gpu)); 
   }
    
   if( m_in_buffer_gpu )
   {
      CUDA_CHECK_ERROR(cudaFree( m_in_buffer_gpu)); 
   }
   if( m_out_buffer_gpu )
   {
      CUDA_CHECK_ERROR(cudaFree( m_out_buffer_gpu)); 
   }
}

/*
gridding() + cuFFT combined in cuda 
Steps: 
- Done: Include all the same inputs from both gridding and dirty_imaging() 
- Done: Add dirty_image and gridding() initialisations 
- uv_grid_real/imag_param: same as uv_grid_real/imag (Passing the same parameters)
*/
void CPacerImager::gridding_dirty_image( CBgFits& fits_vis_real, CBgFits& fits_vis_imag, CBgFits& fits_vis_u, CBgFits& fits_vis_v, CBgFits& fits_vis_w,
               CBgFits& uv_grid_real, CBgFits& uv_grid_imag, CBgFits& uv_grid_counter, double delta_u, double delta_v, 
               double frequency_mhz, 
               int n_pixels,
               double min_uv /*=-1000*/,
               const char* weighting /*="" weighting : U for uniform (others not implemented) */,
               /*CBgFits& uv_grid_real_param, CBgFits& uv_grid_imag_param, CBgFits& uv_grid_counter, ALREADY PASSED */ 
               bool bSaveIntermediate /*=false*/ , const char* szBaseOutFitsName /*=NULL*/, 
               bool bSaveImaginary /*=true*/ , bool bFFTUnShift /*=true*/
            )
{
  PRINTF_DEBUG("DEBUG : GRIDDING : min_uv = %.4f\n",min_uv);

  // Start: Full gridding() 
  clock_t start_time = clock();
  printf("\n CLOCK GRIDDING + IMAGING, START: \n"); 

  // gridding() STUFF: 
  // u, v w statistics 
  double u_mean, u_rms, u_min, u_max, v_mean, v_rms, v_min, v_max, w_mean, w_rms, w_min, w_max;
  fits_vis_u.GetStat( u_mean, u_rms, u_min, u_max );
  fits_vis_v.GetStat( v_mean, v_rms, v_min, v_max );
  fits_vis_w.GetStat( w_mean, w_rms, w_min, w_max );

  // Input size: u, v and w 
  int u_xSize = fits_vis_u.GetXSize();
  int u_ySize = fits_vis_u.GetYSize();

  int vis_real_xSize =  fits_vis_real.GetXSize(); 
  int vis_real_ySize =  fits_vis_real.GetYSize(); 

  int vis_imag_xSize =  fits_vis_imag.GetXSize(); 
  int vis_imag_ySize =  fits_vis_real.GetYSize(); 

  // Output size: uv_grid_real, uv_grid_imag, uv_grid_counter 
  int uv_grid_counter_xSize = uv_grid_counter.GetXSize();
  int uv_grid_counter_ySize = uv_grid_counter.GetYSize();

  int xySize = (u_xSize*u_ySize);
  int image_size = (uv_grid_counter_xSize*uv_grid_counter_ySize); 
  int vis_real_size = (vis_real_xSize*vis_real_ySize);
  int vis_imag_size = (vis_real_xSize*vis_real_ySize);

  // uv_grid_counter_xSize = width
  // uv_grid_counter_ySize = height
  // size = image_size: (width x height)

  printf("\n SIZE CHECK xySize = %d", xySize);
  printf("\n SIZE CHECK image_size = %d", image_size);
  printf("\n SIZE CHECK vis_real_size = %d",vis_real_size); 
  printf("\n SIZE CHECK vis_imag_size = %d", vis_imag_size); 

  // In order to include conjugates at (-u,-v) UV point in gridding
  u_min = -u_max;
  v_min = -v_max;

  // Calculating the wavelength from frequency in MHz, and printing out the values 
  double frequency_hz = frequency_mhz*1e6;
  double wavelength_m = VEL_LIGHT / frequency_hz;
  PRINTF_DEBUG("DEBUG : GRIDDING: wavelength = %.4f [m] , frequency = %.4f [MHz]\n",wavelength_m,frequency_mhz);

  // Printing out the values of delta_u, delta_v and delta_w
  if(CPacerImager::m_ImagerDebugLevel>=IMAGER_DEBUG_LEVEL)
  {  
     double pixscale_zenith_deg = (1.00/(n_pixels*delta_u))*(180.00/M_PI); // in degrees 
     double pixscale_radians = 1.00/(2.00*u_max);
     double pixscale_deg_version2 = pixscale_radians*(180.00/M_PI);
     printf("DEBUG : pixscale old = %.8f [deg] vs. NEW = %.8f [deg]\n",pixscale_zenith_deg,pixscale_deg_version2);
     m_PixscaleAtZenith = pixscale_deg_version2; 
     printf("DEBUG : GRIDDING: U limits %.8f - %.8f , delta_u = %.8f -> pixscale at zenith = %.8f [deg]\n",u_min, u_max , delta_u , m_PixscaleAtZenith );
     printf("DEBUG : GRIDDING: V limits %.8f - %.8f , delta_v = %.8f\n", v_min, v_max , delta_v );
     printf("DEBUG : GRIDDING: W limits %.8f - %.8f\n", w_min, w_max );
  }

  // Calculating the x and y coordinates of the image centre 
  int center_x = int(n_pixels/2);
  int center_y = int(n_pixels/2);

  // Setting the initial values of is_odd_x, is_odd_y = 0 
  int is_odd_x = 0; 
  int is_odd_y = 0;
  if( (n_pixels % 2) == 1 )
  {
     is_odd_x = 1;
     is_odd_y = 1; 
  }

  // Setting the initial values of the uv_grid 
  uv_grid_real.SetValue( 0.00 );
  uv_grid_imag.SetValue( 0.00 );
  uv_grid_counter.SetValue( 0.00 );

  // out_image_real and out_image_imag 
  CBgFits out_image_real( uv_grid_real.GetXSize(), uv_grid_real.GetYSize() ), out_image_imag( uv_grid_real.GetXSize(), uv_grid_real.GetYSize() ); 

  // Setting the initial values of out_image_real/out_image_imag 
  out_image_real.SetValue( 0.00 );
  out_image_imag.SetValue( 0.00 );

  // Step 1: Declare GPU(Device) and CPU(Host) Variables 
  // CPU input variables 
  float *u_cpu = fits_vis_u.get_data();
  float *v_cpu = fits_vis_v.get_data();
  float *vis_real_cpu = fits_vis_real.get_data();
  float *vis_imag_cpu = fits_vis_imag.get_data();

  // CPU Output variables
  float *uv_grid_real_cpu = uv_grid_real.get_data();
  float *uv_grid_imag_cpu = uv_grid_imag.get_data();
  float *uv_grid_counter_cpu = uv_grid_counter.get_data();

  // GPU Variables: (Corresponding GPU Variables) declared in class 

  // Calculating cudaMalloc() 
  clock_t start_time2 = clock();

  // Step 2: Allocate memory for GPU variables
  // Memory for GPU input variables: 
  // CUDA_CHECK_ERROR(cudaMalloc((float**)&vis_real_gpu, xySize*sizeof(float)));
  // CUDA_CHECK_ERROR(cudaMalloc((float**)&vis_imag_gpu, xySize*sizeof(float)));
  // CUDA_CHECK_ERROR(cudaMalloc((float**)&u_gpu, xySize*sizeof(float)));
  // CUDA_CHECK_ERROR(cudaMalloc((float**)&v_gpu, xySize*sizeof(float)));

  // Allocate only if not-allocated 

  AllocGPUMemory(xySize, image_size);

  printf("\n GRIDDING CHECK: Step 2 Memory allocated for GPU variables");
  printf("\n GRIDDING CHECK: Step 2 GPU Variables initialised: AllocGPUMemory() called");

  // End of cudaMalloc() 
  clock_t end_time2 = clock();
  double duration_sec2 = double(end_time2-start_time2)/CLOCKS_PER_SEC;
  double duration_ms2 = duration_sec2*1000;
  printf("\n ** CLOCK cudaMalloc() took : %.6f [seconds], %.3f [ms]\n",duration_sec2,duration_ms2);

  // Step 3: Copy contents from CPU to GPU [input variables]
  // cudaMemcpy(destination, source, size, HostToDevice)

  // Start of cudaMemcpy()
  clock_t start_time3 = clock();

  CUDA_CHECK_ERROR(cudaMemcpy((float*)u_gpu, (float*)u_cpu, sizeof(float)*xySize, cudaMemcpyHostToDevice)); 
  CUDA_CHECK_ERROR(cudaMemcpy((float*)v_gpu, (float*)v_cpu, sizeof(float)*xySize, cudaMemcpyHostToDevice)); 
  CUDA_CHECK_ERROR(cudaMemcpy((float*)vis_real_gpu, (float*)vis_real_cpu, sizeof(float)*xySize, cudaMemcpyHostToDevice)); 
  CUDA_CHECK_ERROR(cudaMemcpy((float*)vis_imag_gpu, (float*)vis_imag_cpu, sizeof(float)*xySize, cudaMemcpyHostToDevice)); 

  clock_t end_time3 = clock();
  double duration_sec3 = double(end_time3-start_time3)/CLOCKS_PER_SEC;
  double duration_ms3 = duration_sec3*1000;
  printf("\n ** CLOCK cudaMemcpy() CPU to GPU took : %.6f [seconds], %.3f [ms]\n",duration_sec3,duration_ms3);

  printf("\n GRIDDING CHECK: Step 3 CPU to GPU copied"); 

  int nBlocks = (xySize + NTHREADS -1)/NTHREADS; 
  printf("\n GRIDDING CHECK: NTHREADS = %d", NTHREADS);
  printf("\n GRIDDING CHECK: nBlocks = %d", nBlocks);

  // Step 4: Call to GPU kernel
  
  // Start of kernel call 
  clock_t start_time4 = clock();

  gridding_imaging_cuda<<<nBlocks,NTHREADS>>>(u_gpu, v_gpu, wavelength_m, image_size, delta_u, delta_v, n_pixels, center_x, center_y, is_odd_x, is_odd_y, vis_real_gpu, vis_imag_gpu, uv_grid_counter_gpu, uv_grid_real_gpu, uv_grid_imag_gpu, min_uv, m_in_buffer_gpu); 
  printf("\n GRIDDING CHECK: Step 4 Calls to kernel");

  // End of kernel call 
  clock_t end_time4 = clock();
  double duration_sec4 = double(end_time4-start_time4)/CLOCKS_PER_SEC;
  double duration_ms4 = duration_sec4*1000;
  printf("\n ** CLOCK kernel call took : %.6f [seconds], %.3f [ms]\n",duration_sec4,duration_ms4);

  // Gives the error in the kernel! 
  CUDA_CHECK_ERROR(cudaGetLastError());
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  // uv_grid_counter_xSize = width
  // uv_grid_counter_ySize = height
  // size = image_size: (width x height)

  // Checking Execution time for cuFFT 
  clock_t start_time6 = clock();

  //  Implement cuFFT as well 
  cufftHandle pFwd=0;
  // cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type); 
  cufftPlan2d(&pFwd, uv_grid_counter_xSize, uv_grid_counter_ySize, CUFFT_C2C);
  // cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction);
  cufftExecC2C(pFwd, m_in_buffer_gpu, m_out_buffer_gpu, CUFFT_FORWARD);

  // End of cuFFT 
  clock_t end_time6 = clock();
  double duration_sec6 = double(end_time6-start_time6)/CLOCKS_PER_SEC;
  double duration_ms6 = duration_sec6*1000;
  printf("\n ** CLOCK cuFFT() took : %.6f [seconds], %.3f [ms]\n",duration_sec6,duration_ms6);
  printf("\n Imaging CHECK: cuFFT completed: \n "); 

  // Step 5: Copy contents from GPU variables to CPU variables
  // cudaMemcpy(destination, source, size, HostToDevice)

  // Start of cudaMemcpy() 
  clock_t start_time5 = clock();

  CUDA_CHECK_ERROR(cudaMemcpy((float*)uv_grid_counter_cpu, (float*)uv_grid_counter_gpu, sizeof(float)*image_size, cudaMemcpyDeviceToHost)); 
  CUDA_CHECK_ERROR(cudaMemcpy((float*)uv_grid_real_cpu, (float*)uv_grid_real_gpu, sizeof(float)*image_size, cudaMemcpyDeviceToHost)); 
  CUDA_CHECK_ERROR(cudaMemcpy((float*)uv_grid_imag_cpu, (float*)uv_grid_imag_gpu, sizeof(float)*image_size, cudaMemcpyDeviceToHost)); 

  // CPU Variable 
  cufftComplex* m_out_data;
  m_out_data = (cufftComplex*)malloc(sizeof(cufftComplex) * image_size);
  CUDA_CHECK_ERROR(cudaMemcpy(m_out_data, m_out_buffer_gpu, sizeof(cufftComplex)*image_size, cudaMemcpyDeviceToHost));

  // End of cudaMemcpy() GPU to CPU 
  clock_t end_time5 = clock();
  double duration_sec5 = double(end_time5-start_time5)/CLOCKS_PER_SEC;
  double duration_ms5 = duration_sec5*1000;
  printf("\n ** CLOCK cudaMemcpy() GPU to CPU took : %.6f [seconds], %.3f [ms]\n",duration_sec5,duration_ms5);
  printf("\n GRIDDING CHECK: Step 5 GPU to CPU copied"); 

  // Step 6: Free GPU memory 
  // cudaFree(vis_real_gpu);
  // cudaFree(vis_imag_gpu);
  // cudaFree(u_gpu);
  // cudaFree(v_gpu);
  // cudaFree(uv_grid_real_gpu);
  // cudaFree(uv_grid_imag_gpu);
  // cudaFree(uv_grid_counter_gpu);
  // printf("\n GRIDDING CHECK: Step 6 Memory Free! \n ");

  // Uniform weighting (Not implemented here)
  if( strcmp(weighting, "U" ) == 0 )
  {
     uv_grid_real.Divide( uv_grid_counter );
     uv_grid_imag.Divide( uv_grid_counter );
  } 

  // float pointers to 1D Arrays 
  float* out_data_real = out_image_real.get_data();
  float* out_data_imag = out_image_imag.get_data();
  double fnorm = 1.00/uv_grid_counter.Sum();    

  // Assigning back 
  for(int i = 0; i < image_size; i++) 
   {
     out_data_real[i] = m_out_data[i].x*fnorm; 
     out_data_imag[i] = m_out_data[i].y*fnorm; 
   }   

  // Saving gridding() output files 
  // ... Remaining code removed
  
  // Saving dirty_image() output files
  // ... Remaining code removed

// ... Remaining code removed

// gridding + imaging ends
clock_t end_time = clock();
double duration_sec = double(end_time-start_time)/CLOCKS_PER_SEC;
double duration_ms = duration_sec*1000;
printf("CLOCK TOTAL GRIDDING + IMAGING TOOK: %.6f [seconds], %.3f [ms]\n",duration_sec,duration_ms);
}
