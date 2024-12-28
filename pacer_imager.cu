/*
gridding + imaging in cuda
Start date: 01/12/2022
End date: 01/12/2022

Notes: 
- This has the basic version of the entire GPU Imager, which includes both cuFFT + gridding() 
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

// FFTW, math etc :
// #include <fftw3.h>
#include <math.h>

// local defines :
#include "pacer_imager_defs.h"

// msfitslib library :
#include <myfile.h>

#ifdef _PACER_PROFILER_ON_
#include <mydate.h>
#endif

// All the cuda kernels have been written here: 
// #include "gridding.h"
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

// TEST OPTIONS to compare with MIRIAD image
// see memos : PAWSEY/PaCER/logbook/20220305_pacer_imager_validation.odt, MIRIAD natural weighting (sup=0) etc:
// invert vis=chan_204_20211116T203000_yx.uv map=chan_204_20211116T203000_iyx.map imsize=180,180 beam=chan_204_20211116T203000_iyx.beam  sup=0 options=imaginary stokes=yx select='uvrange(0.0,100000)'
bool CPacerImager::m_bCompareToMiriad = false;

// debug level : see pacer_imager_defs.h for SAVE_FILES_NONE etc
int CPacerImager::m_ImagerDebugLevel = IMAGER_ALL_MSG_LEVEL;

// level of saving intermediate and test files , see pacer_imager_defs.h for defines SAVE_FILES_NONE
int CPacerImager::m_SaveFilesLevel = SAVE_FILES_ALL;

// can also save control files every N-th file 
int CPacerImager::m_SaveControlImageEveryNth=-1;

// show final image statistics :
bool CPacerImager::m_bPrintImageStatistics = false; // default disabled to make imaging as fast as possible

void CPacerImager::SetDebugLevel( int debug_level )
{
   CPacerImager::m_ImagerDebugLevel = debug_level;
   gBGPrintfLevel = debug_level;
}

void CPacerImager::SetFileLevel( int filesave_level )
{
   CPacerImager::m_SaveFilesLevel = filesave_level;
}

void CPacerImager::SetFlaggedAntennas( vector<int>& flagged_antennas )
{
   m_FlaggedAntennas = flagged_antennas;
   if( m_FlaggedAntennas.size() > 0 ){
      if( m_MetaData.m_AntennaPositions.size() > 0 ){
         // flagging antennas in the list :
         for(int i=0;i<m_FlaggedAntennas.size();i++){
            int ant_index = m_FlaggedAntennas[i];
            
            if( ant_index >= 0 && ant_index < m_MetaData.m_AntennaPositions.size() ){
               m_MetaData.m_AntennaPositions[ant_index].flag = 1;
            }
         }
      }
   }
}

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

void CPacerImager::CleanLocalAllocations()
{
   if( m_bLocalAllocation )
   { // only remove when locally allocated, as it can also be passed from outside using SetOutputImagesExternal()
      if( m_pSkyImageReal )
      {
         delete m_pSkyImageReal;
      }
      if( m_pSkyImageImag )
      {
         delete m_pSkyImageImag;
      }
      
      m_bLocalAllocation = false;
   }
}

bool CPacerImager::AllocOutPutImages( int sizeX, int sizeY )
{
   bool bRet = false;
   if( !m_pSkyImageReal )
   {
      m_pSkyImageReal = new CBgFits( sizeX, sizeY );   
      m_bLocalAllocation = true;
      bRet = true;
   }
   else
   {
      CheckSize( *m_pSkyImageReal, sizeX, sizeY );
   }

   if( !m_pSkyImageImag )
   {
      m_pSkyImageImag = new CBgFits( sizeX, sizeY );
      m_bLocalAllocation = true;
      bRet = true;
   }
   else
   {
      CheckSize( *m_pSkyImageImag, sizeX, sizeY );
   }
   
   return bRet;
}

void CPacerImager::SetOutputImagesExternal( CBgFits* pSkyImageRealExt, CBgFits* pSkyImageImagExt )
{
   CleanLocalAllocations();
      
   m_pSkyImageReal = pSkyImageRealExt;
   m_pSkyImageImag = pSkyImageImagExt;
   m_bLocalAllocation = false;
}

void CPacerImager::Initialise()
{
  if( !m_bInitialised )
  {
     m_bInitialised = true;
     
    // test file with antenna positions can be used to overwrite whatever was in .metafits
    if( strlen( m_ImagerParameters.m_AntennaPositionsFile.c_str() ) && MyFile::DoesFileExist(  m_ImagerParameters.m_AntennaPositionsFile.c_str() ) )
    {
       bool bConvertToXYZ = false;
       if( !m_ImagerParameters.m_bConstantUVW )
       { // if non-constant UVW -> non zenith phase centered all-sky image
          bConvertToXYZ = true;
       }
       int n_ants = m_MetaData.m_AntennaPositions.ReadAntennaPositions( m_ImagerParameters.m_AntennaPositionsFile.c_str(), bConvertToXYZ  );
       PRINTF_INFO("INFO : read %d antenna positions from file %s\n",n_ants,m_ImagerParameters.m_AntennaPositionsFile.c_str());
       
       if( /*true ||*/ strlen( m_ImagerParameters.m_MetaDataFile.c_str() ) == 0 )
       { // only calculate UVW here when Metadata is not required
          // initial recalculation of UVW at zenith (no metadata provided -> zenith):       
          m_Baselines = m_MetaData.m_AntennaPositions.CalculateUVW( m_U, m_V, m_W, (CPacerImager::m_SaveFilesLevel>=SAVE_FILES_DEBUG), m_ImagerParameters.m_szOutputDirectory.c_str(), m_bIncludeAutos );
          PRINTF_INFO("INFO : calculated UVW coordinates of %d baselines (include Autos = %d)\n",m_Baselines,m_bIncludeAutos);
       }
       else
       {
          printf("INFO : non-zenith pointing meta data is required to calculate UVW\n");
       }
    }
    else
    {
       PRINTF_WARNING("WARNING : antenna position file %s not specified or does not exist\n",m_ImagerParameters.m_AntennaPositionsFile.c_str());
    }
    
    // read all information from metadata 
    if( strlen( m_ImagerParameters.m_MetaDataFile.c_str() ) && MyFile::DoesFileExist( m_ImagerParameters.m_MetaDataFile.c_str() ) ){
      PRINTF_INFO("INFO : reading meta data from file %s\n",m_ImagerParameters.m_MetaDataFile.c_str());
      if( !m_MetaData.ReadMetaData( m_ImagerParameters.m_MetaDataFile.c_str() ) ){
         PRINTF_ERROR("ERROR : could not read meta data from file %s\n",m_ImagerParameters.m_MetaDataFile.c_str() );
      }
    }       
  }
}

bool CPacerImager::CheckSize( CBgFits& image, int sizeX, int sizeY )
{
   if( image.GetXSize() != sizeX || image.GetYSize() != sizeY ){
      image.Realloc( sizeX, sizeY );      
      PRINTF_INFO("DEBUG : change of image size to (%d,%d) was required\n",sizeX,sizeY);
      
      return true;
   }
   
   // if image size was ok and nothing was required
   return false;
}

// See https://www.gaussianwaves.com/2015/11/interpreting-fft-results-complex-dft-frequency-bins-and-fftshift/ 
// for explanations why it is needed
void CPacerImager::fft_shift( CBgFits& dirty_image, CBgFits& out_image )
{
   int xSize = dirty_image.GetXSize();
   int ySize = dirty_image.GetYSize();

   // TODO : create member object m_tmp_image to avoid allocation every time this function is called 
   CBgFits tmp_image( xSize, ySize );
   
   int center_freq_x = int( xSize/2 );
   int center_freq_y = int( ySize/2 );
   
   int is_odd = 0;
   if ( (xSize%2) == 1 && (ySize%2) == 1 ){
      is_odd = 1;
   }

   // TODO : optimise similar to gridder.c in RTS or check imagefromuv.c , LM_CopyFromFFT which is totally different and may have to do with image orientation, but also is faster !!!
   // X (horizontal FFT shift) :
   for(int y=0;y<ySize;y++){ 
      float* tmp_data = tmp_image.get_line(y);
      float* image_data = dirty_image.get_line(y);
      
      // TODO / WARNING : lools like here for x=center_freq_x and images size = 2N -> center_freq_x = N -> center_freq_x+x can by N+N=2N which is outside image !!!
      for(int x=0;x<=center_freq_x;x++){ // check <= -> <
         tmp_data[center_freq_x+x] = image_data[x];
      }
      for(int x=(center_freq_x+is_odd);x<xSize;x++){
         tmp_data[x-(center_freq_x+is_odd)] = image_data[x];
      }      
   }

   for(int x=0;x<xSize;x++){ 
      for(int y=0;y<=center_freq_y;y++){ // check <= -> <
         out_image.setXY(x,center_freq_y+y,tmp_image.getXY(x,y));
      }
      for(int y=(center_freq_y+is_odd);y<ySize;y++){
         out_image.setXY( x , y-(center_freq_y+is_odd),tmp_image.getXY(x,y));
      }      
   }
}

// UV data are with DC in the center -> have to be FFTshfted to form input to FFT function :
// See https://www.gaussianwaves.com/2015/11/interpreting-fft-results-complex-dft-frequency-bins-and-fftshift/ 
// for explanations why it is needed
void CPacerImager::fft_unshift( CBgFits& dirty_image, CBgFits& out_image )
{
   int xSize = dirty_image.GetXSize();
   int ySize = dirty_image.GetYSize();
   
   // TODO : create member object m_tmp_image to avoid allocation every time this function is called
   CBgFits tmp_image( xSize, ySize );
   
   int center_freq_x = int( xSize/2 );
   int center_freq_y = int( ySize/2 );
   
   int is_odd = 0;
   if ( (xSize%2) == 1 && (ySize%2) == 1 ){
      is_odd = 1;
   }
   
   // TODO : optimise similar to gridder.c in RTS or check imagefromuv.c , LM_CopyFromFFT which is totally different and may have to do with image orientation, but also is faster !!!
   // X (horizontal FFT shift) :
   for(int y=0;y<ySize;y++){ 
      float* tmp_data = tmp_image.get_line(y);
      float* image_data = dirty_image.get_line(y);
      
      for(int x=0;x<center_freq_x;x++){ // check <= -> <
         tmp_data[center_freq_x+x+is_odd] = image_data[x];
      }
      for(int x=center_freq_x;x<xSize;x++){
         tmp_data[x-center_freq_x] = image_data[x];
      }      
   }

   for(int x=0;x<xSize;x++){ 
      for(int y=0;y<center_freq_y;y++){ // check <= -> <
         out_image.setXY( x, center_freq_y+y+is_odd, tmp_image.getXY(x,y));
      }
      for(int y=center_freq_y;y<ySize;y++){
         out_image.setXY(x,y-center_freq_y,tmp_image.getXY(x,y));
      }      
   }
}

bool CPacerImager::SaveSkyImage( const char* outFitsName, CBgFits* pFits )
{
   if( !pFits ){
      PRINTF_ERROR("ERROR in code SaveSkyImage, pFits pointer not set\n");
      return false;
   }

   PRINTF_INFO("INFO : saving image %s\n",outFitsName);   
   pFits->SetFileName( outFitsName );
   
   // fill FITS header :
   // TODO :
   pFits->SetKeyword("TELESCOP","EDA2");
   
   // scripts/add_fits_header.py
   // TODO - use properly calculated values :
   double pixscale = m_PixscaleAtZenith; // was hardcoded 0.70312500;
   // azh2radec 1581220006 mwa 0 90
   // (RA,DEC) = ( 312.07545047 , -26.70331900 )
   // 20.80503003133333333333
   double ra_deg = 312.07545047; // = 20.80503003133333333333 hours ; // was 2.13673600000E+02;
   double dec_deg = -2.67033000000E+01;
   
   int crpix1 = int(pFits->GetXSize()/2) + 1;
   pFits->SetKeyword("CRPIX1", crpix1 );   
   pFits->SetKeyword("CDELT1", pixscale );
   pFits->SetKeyword("CRVAL1", ra_deg ); // RA of the centre 
   pFits->SetKeyword("CTYPE1","RA---SIN");

   int crpix2 = int(pFits->GetYSize()/2) + 1;
   pFits->SetKeyword("CRPIX2", crpix2 );   
   pFits->SetKeyword("CDELT2", pixscale );
   pFits->SetKeyword("CRVAL2", dec_deg ); // RA of the centre 
   pFits->SetKeyword("CTYPE2","DEC---SIN");


   
   pFits->WriteFits( outFitsName );
   
   return true;
}

bool CPacerImager::CalculateUVW()
{
   Initialise();
 
   bool bRecalculationRequired = false;
   
   if( m_Baselines <=0 || m_U.GetXSize() <= 0 || m_V.GetXSize() <= 0 || m_W.GetXSize() <= 0 || !m_ImagerParameters.m_bConstantUVW ){
      bRecalculationRequired = true;      
   }
   
   if( bRecalculationRequired ){
      PRINTF_DEBUG("DEBUG : recalculation of UVW is required\n");
      
      m_Baselines = m_MetaData.m_AntennaPositions.CalculateUVW( m_U, m_V, m_W, true, m_ImagerParameters.m_szOutputDirectory.c_str(), m_bIncludeAutos );
      PRINTF_INFO("INFO : calculated UVW coordinates of %d baselines\n",m_Baselines); 
   }
   
   return (m_Baselines>0);
}

bool CPacerImager::ReadOrCalcUVW( const char* basename, const char* szPostfix )
{
  string fits_file_u = basename;
  fits_file_u += "_u";
  if( strlen( szPostfix ) ){
     fits_file_u += szPostfix;
  }
  fits_file_u += ".fits";

  string fits_file_v = basename;
  fits_file_v += "_v";
  if( strlen( szPostfix ) ){
     fits_file_v += szPostfix;
  }
  fits_file_v += ".fits";

  string fits_file_w = basename;
  fits_file_w += "_w";
  if( strlen( szPostfix ) ){
     fits_file_w += szPostfix;
  }
  fits_file_w += ".fits";

  if(CPacerImager::m_ImagerDebugLevel>=IMAGER_DEBUG_LEVEL){  
     printf("DEBUG : Expecting the following files to exist:\n");
     printf("\t%s\n",fits_file_u.c_str()); 
     printf("\t%s\n",fits_file_v.c_str()); 
     printf("\t%s\n",fits_file_w.c_str()); 
  }


  int n_ant = m_MetaData.m_AntennaPositions.size();
  bool bCalculateMetaFits = false;
  if( n_ant>0 && (m_MetaData.HasMetaFits() || m_ImagerParameters.m_bConstantUVW ) ){
//  if( n_ant > 0 ){
     printf("DEBUG : can calculate UVW n_ant = %d , has_metadata = %d , constant UVW = %d\n",n_ant,m_MetaData.HasMetaFits(),m_ImagerParameters.m_bConstantUVW);
     bCalculateMetaFits = true;
  }else{
     printf("DEBUG : cannot calculate UVW n_ant = %d , has_metadata = %d , constant UVW = %d\n",n_ant,m_MetaData.HasMetaFits(),m_ImagerParameters.m_bConstantUVW);
  }

  if( bCalculateMetaFits ){
     if( !CalculateUVW() ){
        printf("ERROR : could not calculate UVW coordinates\n");
        return false;
     }
  }else{
     PRINTF_WARNING("WARNING : antenna position file %s not specified or does not exist -> will try using UVW FITS files : %s,%s,%s\n",m_ImagerParameters.m_AntennaPositionsFile.c_str(),fits_file_u.c_str(),fits_file_v.c_str(),fits_file_w.c_str());

     // U :
     PRINTF_INFO("Reading fits file %s ...\n",fits_file_u.c_str());
     if( m_U.ReadFits( fits_file_u.c_str(), 0, 1, 1 ) ){
        printf("ERROR : could not read U FITS file %s\n",fits_file_u.c_str());
        return false;
     }else{
        PRINTF_INFO("OK : fits file %s read ok\n",fits_file_u.c_str());
     }
  
     // V : 
     PRINTF_INFO("Reading fits file %s ...\n",fits_file_v.c_str());
     if( m_V.ReadFits( fits_file_v.c_str(), 0, 1, 1 ) ){
        printf("ERROR : could not read V FITS file %s\n",fits_file_v.c_str());
        return false;
     }else{
        PRINTF_INFO("OK : fits file %s read ok\n",fits_file_v.c_str());
     }
  
     // W : 
     PRINTF_INFO("Reading fits file %s ...\n",fits_file_w.c_str());
     if( m_W.ReadFits( fits_file_w.c_str(), 0, 1, 1 ) ){
        printf("ERROR : could not read W FITS file %s\n",fits_file_w.c_str());
        return false;
     }else{
        PRINTF_INFO("OK : fits file %s read ok\n",fits_file_w.c_str());
     }
  }

  return true;
}

bool CPacerImager::read_corr_matrix( const char* basename, CBgFits& fits_vis_real, CBgFits& fits_vis_imag, 
                                     const char* szPostfix )
{
  // ensures initalisation of object structures 
  Initialise();
  
  // creating FITS file names for REAL, IMAG and U,V,W input FITS files :
  string fits_file_real = basename;
  fits_file_real += "_vis_real";
  if( strlen( szPostfix ) ){
     fits_file_real += szPostfix;
  }
  fits_file_real += ".fits";
 
  string fits_file_imag = basename;
  fits_file_imag += "_vis_imag";
  if( strlen( szPostfix ) ){
     fits_file_imag += szPostfix;
  }
  fits_file_imag += ".fits";
 

  if(CPacerImager::m_ImagerDebugLevel>=IMAGER_INFO_LEVEL){
     printf("Expecting the following files to exist:\n");
     printf("\t%s\n",fits_file_real.c_str()); 
     printf("\t%s\n",fits_file_imag.c_str()); 
  }
  
  // REAL(VIS)
  PRINTF_INFO("Reading fits file %s ...\n",fits_file_real.c_str());
  if( fits_vis_real.ReadFits( fits_file_real.c_str(), 0, 1, 1 ) ){
     printf("ERROR : could not read visibility FITS file %s\n",fits_file_real.c_str());
     return false;
  }else{
     PRINTF_INFO("OK : fits file %s read ok\n",fits_file_real.c_str());
  }

  // IMAG(VIS)
  PRINTF_INFO("Reading fits file %s ...\n",fits_file_imag.c_str());
  if( fits_vis_imag.ReadFits( fits_file_imag.c_str(), 0, 1, 1 ) ){
     printf("ERROR : could not read visibility FITS file %s\n",fits_file_imag.c_str());
     return false;
  }else{
     PRINTF_INFO("OK : fits file %s read ok\n",fits_file_imag.c_str());
  }
  
  // Test horizontal flip :
/*  fits_vis_real.HorFlip();
  fits_vis_imag.HorFlip();
  fits_vis_real.WriteFits("re_hor_flip.fits");
  fits_vis_imag.WriteFits("im_hor_flip.fits");*/
  
  // TEST Conjugate :
  // Conjugate is equivalent to changing FFTW BACKWARD TO FORWARD 
/*  for(int y=0;y<fits_vis_imag.GetYSize();y++){
     for(int x=0;x<fits_vis_imag.GetXSize();x++){
        if( x!=y ){
           double val = fits_vis_imag.getXY(x,y);
           fits_vis_imag.setXY(x,y,-val);
        }
     }
  }*/

  // reads or calculates UVW coordinates
  bool ret = ReadOrCalcUVW( basename , szPostfix );
  return ret;  
}

/*
gridding() + cuFFT combined in cuda 
Start Date: 01/12/2022
Expected End Date: 15/01/2022 
Actual End Date: 6:30 pm 01/12/2022

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
  if( CPacerImager::m_SaveFilesLevel >= SAVE_FILES_DEBUG )
  {
     char uv_grid_re_name[1024],uv_grid_im_name[1024],uv_grid_counter_name[1024];
     sprintf(uv_grid_re_name,"%s/uv_grid_real_%dx%d.fits",m_ImagerParameters.m_szOutputDirectory.c_str(),n_pixels,n_pixels);
     sprintf(uv_grid_im_name,"%s/uv_grid_imag_%dx%d.fits",m_ImagerParameters.m_szOutputDirectory.c_str(),n_pixels,n_pixels);
     sprintf(uv_grid_counter_name,"%s/uv_grid_counter_%dx%d.fits",m_ImagerParameters.m_szOutputDirectory.c_str(),n_pixels,n_pixels);
    
     if( uv_grid_real.WriteFits( uv_grid_re_name ) )
     {
        printf("ERROR : could not write output file %s\n",uv_grid_re_name);
     }
     else
     {
        PRINTF_INFO("INFO : saved file %s\n",uv_grid_re_name);
     }

     if( uv_grid_imag.WriteFits( uv_grid_im_name ) )
     {
        printf("ERROR : could not write output file %s\n",uv_grid_im_name);
     }
     else
     {
        PRINTF_INFO("INFO : saved file %s\n",uv_grid_im_name);
     }
  
     if( uv_grid_counter.WriteFits( uv_grid_counter_name ) )
     {
        printf("ERROR : could not write output file %s\n",uv_grid_counter_name);
     }
     else
     {
        PRINTF_INFO("INFO : saved file %s\n",uv_grid_counter_name);
     }
  }

  // Saving dirty_image() output files: 
  char outDirtyImageReal[1024],outDirtyImageImag[1024];   
   
   if( bSaveIntermediate ){ // I will keep this if - assuming it's always TRUE, but there is still control using , if bSaveIntermediate=false it has priority over m_SaveFilesLevel
      if( CPacerImager::m_SaveFilesLevel >= SAVE_FILES_DEBUG ){
         sprintf(outDirtyImageReal,"%s/dirty_test_real_%dx%d.fits",m_ImagerParameters.m_szOutputDirectory.c_str(),uv_grid_counter_xSize,uv_grid_counter_ySize);
         sprintf(outDirtyImageImag,"%s/dirty_test_imag_%dx%d.fits",m_ImagerParameters.m_szOutputDirectory.c_str(),uv_grid_counter_xSize,uv_grid_counter_ySize);
   
         out_image_real.WriteFits( outDirtyImageReal );
         out_image_imag.WriteFits( outDirtyImageImag );
      }
   }
   
   // 2022-04-02 : test change to use member variable for final image (have to be careful with threads and to not use this class as global variable):
   // calculate and save FFT-shifted image :
   // CBgFits out_image_real2( out_image_real.GetXSize(), out_image_real.GetYSize() ), out_image_imag2( out_image_real.GetXSize(), out_image_real.GetYSize() );
   AllocOutPutImages( out_image_real.GetXSize(), out_image_real.GetYSize() );
   
   if( !m_pSkyImageReal || !m_pSkyImageImag )
   {
      printf("ERROR in code : internal image buffers not allocated -> cannot continue\n");
      return;
   }

   fft_shift( out_image_real, *m_pSkyImageReal );
   fft_shift( out_image_imag, *m_pSkyImageImag );
   
   int rest = 1; // just so that by default it is !=0 -> image not saved 
   if( CPacerImager::m_SaveControlImageEveryNth > 0 )
   {
      rest = (m_SkyImageCounter % CPacerImager::m_SaveControlImageEveryNth);
      if( rest == 0 )
      {
          PRINTF_INFO("INFO : saving %d-th control sky image\n",m_SkyImageCounter);
      }
   }

   if( CPacerImager::m_SaveFilesLevel >= SAVE_FILES_FINAL || rest==0 )
   {   
      if( szBaseOutFitsName )
      {
         sprintf(outDirtyImageReal,"%s/%s_real.fits",m_ImagerParameters.m_szOutputDirectory.c_str(),szBaseOutFitsName);
      }
      else
      {
         // sprintf(outDirtyImageReal,"dirty_test_real_fftshift_%dx%d.fits",width,height);
         // const char* get_filename(  time_t ut_time , char* out_buffer, int usec=0, const char* full_dir_path="./", const char* prefix="dirty_image_", const char* postfix="", const char* formater="%.2u%.2u%.2uT%.2u%.2u%.2u" );
         get_filename( m_ImagerParameters.m_fUnixTime, outDirtyImageReal, m_ImagerParameters.m_szOutputDirectory.c_str(), "dirty_image_", "_real" ); // uxtime=0 -> it will be taken as current system time
      }
      SaveSkyImage( outDirtyImageReal , m_pSkyImageReal );
   
      if( bSaveImaginary )
      {
         if( szBaseOutFitsName )
         {
            sprintf(outDirtyImageImag,"%s/%s_imag.fits",m_ImagerParameters.m_szOutputDirectory.c_str(),szBaseOutFitsName);
         }
         else
         {
            // sprintf(outDirtyImageImag,"dirty_test_imag_fftshift_%dx%d.fits",width,height);
            get_filename( m_ImagerParameters.m_fUnixTime, outDirtyImageImag, m_ImagerParameters.m_szOutputDirectory.c_str(), "dirty_image_", "_imag" );
         }

         m_pSkyImageImag->SetFileName( outDirtyImageImag );      
         m_pSkyImageImag->WriteFits( outDirtyImageImag );
      }
   }
   
   if( CPacerImager::m_bPrintImageStatistics )
   {
      double mean, rms, minval, maxval, median, iqr, rmsiqr;
      int cnt;
      int radius = int( sqrt( m_pSkyImageReal->GetXSize()*m_pSkyImageReal->GetXSize() + m_pSkyImageReal->GetYSize()*m_pSkyImageReal->GetYSize() ) ) + 10;
      // m_SkyImageReal.GetStat( mean, rms, minval, maxval );
      m_pSkyImageReal->GetStatRadiusAll( mean, rms, minval, maxval, median, iqr, rmsiqr, cnt, radius, true );
      printf("STAT : full image %s statistics in radius = %d around the center using %d pixels : mean = %.6f , rms = %.6f, minval = %.6f, maxval = %.6f, median = %.6f, rms_iqr = %.6f\n",outDirtyImageReal,radius,cnt,mean, rms, minval, maxval, median, rmsiqr );
      
      
      // TODO : this will be parameterised to specified requested windown in the image to get RMS value from:
      double mean_window, rms_window, minval_window, maxval_window, median_window, iqr_window, rmsiqr_window;
      radius = 10; // TODO : make it use parameter and also position in the image 
      m_pSkyImageReal->GetStatRadiusAll( mean_window, rms_window, minval_window, maxval_window, median_window, iqr_window, rmsiqr_window, cnt, radius, true );
      printf("STAT : statistics of %s in radius = %d around the center using %d pixels : mean = %.6f , rms = %.6f, minval = %.6f, maxval = %.6f, median = %.6f, rms_iqr = %.6f\n",outDirtyImageReal,radius,cnt,mean_window, rms_window, minval_window, maxval_window, median_window, rmsiqr_window );
   }
   
   // TODO : re-grid to SKY COORDINATES !!!
   // convert cos(alpha) to alpha - see notes !!!
   // how to do it ???

// gridding + imaging ends
clock_t end_time = clock();
double duration_sec = double(end_time-start_time)/CLOCKS_PER_SEC;
double duration_ms = duration_sec*1000;
printf("CLOCK TOTAL GRIDDING + IMAGING TOOK: %.6f [seconds], %.3f [ms]\n",duration_sec,duration_ms);

}

bool CPacerImager::ApplySolutions( CBgFits& fits_vis_real, CBgFits& fits_vis_imag, double frequency_mhz, CCalSols& calsol, const char* szPol )
{
   double freq_diff = fabs( frequency_mhz - calsol.m_frequency_mhz );
   
   if( freq_diff > 5.00 ){
      printf("WARNING : frequency of calibration solutions is different by more than 1 MHz from the actual data -> no calibration applied (data frequency %.1f vs. calsol frequency %.1f)\n",frequency_mhz,calsol.m_frequency_mhz);
      
      return false;
   }
   
   if( fits_vis_real.GetXSize() != calsol.size() ){
      printf("WARNING : wrong number of calibration solutions (%d vs. required %d)\n",int(calsol.size()),fits_vis_real.GetXSize());
      
      return false;
   }
   
   int pol_idx = 0;
   if ( strcmp(szPol,"Y") == 0 ){
      pol_idx = 3;
   }
   
   int n_ant = fits_vis_real.GetXSize();
   for(int y=0;y<n_ant;y++){
      for(int x=0;x<n_ant;x++){
         double re = fits_vis_real.getXY(x,y);
         double im = fits_vis_imag.getXY(x,y);
         
         std::complex<double> z(re,im);
         
         std::complex<double> g1 = calsol[x].m_cal[pol_idx]; // or y ? TODO , TBD 
         std::complex<double> g2_c = std::conj( calsol[y].m_cal[pol_idx] ); // or x ? TODO , TBD
         
// 1/g :         
//         std::complex<double> z_cal = (1.00/g1)*z*(1.00/g2_c); // or g1 , g2_c without 1.00/ ??? TBD vs. --invert amplitude in read cal. sol. from MCCS DB 
// g :
         std::complex<double> z_cal = (g1)*z*(g2_c);
         if( x<=5 && y<=5 ){
            printf("DEBUG : z = %.4f + i%4f -> %.4f + i%4f\n",z.real(),z.imag(),z_cal.real(),z_cal.imag());
         }
         
         fits_vis_real.setXY(x,y,z_cal.real());
         fits_vis_imag.setXY(x,y,z_cal.imag());
      }
   }
   
   return true;
}

// run_imager 
bool CPacerImager::run_imager( CBgFits& fits_vis_real, CBgFits& fits_vis_imag, CBgFits& fits_vis_u, CBgFits& fits_vis_v, CBgFits& fits_vis_w, 
                               double frequency_mhz, 
                               int n_pixels,
                               double FOV_degrees,
                               double min_uv,                  /*=-1000,*/
                              //  bool do_gridding,               /*=true*/
                              //  bool do_dirty_image,            /*=true*/
                               bool do_gridding_imaging,       /*=true*/
                               const char* weighting,          /* ="" */   // weighting : U for uniform (others not implemented)
                               const char* in_fits_file_uv_re, /*=""*/ // gridded visibilities can be provided externally
                               const char* in_fits_file_uv_im, /*=""*/ // gridded visibilities can be provided externally
                               const char* szBaseOutFitsName   /*=NULL*/
)
{
  // ensures initalisation of object structures 
  Initialise();

  PACER_PROFILER_START
  
  // based on RTS : UV pixel size as function FOVtoGridsize in  /home/msok/mwa_software/RTS_128t/src/gridder.c  
  double frequency_hz = frequency_mhz*1e6;
  double wavelength_m = VEL_LIGHT / frequency_hz;
  double FoV_radians = FOV_degrees*M_PI/180.;
  // WARNING: it actually cancels out to end up being 1/PI :
  // TODO simplity + 1/(2FoV) !!! should be 
//  double delta_u = ( (VEL_LIGHT/frequency_hz)/(FOV_degrees*M_PI/180.) ) / wavelength_m; // in meters (NOT WAVELENGHTS)
//  double delta_v = ( (VEL_LIGHT/frequency_hz)/(FOV_degrees*M_PI/180.) ) / wavelength_m; // in meters (NOT WAVELENGHTS)
// TODO : 
  double delta_u = 1.00/(FoV_radians); // should be 2*FoV_radians - see TMS etc 
  double delta_v = 1.00/(FoV_radians); // Rick Perley page 16 : /home/msok/Desktop/PAWSEY/PaCER/doc/Imaging_basics/ATNF2014Imaging.pdf
  
  // test forced to this value (based on pixscale in MIRIAD):
  if( m_bCompareToMiriad ){
     // Brute force comparison to MIRIAD assuming pixscale from the final image FITS FILE = 0.771290200761 degree 
     if( fabs(frequency_mhz-159.375) <= 0.5 ){
        delta_u = 0.412697967; // at ch=204
        delta_v = 0.412697967; // at ch=204
     }
     if( fabs(frequency_mhz-246.09375) <= 0.5 ){
        delta_u = .63624180895350226815; // at ch=315
        delta_v = .63624180895350226815; // at ch=315
     }
  }
  
  // Apply Calibration if provided :
  if( m_CalibrationSolutions.size() > 0 )
  {
     if( m_CalibrationSolutions.size() == fits_vis_real.GetXSize() )
     {
        printf("INFO : applying calibration solutions (for %d antennas)\n",int(m_CalibrationSolutions.size()));
        ApplySolutions( fits_vis_real, fits_vis_imag, frequency_mhz, m_CalibrationSolutions );
     }
     else
     {
        printf("WARNING : wrong number of calibration solutions (%d vs. required %d)\n",int(m_CalibrationSolutions.size()),fits_vis_real.GetXSize());
     }
  }

  CBgFits uv_grid_counter( n_pixels, n_pixels ),uv_grid_real( n_pixels, n_pixels ) , uv_grid_imag( n_pixels, n_pixels );  

  if(do_gridding_imaging)
  {
     gridding_dirty_image(fits_vis_real, fits_vis_imag, fits_vis_u, fits_vis_v, fits_vis_w, uv_grid_real, uv_grid_imag, uv_grid_counter, 
                           delta_u, delta_v,frequency_mhz, n_pixels, min_uv, weighting, 
                           true, szBaseOutFitsName, true, false 
                           ); 
  }      

  // increse image counter:
  m_SkyImageCounter++;
  if( m_SkyImageCounter >= INT_MAX )
  {
    PRINTF_WARNING("WARNING : image counter reached maximum value for int = %d -> reset to zero\n",INT_MAX);
    m_SkyImageCounter = 0;
  }
  
  PACER_PROFILER_END("full imaging (gridding + dirty image) took")

  return  true;   
}

// run_imager 
//-----------------------------------------------------------------------------------------------------------------------------
// Wrapper to run_imager :
// Reads FITS files and executes overloaded function run_imager ( as above )
//-----------------------------------------------------------------------------------------------------------------------------
bool CPacerImager::run_imager( const char* basename, const char* szPostfix,
                               double frequency_mhz, 
                               int n_pixels,
                               double FOV_degrees,
                               double min_uv,                  /*=-1000,*/
                              //  bool do_gridding,               /*=true*/
                              //  bool do_dirty_image,            /*=true*/
                               bool do_gridding_imaging,          /*=true*/
                               const char* weighting,          /* ="" */   // weighting : U for uniform (others not implemented)
                               const char* in_fits_file_uv_re, /*=""*/ // gridded visibilities can be provided externally
                               const char* in_fits_file_uv_im, /*=""*/ // gridded visibilities can be provided externally                               
                               const char* szBaseOutFitsName   /*=NULL*/
                  )
{
   // ensures initalisation of object structures 
   Initialise();  

   // read input data (correlation matrix and UVW) :
   CBgFits fits_vis_real, fits_vis_imag;

   if( read_corr_matrix( basename, fits_vis_real, fits_vis_imag, szPostfix ) )
   { // also included reading or calculation of UVW 
      PRINTF_INFO("OK : input files read ok\n");
   }
   else
   {
      printf("ERROR : could not read one of the input files\n");
      return false;
   }
   
   // read calibration solutions (if specified) :
   if( m_CalibrationSolutions.read_calsolutions() > 0 )
   {
      m_CalibrationSolutions.show();
   }

   bool ret = run_imager( fits_vis_real, fits_vis_imag, m_U, m_V, m_W,
                         frequency_mhz, 
                         n_pixels, 
                         FOV_degrees, 
                         min_uv,
                        //  do_gridding, 
                        //  do_dirty_image, 
                         do_gridding_imaging, 
                         weighting, 
                         in_fits_file_uv_re, in_fits_file_uv_im, 
                         szBaseOutFitsName
                        ); 

   return ret;               
}

//-----------------------------------------------------------------------------------------------------------------------------
// Wrapper to run_imager - executes overloaded function run_imager ( as above ) :
//  INPUT : pointer to data
//-----------------------------------------------------------------------------------------------------------------------------
bool CPacerImager::run_imager( float* data_real, 
                               float* data_imag,
                               int n_ant, 
                               int n_pol,
                               double frequency_mhz, 
                               int n_pixels,
                               double FOV_degrees,
                               double min_uv,                   /*=-1000,*/
                              //  bool do_gridding,                /*=true  */
                              //  bool do_dirty_image,             /*=true  */
                               bool do_gridding_imaging,        /*=true */
                               const char* weighting,           /* =""   */   // weighting : U for uniform (others not implement
                               const char* szBaseOutFitsName,   /* =NULL */
                               bool bCornerTurnRequired         /* =true , TODO : change default to false and perform corner-turn in eda2 imager code using correlation matrix from xGPU correlator */
                             ) 
{
  // ensures initalisation of object structures 
  Initialise();

  CBgFits fits_vis_real( n_ant, n_ant ), fits_vis_imag( n_ant, n_ant );
  
  if( bCornerTurnRequired )
  {
     fits_vis_real.SetValue(0.00);
     fits_vis_imag.SetValue(0.00);

     // ~corner-turn operation which can be quickly done on GPU: 
     int counter=0;
     for(int i=0;i<n_ant;i++)
     {
        for(int j=0;j<(i + 1);j++)
        {
           fits_vis_real.setXY( i, j , data_real[counter] );
           fits_vis_real.setXY( j, i , data_real[counter] );
           counter++;
        }
     }

     counter=0;
     for(int i=0;i<n_ant;i++)
     {
        for(int j=0;j<(i + 1);j++)
        {
           fits_vis_imag.setXY( i, j , data_imag[counter] );
           fits_vis_imag.setXY( j, i , -(data_imag[counter]) );
           counter++;
        }
     }
  }
  else
  {
     fits_vis_real.SetData( data_real );
     fits_vis_imag.SetData( data_imag );
  }
  
  // const char* get_filename(  time_t ut_time , char* out_buffer, int usec=0, const char* full_dir_path="./", const char* prefix="dirty_image_", const char* postfix=""
  if( CPacerImager::m_SaveFilesLevel >= SAVE_FILES_INFO ){
     char outDirtyImageReal[1024];
     // sprintf(outDirtyImageReal,"%s/%s_vis_real.fits",m_ImagerParameters.m_szOutputDirectory.c_str(),szBaseOutFitsName);
     get_filename( m_ImagerParameters.m_fUnixTime, outDirtyImageReal, m_ImagerParameters.m_szOutputDirectory.c_str(), "visibility_", "_real" );
   
     fits_vis_real.WriteFits( outDirtyImageReal );
  }
     
  if( CPacerImager::m_SaveFilesLevel >= SAVE_FILES_INFO ){
     char outDirtyImageImag[1024];
     // sprintf(outDirtyImageImag,"%s/%s_vis_imag.fits",m_ImagerParameters.m_szOutputDirectory.c_str(),szBaseOutFitsName);
     get_filename( m_ImagerParameters.m_fUnixTime, outDirtyImageImag, m_ImagerParameters.m_szOutputDirectory.c_str(), "visibility_", "_imag" );
        
     fits_vis_imag.WriteFits( outDirtyImageImag );
  }

  // calculate UVW (if required)
  CalculateUVW();
  
  bool ret = run_imager( fits_vis_real, fits_vis_imag, m_U, m_V, m_W,
                         frequency_mhz, 
                         n_pixels, 
                         FOV_degrees, 
                         min_uv,
                        //  do_gridding, 
                        //  do_dirty_image,
                         do_gridding_imaging, 
                         weighting, "","", szBaseOutFitsName );

   return ret;    
}
