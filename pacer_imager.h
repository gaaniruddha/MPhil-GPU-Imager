#ifndef _PACER_IMAGER_H__
#define _PACER_IMAGER_H__

#include "antenna_positions.h"
#include "apply_calibration.h"
#include "pacer_imager_parameters.h"
#include <observation_metadata.h>
#include <bg_fits.h>

#include <string>
using namespace std;

// FFTW, math etc :
// #include <fftw3.h>

// for all cuda related stuff! 
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
 
protected :
   // FFT shift to convert from DC in bin 0 of the FFTW output array to DC in the center bin :
   void fft_shift( CBgFits& dirty_image, CBgFits& out_image );

   // FFT unshift converts from DC in the center bin to DC in bin 0 (as expected input to complex FFTW)
   void fft_unshift( CBgFits& dirty_image, CBgFits& out_image );

   // check if image size is as required and re-alloc if not:
   // Return : 
   //    false - if image size was ok and nothing was required
   //    true  - if image size was changed
   bool CheckSize( CBgFits& image, int sizeX, int sizeY );
   
   // allocate memory if not yet done (NULL pointers) or check image size and re-size if needed :
   bool AllocOutPutImages( int sizeX, int sizeY );
   
   // clean local variables allocated locally 
   void CleanLocalAllocations();

   // 06/12/22 
   // Allocating GPU Memory 
   void AllocGPUMemory(int XYSIZE, int IMAGE_SIZE); 

   // Clean GPU Memory 
   void CleanGPUMemory(); 

public :
   // TODO: decide if this should be static or member variables
   // debug level for the whole library / program :
   static int m_ImagerDebugLevel; // see pacer_imager_defs.h for defines IMAGER_DEBUG_LEVEL etc
   
   // level of saving intermediate and test files , see pacer_imager_defs.h for defines SAVE_FILES_NONE
   static int m_SaveFilesLevel;
   
   // can also save control files every N-th file 
   static int m_SaveControlImageEveryNth;
   
   // show statistics 
   static bool m_bPrintImageStatistics;
   
   // TESTING and VALIDATION :
   static bool m_bCompareToMiriad; // should always be FALSE and set to TRUE only to test against MIRIAD 
   
   // include auto-correlations in the imaging :
   bool m_bIncludeAutos;

   // parameters :
   // WARNING : I am assuming they are the same for all objects of this class.
   //           we shall see and potentially remove "static"
   CImagerParameters m_ImagerParameters;

   // Antenna positions :   
   // CAntennaPositions m_AntennaPositions;
   
   // meta data :
   CObsMetadata m_MetaData;
   
   // calibration solutions for a single frequency channel (assuming one object of CPacerImager is per frequency channel)
   CCalSols m_CalibrationSolutions;
   
   // Flagged antennas , if list m_AntennaPositions is filled it will also be updated (field flag)
   vector<int> m_FlaggedAntennas;
   
   // flag if initialisation already performed
   bool m_bInitialised;
   
   
   // UVW for SKA-Low station zenith phase-centered all-sky imaging :
   CBgFits m_U;
   CBgFits m_V;
   CBgFits m_W;
   int m_Baselines; // number of calculated baselines, also indicator if m_U, m_V and m_W have been initialised 
   
   // values calculated for the current image :
   double m_PixscaleAtZenith;
   
   //
   // Resulting sky images :
   // WARNING : be careful with using this object as global !
   CBgFits* m_pSkyImageReal;
   CBgFits* m_pSkyImageImag;
   bool     m_bLocalAllocation;

   //-------------------------------------------------------------------------------------------------------------
   // image counter : counts how many sky images have been already created
   //-------------------------------------------------------------------------------------------------------------
   int      m_SkyImageCounter;
   

   CPacerImager();
   ~CPacerImager();
   
   //-----------------------------------------------------------------------------------------------------------------------------
   // Initialisation of data structures in the object, which are required for imaging 
   //-----------------------------------------------------------------------------------------------------------------------------   
   void Initialise(); // implement initialisation of object here, read antenna positions, calculate UVW if constant etc 
 
   // Set / Get functions :
   //-----------------------------------------------------------------------------------------------------------------------------
   // verbosity level 
   //-----------------------------------------------------------------------------------------------------------------------------
   static void SetDebugLevel( int debug_level );
   
   //-----------------------------------------------------------------------------------------------------------------------------
   // File save level
   //-----------------------------------------------------------------------------------------------------------------------------
   static void SetFileLevel( int filesave_level );
   
   void SetFlaggedAntennas( vector<int>& flagged_antennas);

   //-----------------------------------------------------------------------------------------------------------------------------
   // calculates UVW and also checks if it is required at all 
   //-----------------------------------------------------------------------------------------------------------------------------
   bool CalculateUVW();
   
   //-----------------------------------------------------------------------------------------------------------------------------
   // Reads UVW from FITS files with files names basename + szPostfix + U/V/W.fits
   // or calculates using antenna positions
   //-----------------------------------------------------------------------------------------------------------------------------
   bool ReadOrCalcUVW( const char* basename, const char* szPostfix );
   
   //-----------------------------------------------------------------------------------------------------------------------------
   // Set external buffers for output images (Real/Imag) 
   //-----------------------------------------------------------------------------------------------------------------------------
   void SetOutputImagesExternal( CBgFits* pSkyImageRealExt, CBgFits* pSkyImageImagExt );
   
   //-----------------------------------------------------------------------------------------------------------------------------
   // 1st version producing a dirty image (tested on both MWA and SKA-Low).
   // TODO : Test cases can be found in PaCER documentation 
   //-----------------------------------------------------------------------------------------------------------------------------
   void dirty_image( CBgFits& uv_grid_real_param, CBgFits& uv_grid_imag_param, CBgFits& uv_grid_counter, 
                     bool bSaveIntermediate=false, const char* szBaseOutFitsName=NULL, bool bSaveImaginary=true, bool bFFTUnShift=true );

   //-----------------------------------------------------------------------------------------------------------------------------
   // read input data correlation matrix and UVW from FITS files (this is the first type of input )
   // TODO : add other input formats (e.g. binary data)
   //-----------------------------------------------------------------------------------------------------------------------------
   bool read_corr_matrix( const char* basename, CBgFits& fits_vis_real, CBgFits& fits_vis_imag, const char* szPostfix );

   //------------
   void gridding_dirty_image( CBgFits& fits_vis_real, CBgFits& fits_vis_imag, CBgFits& fits_vis_u, CBgFits& fits_vis_v, CBgFits& fits_vis_w,
               CBgFits& uv_grid_real, CBgFits& uv_grid_imag, CBgFits& uv_grid_counter, double delta_u, double delta_v, 
               double frequency_mhz, 
               int n_pixels,
               double min_uv /*=-1000*/,
               const char* weighting /*="" weighting : U for uniform (others not implemented) */,
               /*CBgFits& uv_grid_real_param, CBgFits& uv_grid_imag_param, CBgFits& uv_grid_counter, ALREADY PASSED */ 
               bool bSaveIntermediate /*=false*/ , const char* szBaseOutFitsName /*=NULL*/, 
               bool bSaveImaginary /*=true*/ , bool bFFTUnShift /*=true*/); 
   // ------------------------------------------------------------------------------------------------------------------
   // Executes imager:
   // INPUT  : 
   //          fits_vis_real, fits_vis_imag : visibilities (REAL and IMAG 2D arrays as FITS class) 
   //          fits_vis_u, fits_vis_v, fits_vis_w : UVW (real values baselines in units of wavelength - see TMS)     
   // -----------------------------------------------------------------------------------------------------------------------------
   bool run_imager( CBgFits& fits_vis_real, CBgFits& fits_vis_imag, CBgFits& fits_vis_u, CBgFits& fits_vis_v, CBgFits& fits_vis_w, 
                    double frequency_mhz,
                    int    n_pixels,
                    double FOV_degrees,
                    double min_uv=-1000,        // minimum UV 
                  //   bool   do_gridding=true,    // excute gridding  (?)
                  //   bool   do_dirty_image=true, // form dirty image (?)
                    bool   do_gridding_imaging=true, 
                    const char* weighting="",   // weighting : U for uniform (others not implemented)
                    const char* in_fits_file_uv_re="", // gridded visibilities can be provided externally
                    const char* in_fits_file_uv_im="",  // gridded visibilities can be provided externally
                    const char* szBaseOutFitsName=NULL
                  );

   //-----------------------------------------------------------------------------------------------------------------------------
   // Wrapper to run_imager :
   // Reads FITS files and executes overloaded function run_imager ( as above )
   //-----------------------------------------------------------------------------------------------------------------------------
   bool run_imager( const char* basename, const char* szPostfix,
                    double frequency_mhz,
                    int    n_pixels,
                    double FOV_degrees,
                    double min_uv=-1000,        // minimum UV 
                  //   bool   do_gridding=true,    // excute gridding  (?)
                  //   bool   do_dirty_image=true, // form dirty image (?)
                    bool do_gridding_imaging=true,
                    const char* weighting="",   // weighting : U for uniform (others not implemented)
                    const char* in_fits_file_uv_re="", // gridded visibilities can be provided externally
                    const char* in_fits_file_uv_im="", // gridded visibilities can be provided externally                    
                    const char* szBaseOutFitsName=NULL
                  );
                  
   bool run_imager( float* data_real, 
                    float* data_imag,
                    int n_ant, 
                    int n_pol,
                    double frequency_mhz, 
                    int n_pixels,
                    double FOV_degrees,
                    double min_uv=-1000,      // minimum UV
                  //   bool do_gridding=true,    // excute gridding  (?)
                  //   bool do_dirty_image=true, // form dirty image (?)
                    bool do_gridding_imaging=true, 
                    const char* weighting="", // weighting : U for uniform (others not implemented)
                    const char* szBaseOutFitsName=NULL,
                    bool bCornerTurnRequired=true // changes indexing of data "corner-turn" from xGPU structure to continues (FITS image-like)
                  );

   //-----------------------------------------------------------------------------------------------------------------------------
   // Apply calibration solutions :
   //-----------------------------------------------------------------------------------------------------------------------------
   bool ApplySolutions( CBgFits& fits_vis_real, CBgFits& fits_vis_imag, double frequency_mhz, CCalSols& calsol, const char* szPol="X" );

   //-----------------------------------------------------------------------------------------------------------------------------
   // Function saving output FITS files: 
   //-----------------------------------------------------------------------------------------------------------------------------
   bool SaveSkyImage( const char* outFitsName, CBgFits* pFits );
                  
};

#endif 