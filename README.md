**Test-version of the GPU imager for a single time and frequency channel.**
- gridding_imaging.h: Contains code for the gridding kernel defined.
- pacer_imager.h: Contains all the function declarations.
- pacer_imager.cu: Contains all the function definitions of those functions declared in pacer_imager.h
- GPU_Imager_cufftPlanMany.pdf: Contains code for the test version of the GPU imager for a single time step with multiple channels.
- Imager_Benchmarks_Visualisations.xlsx: Benchmarks of the different versions of the test GPU imagers. Also contains plots of these benchmarks generated using Google Sheets. 
  
**Notes:**
- The starting version of the code was provided to me by my supervisor.
- Hence, I have removed (almost all except a few where code continuity was needed) the sections coded by him and only kept the code segments written by me. 
