**Test-version of the GPU-imager for a single time and frequency channel.**
- gridding_imaging.h: Contains code for the gridding kernel defined.
- pacer_imager.h: Contains all function declarations.
- pacer_imager.cu: Contains all the function definitions of those functions declared in pacer_imager.h
- pacer_imager_main.cpp: The main program takes in user inputs, calls the gridding kernel, and performs cuFFT.
  
