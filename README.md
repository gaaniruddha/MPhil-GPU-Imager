**Test-version of the GPU-imager for a single time and frequency channel.**
- gridding_imaging.h: Contains code for the gridding kernel defined.
- pacer_imager.h: Contains all function declarations.
- pacer_imager.cu: Contains all the function definitions of those functions declared in pacer_imager.h
- pacer_imager_main.cpp: The main program takes in user inputs, calls the gridding kernel, and performs cuFFT.
  
**Notes:**
- The starting version of the code was provided to me by my supervisor.
- Hence I have removed (~90%) the sections coded by him, and only kept the code segments written by me. 
