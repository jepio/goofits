#include "BifPdf.hh"

EXEC_TARGET fptype device_Bif (fptype* evt, fptype* p, unsigned int* indices) {
  fptype x = evt[indices[2 + indices[0]]]; 
  fptype mean = p[indices[1]];
  fptype sL = p[indices[2]];
  fptype sR = p[indices[3]];

  fptype arg = x - mean;
  fptype coef = 0.0;
  if (arg < 0.0){
    if (FABS(sL) > 1e-30) coef= -0.5/(sL*sL);
  }
  else{
    if (FABS(sR) > 1e-30) coef= -0.5/(sR*sR);
  }

  fptype ret = EXP(coef*arg*arg);

  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_Bif = device_Bif; 

__host__ BifPdf::BifPdf (std::string n, Variable* _x,
Variable* mean, Variable* sL, Variable* sR) 
  : GooPdf(_x, n) 
{
  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(mean));
  pindices.push_back(registerParameter(sL));
  pindices.push_back(registerParameter(sR));
  GET_FUNCTION_ADDR(ptr_to_Bif);
  initialise(pindices); 
}

