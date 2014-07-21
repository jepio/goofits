#include "LanPdf.hh"

EXEC_TARGET fptype device_Lan (fptype* evt, fptype* p, unsigned int* indices) {
  fptype x = evt[indices[2 + indices[0]]]; 
  fptype mean = p[indices[1]];
  fptype scale = p[indices[2]];

  fptype arg = 0.0;
  if (FABS(scale) > 1e-30) arg = (x - mean)/scale;
   
  fptype ret = EXP(-arg) + arg;
  ret *= -0.5;
  ret = EXP(ret);

  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_Lan = device_Lan; 

__host__ LanPdf::LanPdf (std::string n, Variable* _x,
Variable* mean, Variable* scale) 
  : GooPdf(_x, n) 
{
  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(mean));
  pindices.push_back(registerParameter(scale));
  GET_FUNCTION_ADDR(ptr_to_Lan);
  initialise(pindices); 
}

