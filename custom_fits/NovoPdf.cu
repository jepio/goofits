#include "NovoPdf.hh"

EXEC_TARGET fptype device_Novo (fptype* evt, fptype* p, unsigned int* indices) {
  fptype x = evt[indices[2 + indices[0]]]; 
  fptype peak = p[indices[1]];
  fptype width = p[indices[2]];
  fptype tail = p[indices[3]];

  fptype qa=0,qb=0,qc=0,qx=0,qy=0;
  if (fabs(tail) < 1.e-7)
      qc = 0.5*POW(((x-peak)/width),2);
  else {
      qa = tail*SQRT(LOG(4.));
      qb = SINH(qa)/qa;
      qx = (x - peak)/width*qb;
      qy = 1.+tail*qx;

      if (qy > 1.E-7)
          qc = 0.5*(POW((LOG(qy)/tail),2) + tail*tail);
      else
          qc = 15.0;
  }
  fptype ret = EXP(-qc);
  
  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_Novo = device_Novo; 

__host__ NovoPdf::NovoPdf (std::string n, Variable* _x,
Variable* p, Variable* w, Variable* t) 
  : GooPdf(_x, n) 
{
  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(p));
  pindices.push_back(registerParameter(w));
  pindices.push_back(registerParameter(t));
  GET_FUNCTION_ADDR(ptr_to_Novo);
  initialise(pindices); 
}

