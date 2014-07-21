#ifndef NOVO_PDF_HH
#define NOVO_PDF_HH

#include "GooPdf.hh" 

class NovoPdf : public GooPdf{
public:
  NovoPdf (std::string n, Variable* _x, Variable* p, Variable* w, Variable* t); 
//  __host__ fptype integrate (fptype lo, fptype hi) const; 
//  __host__ virtual bool hasAnalyticIntegral () const {return true;} 



private:

};

#endif
