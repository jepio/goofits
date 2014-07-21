#ifndef BIF_PDF_HH
#define BIF_PDF_HH

#include "GooPdf.hh" 

class BifPdf : public GooPdf {
public:
  BifPdf (std::string n, Variable* _x, Variable* mean, Variable* sL, Variable* sR); 
//  __host__ fptype integrate (fptype lo, fptype hi) const; 
//  __host__ virtual bool hasAnalyticIntegral () const {return true;} 



private:

};

#endif
