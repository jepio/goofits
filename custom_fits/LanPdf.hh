#ifndef LAN_PDF_HH
#define LAN_PDF_HH

#include "GooPdf.hh" 

class LanPdf : public GooPdf {
public:
  LanPdf (std::string n, Variable* _x, Variable* mean, Variable* scale); 
//  __host__ fptype integrate (fptype lo, fptype hi) const; 
//  __host__ virtual bool hasAnalyticIntegral () const {return true;} 



private:

};

#endif
