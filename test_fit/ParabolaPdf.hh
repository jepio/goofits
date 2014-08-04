#include "GooPdf.hh"

class ParabolaPdf : public GooPdf{
    public:
        ParabolaPdf(std::string n, Variable *_x, Variable *a, Variable* p,Variable* q);
};
