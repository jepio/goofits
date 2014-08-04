#include "ParabolaPdf.hh"

EXEC_TARGET fptype device_Parabola(fptype *evt, fptype *pars, unsigned int*
indices){
    fptype x = evt[indices[2+indices[0]]];
    fptype a = pars[indices[1]];
    fptype p = pars[indices[2]];
    fptype q = pars[indices[3]];
    fptype ret = a*(x-p)*(x-p) +q;
    if (ret < 0)
        ret *= -1;
    return ret;
}

MEM_DEVICE device_function_ptr ptr_to_Parabola = device_Parabola;

__host__ ParabolaPdf::ParabolaPdf(std::string n, Variable* _x, Variable* a,
                                Variable* p, Variable* q) : GooPdf(_x,n)
{
    std::vector<unsigned int> pindices;
    pindices.push_back(registerParameter(a));
    pindices.push_back(registerParameter(p));
    pindices.push_back(registerParameter(q));
    GET_FUNCTION_ADDR(ptr_to_Parabola);
    initialise(pindices);
}
