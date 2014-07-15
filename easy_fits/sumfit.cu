#include "Variable.hh"
#include "FitManager.hh"
#include "UnbinnedDataSet.hh"
// Rozkład
#include "GaussianPdf.hh"
#include "ExpPdf.hh"
#include "AddPdf.hh"
// I/O c++
#include <iostream>
#include <fstream>

int main(void){
    // Nazwa zmiennej, limit dolny, limit górny
    Variable *xvar = new Variable("xvar", 0, 20);
    
    // Przypisanie zmiennej do zbioru.
    // Może być też vector zmiennych.
    UnbinnedDataSet data(xvar);

    // Wczytanie danych
    std::ifstream plik;
    plik.open("sum.txt");
    while (!plik.eof()){
        // Wypełnienie zmiennej wartością
        plik >> xvar->value;
        // Dodanie eventu do zbioru
        data.addEvent();
    }
    
    // Nazwa, wartość początkowa, limit dolny, limit górny
    Variable * mean = new Variable("mean", 0 ,-10,10);
    Variable * sigma = new Variable("sigm", 1, 0.0, 1.5);
    Variable * alpha = new Variable("alpha", 1, -10, 10);
    // Nazwa, obserwabla, parametry.
    // Konstrukcja zależy od konkretnego rozkładu
    GaussianPdf gauss("gauss", xvar, mean, sigma);
    ExpPdf expo("exp", xvar, alpha);
    // Wektor rozkładów
    vector<PdfBase*> pdfList;
    pdfList.push_back(&expo);
    pdfList.push_back(&gauss);
    // Wektor wag (# przypadków)
    vector<Variable*> vars;
    Variable *expoFrac = new Variable("expoFrac",0.5,0.,10000);
    Variable *gausFrac = new Variable("gausFrac",0.5,0.,10000);
    vars.push_back(expoFrac);
    vars.push_back(gausFrac);
    // Suma rozkładów
    AddPdf * product = new AddPdf("sum",vars, pdfList);
    
    product->setData(&data);

    // Przekazanie rozkładu z danymi do FitManager'a.
    FitManager fitter (product);
    // Uruchomienie fitu. Wynik zostanie wypisany na stdin
    fitter.fit();
    
    return 0;
}
