#include "Variable.hh"
#include "FitManager.hh"
#include "UnbinnedDataSet.hh"
// Rozkład
#include "GaussianPdf.hh"
#include "BWPdf.hh"
#include "ProdPdf.hh"
// I/O c++
#include <iostream>
#include <fstream>

int main(void){
    // Nazwa zmiennej, limit dolny, limit górny
    Variable *xvar = new Variable("xvar", 0, 5);
    
    // Przypisanie zmiennej do zbioru.
    // Może być też vector zmiennych.
    UnbinnedDataSet data(xvar);

    // Wczytanie danych
    std::ifstream plik;
    plik.open("prod.txt");
    while (!plik.eof()){
        // Wypełnienie zmiennej wartością
        plik >> xvar->value;
        // Dodanie eventu do zbioru
        data.addEvent();
    }
    
    // Nazwa, wartość początkowa, limit dolny, limit górny
    Variable * mean = new Variable("mean", 0 ,-10,10);
    Variable * sigma = new Variable("sigm", 1, 0.0, 1.5);
    Variable * meanb = new Variable("mean_bw", 1, -10, 10);
    Variable * sigb = new Variable("sigma_bw", 1, 0.0, 1.5);
    // Nazwa, obserwabla, parametry.
    // Konstrukcja zależy od konkretnego rozkładu
    GaussianPdf gauss("gauss", xvar, mean, sigma);
    BWPdf bw("breit-wigner", xvar, meanb, sigb);
    
    // Wektor rozkładów
    vector<PdfBase*> pdfList;
    pdfList.push_back(&gauss);
    pdfList.push_back(&bw);
    
    // Iloczyn rozkładów
    ProdPdf * product = new ProdPdf("product", pdfList);
    
    product->setData(&data);

    // Przekazanie rozkładu z danymi do FitManager'a.
    FitManager fitter (product);
    // Uruchomienie fitu. Wynik zostanie wypisany na stdin
    fitter.fit();
    
    return 0;
}
