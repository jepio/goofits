#include "Variable.hh"
#include "FitManager.hh"
#include "UnbinnedDataSet.hh"
// Rozkład
#include "GaussianPdf.hh"
// I/O c++
#include <iostream>
#include <fstream>

int main(void){
    // Nazwa zmiennej, limit dolny, limit górny
    Variable *xvar = new Variable("xvar", -5, 5);
    
    // Przypisanie zmiennej do zbioru.
    // Może być też vector zmiennych.
    UnbinnedDataSet data(xvar);

    // Wczytanie danych
    std::ifstream plik;
    plik.open("dane.txt");
    while (!plik.eof()){
        // Wypełnienie zmiennej wartością
        plik >> xvar->value;
        // Dodanie eventu do zbioru
        data.addEvent();
    }
    
    // Nazwa, wartość początkowa, limit dolny, limit górny
    Variable * mean = new Variable("mean", 0 ,-10,10);
    Variable * sigma = new Variable("sigm", 1, 0.5, 1.5);
    
    // Nazwa, obserwabla, parametry.
    // Konstrukcja zależy od konkretnego rozkładu
    GaussianPdf gauss("gauss", xvar, mean, sigma);
    
    // Dane data należą do rozkładu gauss
    gauss.setData(&data);

    // Przekazanie rozkładu z danymi do FitManager'a.
    FitManager fitter (&gauss);
    // Uruchomienie fitu. Wynik zostanie wypisany na stdin
    fitter.fit();
    
    return 0;
}
