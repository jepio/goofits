#include "Variable.hh"
#include "GaussianPdf.hh"
#include "ExpPdf.hh"
#include "AddPdf.hh"
#include "FitManager.hh"
#include "UnbinnedDataSet.hh"

#include <iostream>
#include <fstream>

#include "TH1F.h"
#include "TCanvas.h"

int main(){
    std::ifstream input_file;
    input_file.open("mixed.dat");

    Variable * xvar = new Variable("xvar", 0, 20);
    xvar->numbins = 1000; // Don't quite know what the effect of this is.
    vector<Variable*> vars;
    vars.push_back(xvar);
    UnbinnedDataSet data(vars);
    
    // Filling the data set and the histogram of the variable.
    int totalData = 0;
    TH1F xvarhist("xvarHist","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    xvarhist.SetStats(false);
    double value;
    while (!input_file.eof()){
        input_file >> value;
        xvar->value = value;
        data.addEvent();
        xvarhist.Fill(value);
        totalData++;
        //std::cout << totalData << " " << value << std::endl; 

    }
    std::cout << totalData << std::endl;
    
    // Defining the PDF parameters.
    Variable * mean1 = new Variable("mean1",5,0,20);
    Variable * mean2 = new Variable("mean2",10,0,20);
    Variable * sigma1 = new Variable("sigma1",0.4,0,5);
    Variable * sigma2 = new Variable("sigma2",0.6,0,5);
    Variable * expo_par = new Variable("exp_par",-5,-10,10);

    GaussianPdf gaus1("gaus1",xvar,mean1,sigma1);
    GaussianPdf gaus2("gaus2",xvar,mean2,sigma2);
    ExpPdf exppdf("exppdf",xvar,expo_par);

    vector<PdfBase*> pdfs;
    pdfs.push_back(&exppdf);
    pdfs.push_back(&gaus1);
    pdfs.push_back(&gaus2);

    vars.clear();
    Variable * weight_exp = new Variable("we_exp",0.9,0.0,1.);
    Variable * weight_gaus = new Variable("we_gaus",0.04,0.0,1.);
    vars.push_back(weight_exp);
    vars.push_back(weight_gaus);
    // Creating the sum PDF.
    AddPdf total("total", vars, pdfs);
    
    total.setData(&data);
    FitManager fitter(&total);
    fitter.fit();
    fitter.getMinuitValues();

    // Drawing the results
    // First creating a grid where the PDFs will be evaluated.
    UnbinnedDataSet grid(xvar);
    double step = (xvar->upperlimit - xvar->lowerlimit)/xvar->numbins;
    for (int i = 0; i < xvar->numbins;i++){
        xvar->value = xvar->lowerlimit + (i+0.5)*step;
        grid.addEvent();
    }
    total.setData(&grid);
    vector<vector<double> > pdfVals;
    total.getCompProbsAtDataPoints(pdfVals);
    // Filling histograms of the total PDF and the components.
    TH1F pdfhist("pdfhist","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    TH1F exphist("exphist","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    TH1F g1hist("g1hist","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    TH1F g2hist("g2hist","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    pdfhist.SetStats(false);
    exphist.SetStats(false);
    g1hist.SetStats(false);
    g2hist.SetStats(false);

    double totalPdf = 0;
    for (int i=0;i<grid.getNumEvents();i++){
        grid.loadEvent(i);
        pdfhist.Fill(xvar->value,pdfVals[0][i]);
        exphist.Fill(xvar->value,pdfVals[1][i]);
        g1hist.Fill(xvar->value,pdfVals[2][i]);
        g2hist.Fill(xvar->value,pdfVals[3][i]);
        totalPdf += pdfVals[0][i];
    }
    // Normalizing all the PDFs to the data.
    for (int i=0;i<xvar->numbins;i++){
        double val = pdfhist.GetBinContent(i+1);
        val /= totalPdf;
        val *= totalData;
        pdfhist.SetBinContent(i+1,val);
        val = exphist.GetBinContent(i+1);
        val /= totalPdf;
        val *= totalData;
        val *= weight_exp->value;
        exphist.SetBinContent(i+1,val);
        val = g1hist.GetBinContent(i+1);
        val /= totalPdf;
        val *= totalData;
        val *= weight_gaus->value;
        g1hist.SetBinContent(i+1,val);
        val = g2hist.GetBinContent(i+1);
        val /= totalPdf;
        val *= totalData;
        val *= (1-weight_exp->value-weight_gaus->value);
        g2hist.SetBinContent(i+1,val);
    }
    // Plotting
    TCanvas foo;
    xvarhist.SetMarkerStyle(8);
    xvarhist.SetMarkerSize(0.5);
    xvarhist.Draw("p");
    pdfhist.SetLineColor(kBlue);
    pdfhist.SetLineWidth(3);
    pdfhist.Draw("lsame");
    exphist.SetLineColor(kRed);
    exphist.SetLineStyle(kDashed);
    exphist.SetLineWidth(3);
    exphist.Draw("lsame");
    g1hist.SetLineColor(kBlue);
    g2hist.SetLineColor(kBlue);
    g1hist.SetLineWidth(3);
    g2hist.SetLineWidth(3);
    g1hist.Draw("lsame");
    g2hist.Draw("lsame");
    foo.SaveAs("xhist.png");
    foo.SaveAs("xhist.pdf");
    
    return 0;
}
