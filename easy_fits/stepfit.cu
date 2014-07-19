/*
   Investigating the usage of the step function as a map function for mapped
   PDF.
   So far it seems that there might be some trouble with this, because of
   normalisation. In AddPdf all components are normalized separately. It is
   possible that this won't be a problem if map pdf doesn't normalize the
   function it is given. Then the weights would be equal for all steps.
   Requires more testing. At the moment all weights are chosen such that the PDF
   has the proper shape.

   Run with `./stepfit` or `./stepfit number` where number is an integer (can't
   be much larger than a couple hundred).
*/

#include "Variable.hh"
#include "FitManager.hh"
#include "UnbinnedDataSet.hh"
#include "StepPdf.hh"
#include "AddPdf.hh"

#include "TCanvas.h"
#include "TH1F.h"


int main(int argc, char **argv){
    // Creating grid
    Variable *xvar = new Variable("xvar",0,30);
    UnbinnedDataSet grid(xvar);
    for (int i=0;i<xvar->numbins;i++){
        double step = xvar->upperlimit - xvar->lowerlimit;
        step /= xvar->numbins;
        xvar->value = xvar->lowerlimit+(i+0.5)*step;
        grid.addEvent();
    }
    // Amount of steps to create
    int N;
    if (argc == 2)
        N = atoi(argv[1]);
    else
        N = 20;
    // Creating step points
    Variable **step = new Variable*[N];
    for (int i=0;i<N;i++){
        step[i] = new Variable("offset",(i+1.)/(N+1)*xvar->upperlimit);
    }
    // Creating step functions
    StepPdf **funcs = new StepPdf*[N];
    for (int i=0;i<N;i++){
        funcs[i] = new StepPdf("step",xvar,step[i]);
    }
    vector<PdfBase*> comps(funcs,funcs+N);
    // Creating step weights
    Variable **weight = new Variable*[N];
    for (int i=0;i<N;i++){
        weight[i] = new Variable("weight",N-i);
    }
    vector<Variable*> weights(weight,weight+N);
    // Creating sum of steps
    AddPdf *total = new AddPdf("sum",weights,comps);
    total->setData(&grid);
    // Getting pdf vals at grid points
    vector<vector<double> > pdfvals;
    total->getCompProbsAtDataPoints(pdfvals);
    // Histogramming the grid points
    TH1F pdfhist("pdfhist","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    /*
    TH1F p1("p1","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    TH1F p2("p2","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    TH1F p3("p3","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    */
    pdfhist.SetStats(false);
    for (int i=0;i<grid.getNumEvents();i++){
        grid.loadEvent(i);
        pdfhist.Fill(xvar->value,pdfvals[0][i]);
        /*
        p1.Fill(xvar->value,pdfvals[1][i]);
        p2.Fill(xvar->value,pdfvals[2][i]);
        p3.Fill(xvar->value,pdfvals[3][i]);
        */
    }
    // Plotting
    TCanvas foo;
    pdfhist.SetLineColor(kBlue);
    pdfhist.SetLineWidth(3);
    pdfhist.Draw("l");
    /*
    p1.Draw("lsame");
    p2.Draw("lsame");
    p3.Draw("lsame");
    */
    foo.SaveAs("step.pdf");
    std::cout << "Histogram integral: " << pdfhist.Integral("width") << std::endl;
    return 0;
}
