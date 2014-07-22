#include "Variable.hh"
#include "UnbinnedDataSet.hh"
#include "FitManager.hh"
#include "ParabolaPdf.hh"

// plotting with ROOT
#include "TRandom.h"
#include "TCanvas.h"
#include "TH1F.h"
int main(void)
{
    Variable *xvar = new Variable("xvar",0,4);
    UnbinnedDataSet data(xvar);

    TRandom gen(42);
    TH1F xhist("xhist","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    xhist.SetStats(false);

    for (int i=0;i<10000;i++)
    {
        float temp = gen.Uniform(4);
        xvar->value = gen.Uniform(4);
        if (temp > -1*pow(xvar->value-2,2) + 4)
        {
            i--;
            continue;
        }
        else
        {
            data.addEvent();
            xhist.Fill(xvar->value);
        }
    }

    Variable *a=new Variable("a",-0.5,-2,0);
    Variable *p=new Variable("p",1,0,4);
    Variable *q=new Variable("q",1,0,10);
    ParabolaPdf *pdf = new ParabolaPdf("parabola",xvar,a,p,q);
    pdf->setData(&data);
    FitManager fitter(pdf);
    fitter.fit();
    fitter.getMinuitValues();

    // Generating a grid to evaluate PDFs on
    TH1F pdfhist("pdfhist","",xvar->numbins,xvar->lowerlimit,xvar->upperlimit);
    pdfhist.SetStats(false);
    UnbinnedDataSet grid(xvar);
    for (int i=0;i<xvar->numbins;i++)
    {
        float step = (xvar->upperlimit-xvar->lowerlimit);
        step /= xvar->numbins;
        xvar->value = step*(i+0.5) + xvar->lowerlimit;
        grid.addEvent();
    }
    vector <vector<double> > pdfVals;
    pdf->setData(&grid);
    pdf->getCompProbsAtDataPoints(pdfVals);
    for (int i=0;i<grid.getNumEvents();i++)
    {
        grid.loadEvent(i);
        pdfhist.Fill(xvar->value,pdfVals[0][i]);
    }
    
    float x_total = xhist.Integral();
    float pdf_total = pdfhist.Integral();
    
    pdfhist.Scale(x_total/pdf_total);
    TCanvas foo;
    xhist.SetMarkerStyle(8);
    xhist.SetMarkerSize(0.5);
    xhist.Draw("p");
    pdfhist.SetLineColor(kBlue);
    pdfhist.SetLineWidth(3);
    pdfhist.Draw("lsame");
    foo.SaveAs("example.png");
    

    return 0;
}
