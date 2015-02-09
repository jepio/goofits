cuda_add_executable(own_pdf_fit own_pdf_fit.cu BifPdf.cu 
    NovoPdf.cu LanPdf.cu ${PDF_LIB})
link_goofit(own_pdf_fit)

