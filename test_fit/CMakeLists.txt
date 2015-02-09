cuda_add_executable(my_fit my_fit.cu ParabolaPdf.cu ${PDF_OBJ})
link_goofit(my_fit)
