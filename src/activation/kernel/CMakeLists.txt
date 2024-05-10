file(GLOB kernelfile "*.cu")

add_library(kernel STATIC ${kernelfile})