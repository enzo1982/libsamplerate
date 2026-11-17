# Generate the coefficient tables of the three sinc converters.
#
# Run this script from the directory it is in. It writes the header files
# fastest_coeffs.h, mid_qual_coeffs.h and high_qual_coeffs.h in ../src.
#
# The parameters below describe the filters libsamplerate has been using all
# along. They reproduce the coefficients of the former (much bigger) tables for
# linear interpolation to within 1e-13.
#
#                                         half    window       fudge            stop band
#                  name       increment  length  half length   factor           atten.

make_sinc_table ("fastest_coeffs",        128,   2462,   2462.0, 1.2026882171631, 100.3, "../src/fastest_coeffs.h") ;
make_sinc_table ("slow_mid_qual_coeffs",  491,  22436,  22437.5, 1.0880645960569, 122.0, "../src/mid_qual_coeffs.h") ;
make_sinc_table ("slow_high_qual_coeffs", 2381, 340237, 340238.5, 1.0354878008366, 160.0, "../src/high_qual_coeffs.h") ;
