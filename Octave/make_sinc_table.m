function make_sinc_table (name, increment, half_len, window_half_len, fudge_factor, atten, filename, resolution)

# Write the coefficient tables for one of the sinc converters as a C header
# file for src/src_sinc.c.
#
# The filter is a Kaiser windowed sinc function :
#
#   h (x) = sinc (x / fudge_factor / increment) * kaiser (x / window_half_len)
#
# where x is the distance from the centre of the filter in units of 1/increment
# input samples and the Kaiser window has a beta of (atten + 0.5) / 10. It is
# normalised to unity gain. These are the same parameters as used by
# generate_filter.m and make_filter.m which design a filter on a grid of
# increment points per input sample.
#
#   name            : Name of the C struct, e.g. "fastest_coeffs".
#   increment       : Increment the filter was designed with.
#   half_len        : Filter half length in increments. The filter is taken
#                     to be zero from half_len + 1 on.
#   window_half_len : Half length of the Kaiser window in increments.
#   fudge_factor    : Factor by which the sinc function is stretched.
#   atten           : Stop band attenuation the Kaiser window is made for.
#   filename        : Name of the header file to write.
#   resolution      : Points per input sample at which the filter is sampled
#                     for the tables (optional, default 64).
#
# The converters evaluate the filter between these points using 4 point
# Lagrange interpolation, i.e. the cubic polynomial through the two points on
# either side. With 64 points per input sample the interpolation error is below
# the precision of the single precision coefficients.
#
# The header contains the filter in two forms :
#
#   cubic : For each interval between two points the coefficients c0 ... c3 of
#           the polynomial used in it, so that the filter value at a fraction x
#           of the interval is c0 + x * (c1 + x * (c2 + x * c3)). They are
#           stored at [4 * (cubic_front + k)] for the interval from k to k + 1
#           points from the centre. There are cubic_front intervals in front of
#           the centre and some zero padding at the end, both to allow for SIMD
#           code calculating a few values more than needed. This table is used
#           when downsampling and is in single precision.
#
#   poly  : A polyphase table of resolution + 4 rows of poly_row_len entries.
#           Row r holds the filter values (r - 1) / resolution + j input
#           samples from the centre for j = 0, 1, 2, ... This puts the values
#           needed when not downsampling next to each other. This table is in
#           double precision.
#
# The tables are aligned (using the macro SINC_TABLE_ALIGN defined in
# src/src_sinc.c) so that the four coefficients of an interval of the cubic
# table and four consecutive entries of a row of the polyphase table never
# straddle a cache line.

if nargin < 7,
	error ('Try make_sinc_table ("fastest_coeffs", 128, 2462, 2462, 1.2026882171631, 100.3, "fastest_coeffs.h")') ;
	endif

if nargin < 8,
	resolution = 64 ;
	endif

if rem (resolution, 4) != 0,
	error ("The resolution must be a multiple of 4.") ;
	endif

beta = (atten + 0.5) / 10 ;

#-------------------------------------------------------------------------------
# Normalise the filter to unity gain on the grid it was designed on.

gain = increment / sum (filter_value (-half_len:half_len, increment, window_half_len, fudge_factor, beta)) ;

#-------------------------------------------------------------------------------
# The table of polynomial coefficients.

cubic_front = 8 * resolution ;
cubic_pad = 8 * resolution + 8 ;
cubic_len = cubic_front + fix ((half_len + 1) * resolution / increment) + 3 + cubic_pad ;

# The four points around each interval.
k = (0:cubic_len - 1) - cubic_front ;
y0 = table_value (k - 1, resolution, gain, increment, half_len, window_half_len, fudge_factor, beta) ;
y1 = table_value (k, resolution, gain, increment, half_len, window_half_len, fudge_factor, beta) ;
y2 = table_value (k + 1, resolution, gain, increment, half_len, window_half_len, fudge_factor, beta) ;
y3 = table_value (k + 2, resolution, gain, increment, half_len, window_half_len, fudge_factor, beta) ;

# The polynomial through them, with y1 at x = 0 and y2 at x = 1.
c3 = (y3 - y0) / 6 + (y1 - y2) / 2 ;
c2 = (y0 + y2) / 2 - y1 ;
c1 = y2 - y1 - c2 - c3 ;

# Interval after interval.
cubic = reshape ([y1 ; c1 ; c2 ; c3], 1, []) ;

#-------------------------------------------------------------------------------
# The polyphase table.

poly_rows = resolution + 4 ;
poly_row_len = 4 * fix ((fix (half_len / increment) + 7) / 4) ;

[col, row] = meshgrid (0:poly_row_len - 1, 0:poly_rows - 1) ;
poly = table_value (col * resolution + row - 1, resolution, gain, increment, half_len, window_half_len, fudge_factor, beta) ;

# Row after row.
poly = reshape (poly', 1, []) ;

#-------------------------------------------------------------------------------
# Write the header file.

file = fopen (filename, "w") ;
if file < 0,
	error (sprintf ("Error, not able to open '%s'", filename)) ;
	endif

fprintf (file, "/*\n") ;
fprintf (file, "** Copyright (c) 2026, Robert Kausch <robert.kausch@freac.org>\n") ;
fprintf (file, "** All rights reserved.\n") ;
fprintf (file, "**\n") ;
fprintf (file, "** This code is released under 2-clause BSD license. Please see the\n") ;
fprintf (file, "** file at : https://github.com/libsndfile/libsamplerate/blob/master/COPYING\n") ;
fprintf (file, "*/\n\n") ;

fprintf (file, "/*\n") ;
fprintf (file, "** Generated by Octave/make_sinc_table.m, do not edit.\n") ;
fprintf (file, "**\n") ;
fprintf (file, "** make_sinc_table (\"%s\", %d, %d, %.1f, %.13f, %.1f, \"%s\", %d) ;\n", ...
			name, increment, half_len, window_half_len, fudge_factor, atten, filename_tail (filename), resolution) ;
fprintf (file, "**\n") ;
fprintf (file, "**   increment        : %d\n", increment) ;
fprintf (file, "**   half length      : %d\n", half_len) ;
fprintf (file, "**   resolution       : %d\n", resolution) ;
fprintf (file, "**   cubic table      : %d intervals, %d in front of the centre\n", cubic_len, cubic_front) ;
fprintf (file, "**   polyphase table  : %d rows of %d entries\n", poly_rows, poly_row_len) ;
fprintf (file, "*/\n\n") ;

fprintf (file, "#ifdef _MSC_VER\n#pragma warning (push)\n#pragma warning (disable: 4305)\n#endif\n\n") ;

fprintf (file, "static const struct %s_s\n", name) ;
fprintf (file, "{\tint increment ;\n") ;
fprintf (file, "\tint half_len ;\n") ;
fprintf (file, "\tint resolution ;\n") ;
fprintf (file, "\tint cubic_front ;\n") ;
fprintf (file, "\tint poly_row_len ;\n") ;
fprintf (file, "\tSINC_TABLE_ALIGN (16) coeff_t cubic [%d] ;\n", length (cubic)) ;
fprintf (file, "\tSINC_TABLE_ALIGN (32) double poly [%d] ;\n", length (poly)) ;
fprintf (file, "} %s =\n", name) ;
fprintf (file, "{\t%d,\n\t%d,\n\t%d,\n\t%d,\n\t%d,\n", increment, half_len, resolution, cubic_front, poly_row_len) ;

fprintf (file, "{\t/* cubic */\n") ;
fprintf (file, "% 24.20e,\n", cubic) ;
fprintf (file, "},\n") ;

fprintf (file, "{\t/* poly */\n") ;
fprintf (file, "% 24.20e,\n", poly) ;
fprintf (file, "}\n") ;

fprintf (file, "} ; /* %s */\n\n", name) ;

fprintf (file, "#ifdef _MSC_VER\n#pragma warning (pop)\n#endif\n") ;

fclose (file) ;

printf ("%-24s : %6d + %6d coefficients, gain %.17g\n", filename_tail (filename), length (cubic), length (poly), gain) ;

endfunction

#-------------------------------------------------------------------------------
# The filter value x increments from the centre (without normalisation).

function f = filter_value (x, increment, window_half_len, fudge_factor, beta)

w = besseli (0, beta * sqrt (max (1 - (x / window_half_len) .^ 2, 0))) / besseli (0, beta) ;
w (abs (x) > window_half_len) = 0 ;

f = sinc (x / fudge_factor / increment) .* w ;

endfunction

#-------------------------------------------------------------------------------
# The table entry for position k (in units of 1/resolution input samples).

function f = table_value (k, resolution, gain, increment, half_len, window_half_len, fudge_factor, beta)

x = abs (k) * increment / resolution ;

f = gain * filter_value (x, increment, window_half_len, fudge_factor, beta) ;
f (x >= half_len + 1) = 0 ;

endfunction

#-------------------------------------------------------------------------------

function name = filename_tail (filename)

[dir, name, ext] = fileparts (filename) ;
name = [name ext] ;

endfunction
