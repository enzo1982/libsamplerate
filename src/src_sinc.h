/*
** Copyright (c) 2026, Robert Kausch <robert.kausch@freac.org>
** All rights reserved.
**
** This code is released under 2-clause BSD license. Please see the
** file at : https://github.com/libsndfile/libsamplerate/blob/master/COPYING
*/

/*
**	Things shared by src_sinc.c and the files with the SIMD versions of its
**	inner loops (src_sinc_sse2.c, src_sinc_avx2.c, src_sinc_neon.c and
**	src_sinc_rvv.c).
*/

#ifndef SRC_SINC_H_INCLUDED
#define SRC_SINC_H_INCLUDED

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include "common.h"

/* Number of filter coefficients generated and applied in one go (multiple of 8). */
#define	COEFF_CHUNK				512

/*
** Besides the portable C version of the inner loops in src_sinc.c there are
** SIMD versions in files of their own : src_sinc_sse2.c and src_sinc_avx2.c
** for x86, src_sinc_neon.c for 64 bit ARM and src_sinc_rvv.c for the RISC-V
** vector extension. The build system compiles each of them for its instruction
** set, if the compiler supports that. Which one gets used is decided at run
** time (see sinc_choose_calc_output in src_sinc.c).
*/

#if defined (__GNUC__) || defined (__clang__)
#	define SINC_ALWAYS_INLINE	inline __attribute__ ((always_inline))
#elif defined (_MSC_VER)
#	define SINC_ALWAYS_INLINE	__forceinline
#else
#	define SINC_ALWAYS_INLINE	inline
#endif

/* Customixe max channls from Kconfig. */
#ifndef CONFIG_CHAN_NR
#define MAX_CHANNELS			128
#else
#define MAX_CHANNELS 			CONFIG_CHAN_NR
#endif

typedef float	coeff_t ;

typedef struct SINC_FILTER_tag SINC_FILTER ;

/* Function calculating one output frame (see sinc_calc_output). */
typedef void (*sinc_calc_output_t) (SINC_FILTER *filter, int channels, double input_index, double scale, float *output) ;

struct SINC_FILTER_tag
{	int		sinc_magic_marker ;

	long	in_count, in_used ;
	long	out_count, out_gen ;

	int		coeff_half_len, index_inc ;

	/* Filter half length in input samples (at ratios >= 1.0). */
	double	half_len ;

	double	src_ratio, input_index ;

	/* The coefficient tables and their layout (see Octave/make_sinc_table.m). */
	coeff_t const	*cubic ;
	double const	*poly ;
	int		resolution, cubic_front, poly_row_len ;

	int		b_current, b_end, b_real_end, b_len ;

	/* Sure hope noone does more than 128 channels at once. */
	double acc [MAX_CHANNELS] ;

	float	*buffer ;

	/* The version of the calculation chosen for the CPU we are running on. */
	sinc_calc_output_t calc_output ;
} ;

/*
**	Coefficient generation and application.
**
**	The filter coefficients for one output sample are generated on demand from
**	the coefficient tables using 4 point (cubic) Lagrange interpolation and are
**	then applied to the input data as a plain FIR filter.
**
**	This is built from four primitives which exist in a portable C version (in
**	src_sinc.c), in SSE2 and AVX2 versions for x86 CPUs, in a NEON version for
**	64 bit ARM CPUs and in a version for the RISC-V vector extension.
*/

typedef struct
{	/*
	**	Generate count coefficients for table positions pos, pos + delta, ...
	**	The positions are (non negative) 32.32 fixed point numbers in units of
	**	table intervals and table is the one of the polynomial coefficients.
	**	This may write (and read the table for) up to seven surplus coefficients.
	*/
	void (*build_coeffs) (const coeff_t *table, int64_t pos, int64_t delta, int count, double *out) ;

	/*
	**	Generate the coefficients of one filter wing from four consecutive rows
	**	of the polyphase table. The forward version stores them in out [0],
	**	out [1], ... and the reverse version in out [0], out [-1], ...
	**	Both may write up to three surplus coefficients.
	*/
	void (*build_wing_forward) (const double *rows, int row_len, const double *weights, int count, double *out) ;
	void (*build_wing_reverse) (const double *rows, int row_len, const double *weights, int count, double *out) ;

	/*
	**	Apply count (<= COEFF_CHUNK) coefficients to the interleaved input data
	**	and add the results to acc.
	*/
	void (*dot) (const double *coeffs, const float *data, int count, int channels, double *acc) ;
} SINC_KERNEL ;

/* Lagrange weights for the table entries at -1, 0, 1 and 2 around fraction x. */
static inline void
sinc_lagrange_weights (double x, double *weights)
{	const double a = x * (x - 1.0) / 6.0, b = (x + 1.0) * (x - 2.0) / 2.0 ;

	weights [0] = -a * (x - 2.0) ;
	weights [1] = b * (x - 1.0) ;
	weights [2] = -b * x ;
	weights [3] = a * (x + 1.0) ;
} /* sinc_lagrange_weights */

/*
**	The code common to all versions. These functions get inlined into the
**	sinc_calc_output_* functions, where the kernel is a compile time constant.
*/

/*
**	Apply one wing of a stretched filter (used when downsampling). The first
**	of the count input frames belongs to table position pos (in table entries),
**	each further one is delta entries further on.
*/
static SINC_ALWAYS_INLINE void
sinc_apply_wing (const SINC_KERNEL *kernel, const SINC_FILTER *filter, int channels, double pos, double delta, int count, const float *data, double *acc)
{	double coeffs [COEFF_CHUNK] ;
	int64_t fp_pos, fp_delta ;

	fp_pos = (int64_t) ((pos + filter->cubic_front) * 4294967296.0 + 0.5) ;
	fp_delta = (int64_t) (fabs (delta) * 4294967296.0 + 0.5) ;
	if (delta < 0.0)
		fp_delta = -fp_delta ;

	while (count > 0)
	{	int chunk = MIN (count, COEFF_CHUNK) ;

		kernel->build_coeffs (filter->cubic, fp_pos, fp_delta, chunk, coeffs) ;
		kernel->dot (coeffs, data, chunk, channels, acc) ;

		fp_pos += chunk * fp_delta ;
		data += chunk * channels ;
		count -= chunk ;
		} ;
} /* sinc_apply_wing */

/*
**	Calculate one output frame. input_index is the fractional position of the
**	output sample behind filter->b_current and scale is MIN (src_ratio, 1.0),
**	the factor by which the filter is stretched when downsampling.
*/
static SINC_ALWAYS_INLINE void
sinc_calc_output (const SINC_KERNEL *kernel, SINC_FILTER *filter, int channels, double input_index, double scale, float *output)
{	double		acc_small [8], *acc ;
	const float	*data ;
	double		half_len ;
	int			left, right ;

	acc = (channels <= ARRAY_LEN (acc_small)) ? acc_small : filter->acc ;
	for (int ch = 0 ; ch < channels ; ch++)
		acc [ch] = 0.0 ;

	/*
	** Number of input frames under each wing of the (stretched) filter, not
	** counting the frame at b_current (left wing) and the one after (right wing).
	*/
	half_len = filter->half_len / scale ;
	left = (int) (half_len - input_index) ;
	right = (int) (half_len - (1.0 - input_index)) ;

	/* Avoid accessing filter->buffer outside of the valid data. */
	left = MIN (left, filter->b_current / channels) ;
	right = MIN (right, (filter->b_end - filter->b_current) / channels - 2) ;
	right = MAX (right, -1) ;

	data = filter->buffer + filter->b_current - channels * left ;

	assert (data >= filter->buffer) ;
	assert (data + channels * (left + right + 2) <= filter->buffer + filter->b_end) ;

	if (scale == 1.0 && left + right + 2 <= COEFF_CHUNK)
	{	/*
		** Not downsampling. Consecutive taps of a wing are exactly resolution
		** table entries apart, so they all share the same interpolation
		** weights and are found next to each other in the polyphase table.
		** This is done in double precision all the way.
		*/
		double coeff_buf [COEFF_CHUNK + 8], *coeffs = coeff_buf + 4 ;
		double weights [4] ;
		double pos ;
		int row ;

		pos = input_index * filter->resolution ;
		row = (int) pos ;
		sinc_lagrange_weights (pos - row, weights) ;
		kernel->build_wing_reverse (filter->poly + row * filter->poly_row_len, filter->poly_row_len, weights, left + 1, coeffs + left) ;

		pos = (1.0 - input_index) * filter->resolution ;
		row = (int) pos ;
		sinc_lagrange_weights (pos - row, weights) ;
		kernel->build_wing_forward (filter->poly + row * filter->poly_row_len, filter->poly_row_len, weights, right + 1, coeffs + left + 1) ;

		kernel->dot (coeffs, data, left + right + 2, channels, acc) ;
		}
	else
	{	const double step = scale * filter->resolution ;

		sinc_apply_wing (kernel, filter, channels, (input_index + left) * step, -step, left + 1, data, acc) ;
		sinc_apply_wing (kernel, filter, channels, (1.0 - input_index) * step, step, right + 1, data + channels * (left + 1), acc) ;
		} ;

	for (int ch = 0 ; ch < channels ; ch++)
		output [ch] = (float) (scale * acc [ch]) ;
} /* sinc_calc_output */

/*
**	The versions of sinc_calc_output for the SIMD kernels. Each one is NULL if
**	its file was not built for the respective instruction set.
*/
extern const sinc_calc_output_t sinc_calc_output_sse2 ;
extern const sinc_calc_output_t sinc_calc_output_avx2 ;
extern const sinc_calc_output_t sinc_calc_output_neon ;
extern const sinc_calc_output_t sinc_calc_output_rvv ;

#endif	/* SRC_SINC_H_INCLUDED */
