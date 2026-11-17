/*
** Copyright (c) 2026, Robert Kausch <robert.kausch@freac.org>
** All rights reserved.
**
** This code is released under 2-clause BSD license. Please see the
** file at : https://github.com/libsndfile/libsamplerate/blob/master/COPYING
*/

/*
**	RISC-V vector extension version of the inner loops of the sinc converters
**	(see src_sinc.h).
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <assert.h>
#include <math.h>

#include "common.h"
#include "src_sinc.h"

#if defined (__riscv_v) && defined (__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000

#include <riscv_vector.h>

/*
**	The loops are vector length agnostic : each iteration handles as many
**	elements as fit into the vector registers of the CPU. Single precision data
**	is kept in LMUL 1 and double precision data in LMUL 2 registers, so that
**	both hold the same number of elements. No surplus coefficients get written.
*/

/* Keep the compiler from turning a loop into vector code by itself. */
#if defined (__clang__)
#	define SINC_SCALAR_LOOP	_Pragma ("clang loop vectorize(disable) interleave(disable)")
#elif defined (__GNUC__) && __GNUC__ >= 14
#	define SINC_SCALAR_LOOP	_Pragma ("GCC novector")
#else
#	define SINC_SCALAR_LOOP
#endif

/*
**	This one is deliberately not done on vectors. Fetching the polynomial
**	coefficients for many positions at a time takes indexed loads, which turned
**	out slower than this plain loop (on a SpacemiT X100).
*/
static inline void
sinc_build_coeffs_rvv (const coeff_t *table, int64_t pos, int64_t delta, int count, double *out)
{
	SINC_SCALAR_LOOP
	for (int k = 0 ; k < count ; k++)
	{	const coeff_t *c = table + 4 * (int) (pos >> 32) ;
		const float x = (float) (((uint32_t) pos) >> 1) * (1.0f / 2147483648.0f) ;

		out [k] = c [0] + x * (c [1] + x * (c [2] + x * c [3])) ;

		pos += delta ;
		} ;
} /* sinc_build_coeffs_rvv */

static inline void
sinc_build_wing_forward_rvv (const double *rows, int row_len, const double *weights, int count, double *out)
{
	for (int k = 0 ; k < count ; )
	{	const size_t vl = __riscv_vsetvl_e64m2 ((size_t) (count - k)) ;
		vfloat64m2_t a ;

		a = __riscv_vfmul_vf_f64m2 (__riscv_vle64_v_f64m2 (rows + k, vl), weights [0], vl) ;
		a = __riscv_vfmacc_vf_f64m2 (a, weights [1], __riscv_vle64_v_f64m2 (rows + row_len + k, vl), vl) ;
		a = __riscv_vfmacc_vf_f64m2 (a, weights [2], __riscv_vle64_v_f64m2 (rows + 2 * row_len + k, vl), vl) ;
		a = __riscv_vfmacc_vf_f64m2 (a, weights [3], __riscv_vle64_v_f64m2 (rows + 3 * row_len + k, vl), vl) ;
		__riscv_vse64_v_f64m2 (out + k, a, vl) ;

		k += (int) vl ;
		} ;
} /* sinc_build_wing_forward_rvv */

static inline void
sinc_build_wing_reverse_rvv (const double *rows, int row_len, const double *weights, int count, double *out)
{
	for (int k = 0 ; k < count ; )
	{	const size_t vl = __riscv_vsetvl_e64m2 ((size_t) (count - k)) ;
		vfloat64m2_t a ;

		a = __riscv_vfmul_vf_f64m2 (__riscv_vle64_v_f64m2 (rows + k, vl), weights [0], vl) ;
		a = __riscv_vfmacc_vf_f64m2 (a, weights [1], __riscv_vle64_v_f64m2 (rows + row_len + k, vl), vl) ;
		a = __riscv_vfmacc_vf_f64m2 (a, weights [2], __riscv_vle64_v_f64m2 (rows + 2 * row_len + k, vl), vl) ;
		a = __riscv_vfmacc_vf_f64m2 (a, weights [3], __riscv_vle64_v_f64m2 (rows + 3 * row_len + k, vl), vl) ;

		/* Store them back to front. */
		__riscv_vsse64_v_f64m2 (out - k, - (ptrdiff_t) sizeof (double), a, vl) ;

		k += (int) vl ;
		} ;
} /* sinc_build_wing_reverse_rvv */

/* The sum of all elements of a vector. */
static inline double
sinc_sum_rvv (vfloat64m2_t values)
{	return __riscv_vfmv_f_s_f64m1_f64 (__riscv_vfredusum_vs_f64m2_f64m1 (values, __riscv_vfmv_s_f_f64m1 (0.0, 1), __riscv_vsetvlmax_e64m2 ())) ;
} /* sinc_sum_rvv */

static inline void
sinc_dot_rvv (const double *coeffs, const float *data, int count, int channels, double *acc)
{	const size_t vlmax = __riscv_vsetvlmax_e64m2 () ;

	if (channels == 1)
	{	vfloat64m2_t s0 = __riscv_vfmv_v_f_f64m2 (0.0, vlmax), s1 = s0 ;
		int k = 0 ;

		/* Two partial sums to keep the dependency chains short. */
		for ( ; k + 2 * (int) vlmax <= count ; k += 2 * (int) vlmax)
		{	s0 = __riscv_vfmacc_vv_f64m2 (s0, __riscv_vle64_v_f64m2 (coeffs + k, vlmax),
					__riscv_vfwcvt_f_f_v_f64m2 (__riscv_vle32_v_f32m1 (data + k, vlmax), vlmax), vlmax) ;
			s1 = __riscv_vfmacc_vv_f64m2 (s1, __riscv_vle64_v_f64m2 (coeffs + k + vlmax, vlmax),
					__riscv_vfwcvt_f_f_v_f64m2 (__riscv_vle32_v_f32m1 (data + k + vlmax, vlmax), vlmax), vlmax) ;
			} ;

		while (k < count)
		{	const size_t vl = __riscv_vsetvl_e64m2 ((size_t) (count - k)) ;

			/* Elements beyond vl keep their value. */
			s0 = __riscv_vfmacc_vv_f64m2_tu (s0, __riscv_vle64_v_f64m2 (coeffs + k, vl),
					__riscv_vfwcvt_f_f_v_f64m2 (__riscv_vle32_v_f32m1 (data + k, vl), vl), vl) ;
			k += (int) vl ;
			} ;

		acc [0] += sinc_sum_rvv (__riscv_vfadd_vv_f64m2 (s0, s1, vlmax)) ;
		}
	else if (channels == 2)
	{	vfloat64m2_t left = __riscv_vfmv_v_f_f64m2 (0.0, vlmax), right = left ;

		for (int k = 0 ; k < count ; )
		{	const size_t vl = __riscv_vsetvl_e64m2 ((size_t) (count - k)) ;
			const vfloat64m2_t c = __riscv_vle64_v_f64m2 (coeffs + k, vl) ;
			const vfloat32m1x2_t frames = __riscv_vlseg2e32_v_f32m1x2 (data + 2 * k, vl) ;

			left = __riscv_vfmacc_vv_f64m2_tu (left, c, __riscv_vfwcvt_f_f_v_f64m2 (__riscv_vget_v_f32m1x2_f32m1 (frames, 0), vl), vl) ;
			right = __riscv_vfmacc_vv_f64m2_tu (right, c, __riscv_vfwcvt_f_f_v_f64m2 (__riscv_vget_v_f32m1x2_f32m1 (frames, 1), vl), vl) ;
			k += (int) vl ;
			} ;

		acc [0] += sinc_sum_rvv (left) ;
		acc [1] += sinc_sum_rvv (right) ;
		}
	else
	{	const ptrdiff_t stride = channels * (ptrdiff_t) sizeof (float) ;

		/* One channel after the other. */
		for (int ch = 0 ; ch < channels ; ch++)
		{	vfloat64m2_t sum = __riscv_vfmv_v_f_f64m2 (0.0, vlmax) ;

			for (int k = 0 ; k < count ; )
			{	const size_t vl = __riscv_vsetvl_e64m2 ((size_t) (count - k)) ;

				sum = __riscv_vfmacc_vv_f64m2_tu (sum, __riscv_vle64_v_f64m2 (coeffs + k, vl),
						__riscv_vfwcvt_f_f_v_f64m2 (__riscv_vlse32_v_f32m1 (data + k * channels + ch, stride, vl), vl), vl) ;
				k += (int) vl ;
				} ;

			acc [ch] += sinc_sum_rvv (sum) ;
			} ;
		} ;
} /* sinc_dot_rvv */

static const SINC_KERNEL sinc_kernel_rvv =
{	sinc_build_coeffs_rvv,
	sinc_build_wing_forward_rvv,
	sinc_build_wing_reverse_rvv,
	sinc_dot_rvv
} ;

static void
calc_output_rvv (SINC_FILTER *filter, int channels, double input_index, double scale, float *output)
{	sinc_calc_output (&sinc_kernel_rvv, filter, channels, input_index, scale, output) ;
} /* calc_output_rvv */

LIBSAMPLERATE_DLL_PRIVATE const sinc_calc_output_t sinc_calc_output_rvv = calc_output_rvv ;

#else

/* Not built for the RISC-V vector extension (see src_sinc.h). */
LIBSAMPLERATE_DLL_PRIVATE const sinc_calc_output_t sinc_calc_output_rvv = NULL ;

#endif
