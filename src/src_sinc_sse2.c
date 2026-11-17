/*
** Copyright (c) 2026, Robert Kausch <robert.kausch@freac.org>
** All rights reserved.
**
** This code is released under 2-clause BSD license. Please see the
** file at : https://github.com/libsndfile/libsamplerate/blob/master/COPYING
*/

/*
**	SSE2 version of the inner loops of the sinc converters (see src_sinc.h).
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <assert.h>
#include <math.h>

#include "common.h"
#include "src_sinc.h"

#if (defined (__SSE2__) || defined (_M_X64) || (defined (_M_IX86_FP) && _M_IX86_FP >= 2)) && !defined (_M_ARM64EC)

#include <emmintrin.h>

/* Load two floats and convert them to double. */
static inline __m128d
sinc_load2_pd_sse2 (const float *ptr)
{	return _mm_cvtps_pd (_mm_castsi128_ps (_mm_loadl_epi64 ((const __m128i *) ptr))) ;
} /* sinc_load2_pd_sse2 */

static inline void
sinc_build_coeffs_sse2 (const coeff_t *table, int64_t pos, int64_t delta, int count, double *out)
{	const __m128 frac_scale = _mm_set1_ps (1.0f / 2147483648.0f) ;
	const int64_t pos1 = pos + delta, pos2 = pos + 2 * delta, pos3 = pos + 3 * delta, delta4 = 4 * delta ;
	const __m128i step = _mm_set_epi32 ((int) (delta4 >> 32), (int) delta4, (int) (delta4 >> 32), (int) delta4) ;
	__m128i p0 = _mm_set_epi32 ((int) (pos1 >> 32), (int) pos1, (int) (pos >> 32), (int) pos) ;
	__m128i p1 = _mm_set_epi32 ((int) (pos3 >> 32), (int) pos3, (int) (pos2 >> 32), (int) pos2) ;

	for (int k = 0 ; k < count ; k += 4)
	{	__m128 lo, hi, x, a, c0, c1, c2, c3 ;
		int indx [4] ;

		/* Split the positions into integer and fractional parts. */
		lo = _mm_shuffle_ps (_mm_castsi128_ps (p0), _mm_castsi128_ps (p1), _MM_SHUFFLE (2, 0, 2, 0)) ;
		hi = _mm_shuffle_ps (_mm_castsi128_ps (p0), _mm_castsi128_ps (p1), _MM_SHUFFLE (3, 1, 3, 1)) ;
		_mm_storeu_si128 ((__m128i *) indx, _mm_castps_si128 (hi)) ;
		x = _mm_mul_ps (_mm_cvtepi32_ps (_mm_srli_epi32 (_mm_castps_si128 (lo), 1)), frac_scale) ;

		/* Fetch the polynomial coefficients for each position. */
		c0 = _mm_loadu_ps (table + 4 * indx [0]) ;
		c1 = _mm_loadu_ps (table + 4 * indx [1]) ;
		c2 = _mm_loadu_ps (table + 4 * indx [2]) ;
		c3 = _mm_loadu_ps (table + 4 * indx [3]) ;
		_MM_TRANSPOSE4_PS (c0, c1, c2, c3) ;

		a = _mm_add_ps (_mm_mul_ps (c3, x), c2) ;
		a = _mm_add_ps (_mm_mul_ps (a, x), c1) ;
		a = _mm_add_ps (_mm_mul_ps (a, x), c0) ;

		_mm_storeu_pd (out + k, _mm_cvtps_pd (a)) ;
		_mm_storeu_pd (out + k + 2, _mm_cvtps_pd (_mm_movehl_ps (a, a))) ;

		p0 = _mm_add_epi64 (p0, step) ;
		p1 = _mm_add_epi64 (p1, step) ;
		} ;
} /* sinc_build_coeffs_sse2 */

static inline void
sinc_build_wing_forward_sse2 (const double *rows, int row_len, const double *weights, int count, double *out)
{	const __m128d w0 = _mm_set1_pd (weights [0]), w1 = _mm_set1_pd (weights [1]) ;
	const __m128d w2 = _mm_set1_pd (weights [2]), w3 = _mm_set1_pd (weights [3]) ;

	for (int k = 0 ; k < count ; k += 2)
	{	__m128d a, b ;

		a = _mm_add_pd (_mm_mul_pd (w0, _mm_loadu_pd (rows + k)), _mm_mul_pd (w1, _mm_loadu_pd (rows + row_len + k))) ;
		b = _mm_add_pd (_mm_mul_pd (w2, _mm_loadu_pd (rows + 2 * row_len + k)), _mm_mul_pd (w3, _mm_loadu_pd (rows + 3 * row_len + k))) ;
		_mm_storeu_pd (out + k, _mm_add_pd (a, b)) ;
		} ;
} /* sinc_build_wing_forward_sse2 */

static inline void
sinc_build_wing_reverse_sse2 (const double *rows, int row_len, const double *weights, int count, double *out)
{	const __m128d w0 = _mm_set1_pd (weights [0]), w1 = _mm_set1_pd (weights [1]) ;
	const __m128d w2 = _mm_set1_pd (weights [2]), w3 = _mm_set1_pd (weights [3]) ;

	for (int k = 0 ; k < count ; k += 2)
	{	__m128d a, b ;

		a = _mm_add_pd (_mm_mul_pd (w0, _mm_loadu_pd (rows + k)), _mm_mul_pd (w1, _mm_loadu_pd (rows + row_len + k))) ;
		b = _mm_add_pd (_mm_mul_pd (w2, _mm_loadu_pd (rows + 2 * row_len + k)), _mm_mul_pd (w3, _mm_loadu_pd (rows + 3 * row_len + k))) ;
		a = _mm_add_pd (a, b) ;
		_mm_storeu_pd (out - k - 1, _mm_shuffle_pd (a, a, 1)) ;
		} ;
} /* sinc_build_wing_reverse_sse2 */

static inline void
sinc_dot_mono_sse2 (const double *coeffs, const float *data, int count, double *acc)
{	__m128d s0, s1, s2, s3 ;
	double sum [2] ;
	int k ;

	s0 = s1 = s2 = s3 = _mm_setzero_pd () ;
	for (k = 0 ; k + 8 <= count ; k += 8)
	{	s0 = _mm_add_pd (s0, _mm_mul_pd (_mm_loadu_pd (coeffs + k), sinc_load2_pd_sse2 (data + k))) ;
		s1 = _mm_add_pd (s1, _mm_mul_pd (_mm_loadu_pd (coeffs + k + 2), sinc_load2_pd_sse2 (data + k + 2))) ;
		s2 = _mm_add_pd (s2, _mm_mul_pd (_mm_loadu_pd (coeffs + k + 4), sinc_load2_pd_sse2 (data + k + 4))) ;
		s3 = _mm_add_pd (s3, _mm_mul_pd (_mm_loadu_pd (coeffs + k + 6), sinc_load2_pd_sse2 (data + k + 6))) ;
		} ;

	_mm_storeu_pd (sum, _mm_add_pd (_mm_add_pd (s0, s1), _mm_add_pd (s2, s3))) ;
	for ( ; k < count ; k++)
		sum [0] += coeffs [k] * data [k] ;

	acc [0] += sum [0] + sum [1] ;
} /* sinc_dot_mono_sse2 */

static inline void
sinc_dot_stereo_sse2 (const double *coeffs, const float *data, int count, double *acc)
{	__m128d s0, s1, s2, s3 ;
	double sum [2] ;
	int k ;

	s0 = s1 = s2 = s3 = _mm_setzero_pd () ;
	for (k = 0 ; k + 4 <= count ; k += 4)
	{	const __m128d c0 = _mm_loadu_pd (coeffs + k), c1 = _mm_loadu_pd (coeffs + k + 2) ;

		s0 = _mm_add_pd (s0, _mm_mul_pd (_mm_unpacklo_pd (c0, c0), sinc_load2_pd_sse2 (data + 2 * k))) ;
		s1 = _mm_add_pd (s1, _mm_mul_pd (_mm_unpackhi_pd (c0, c0), sinc_load2_pd_sse2 (data + 2 * k + 2))) ;
		s2 = _mm_add_pd (s2, _mm_mul_pd (_mm_unpacklo_pd (c1, c1), sinc_load2_pd_sse2 (data + 2 * k + 4))) ;
		s3 = _mm_add_pd (s3, _mm_mul_pd (_mm_unpackhi_pd (c1, c1), sinc_load2_pd_sse2 (data + 2 * k + 6))) ;
		} ;

	_mm_storeu_pd (sum, _mm_add_pd (_mm_add_pd (s0, s1), _mm_add_pd (s2, s3))) ;
	for ( ; k < count ; k++)
	{	sum [0] += coeffs [k] * data [2 * k] ;
		sum [1] += coeffs [k] * data [2 * k + 1] ;
		} ;

	acc [0] += sum [0] ;
	acc [1] += sum [1] ;
} /* sinc_dot_stereo_sse2 */

static inline void
sinc_dot_multi_sse2 (const double *coeffs, const float *data, int count, int channels, double *acc)
{	double dcoeffs [2 * COEFF_CHUNK] ;
	int k ;

	assert (count <= COEFF_CHUNK) ;

	/* Duplicate each coefficient to apply it to two channels at once. */
	for (k = 0 ; k < count ; k++)
		_mm_storeu_pd (dcoeffs + 2 * k, _mm_set1_pd (coeffs [k])) ;

	/* Then handle two channels at a time (the last one alone if there is an odd number). */
	for (int ch = 0 ; ch < channels ; ch += 2)
	{	const int single = (ch + 1 == channels) ;
		const float *ptr = data + ch ;
		__m128d s0, s1, s2, s3 ;
		double sum [2] ;

		s0 = s1 = s2 = s3 = _mm_setzero_pd () ;
		for (k = 0 ; k + 4 <= count ; k += 4)
		{	if (single)
			{	s0 = _mm_add_pd (s0, _mm_mul_pd (_mm_loadu_pd (dcoeffs + 2 * k), _mm_cvtps_pd (_mm_load_ss (ptr)))) ;
				s1 = _mm_add_pd (s1, _mm_mul_pd (_mm_loadu_pd (dcoeffs + 2 * k + 2), _mm_cvtps_pd (_mm_load_ss (ptr + channels)))) ;
				s2 = _mm_add_pd (s2, _mm_mul_pd (_mm_loadu_pd (dcoeffs + 2 * k + 4), _mm_cvtps_pd (_mm_load_ss (ptr + 2 * channels)))) ;
				s3 = _mm_add_pd (s3, _mm_mul_pd (_mm_loadu_pd (dcoeffs + 2 * k + 6), _mm_cvtps_pd (_mm_load_ss (ptr + 3 * channels)))) ;
				}
			else
			{	s0 = _mm_add_pd (s0, _mm_mul_pd (_mm_loadu_pd (dcoeffs + 2 * k), sinc_load2_pd_sse2 (ptr))) ;
				s1 = _mm_add_pd (s1, _mm_mul_pd (_mm_loadu_pd (dcoeffs + 2 * k + 2), sinc_load2_pd_sse2 (ptr + channels))) ;
				s2 = _mm_add_pd (s2, _mm_mul_pd (_mm_loadu_pd (dcoeffs + 2 * k + 4), sinc_load2_pd_sse2 (ptr + 2 * channels))) ;
				s3 = _mm_add_pd (s3, _mm_mul_pd (_mm_loadu_pd (dcoeffs + 2 * k + 6), sinc_load2_pd_sse2 (ptr + 3 * channels))) ;
				} ;

			ptr += 4 * channels ;
			} ;

		_mm_storeu_pd (sum, _mm_add_pd (_mm_add_pd (s0, s1), _mm_add_pd (s2, s3))) ;
		for ( ; k < count ; k++)
		{	sum [0] += coeffs [k] * ptr [0] ;
			if (!single)
				sum [1] += coeffs [k] * ptr [1] ;

			ptr += channels ;
			} ;

		acc [ch] += sum [0] ;
		if (!single)
			acc [ch + 1] += sum [1] ;
		} ;
} /* sinc_dot_multi_sse2 */

static inline void
sinc_dot_sse2 (const double *coeffs, const float *data, int count, int channels, double *acc)
{
	if (channels == 1)
		sinc_dot_mono_sse2 (coeffs, data, count, acc) ;
	else if (channels == 2)
		sinc_dot_stereo_sse2 (coeffs, data, count, acc) ;
	else
		sinc_dot_multi_sse2 (coeffs, data, count, channels, acc) ;
} /* sinc_dot_sse2 */

static const SINC_KERNEL sinc_kernel_sse2 =
{	sinc_build_coeffs_sse2,
	sinc_build_wing_forward_sse2,
	sinc_build_wing_reverse_sse2,
	sinc_dot_sse2
} ;

static void
calc_output_sse2 (SINC_FILTER *filter, int channels, double input_index, double scale, float *output)
{	sinc_calc_output (&sinc_kernel_sse2, filter, channels, input_index, scale, output) ;
} /* calc_output_sse2 */

LIBSAMPLERATE_DLL_PRIVATE const sinc_calc_output_t sinc_calc_output_sse2 = calc_output_sse2 ;

#else

/* Not built for SSE2 (see src_sinc.h). */
LIBSAMPLERATE_DLL_PRIVATE const sinc_calc_output_t sinc_calc_output_sse2 = NULL ;

#endif
