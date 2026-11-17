/*
** Copyright (c) 2026, Robert Kausch <robert.kausch@freac.org>
** All rights reserved.
**
** This code is released under 2-clause BSD license. Please see the
** file at : https://github.com/libsndfile/libsamplerate/blob/master/COPYING
*/

/*
**	AVX2 version of the inner loops of the sinc converters (see src_sinc.h).
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <assert.h>
#include <math.h>

#include "common.h"
#include "src_sinc.h"

#if defined (__AVX2__) && !defined (_M_ARM64EC)

#include <immintrin.h>

/*
**	This does the same calculations as the SSE2 version on vectors of twice the
**	size. It does not use FMA, so that the coefficients come out identical to
**	the ones of the SSE2 version.
*/

/* Load four floats and convert them to double. */
static inline __m256d
sinc_load4_pd_avx2 (const float *ptr)
{	return _mm256_cvtps_pd (_mm_loadu_ps (ptr)) ;
} /* sinc_load4_pd_avx2 */

static inline void
sinc_build_coeffs_avx2 (const coeff_t *table, int64_t pos, int64_t delta, int count, double *out)
{	const __m256 frac_scale = _mm256_set1_ps (1.0f / 2147483648.0f) ;
	/*
	** The position vectors hold positions 0, 1, 4, 5 and 2, 3, 6, 7 so that the
	** shuffles below yield them in order. They are derived from broadcasts of
	** pos and delta as this is cheap for 32 bit code as well.
	*/
	const __m256i vpos = _mm256_castpd_si256 (_mm256_broadcast_sd ((const double *) (const void *) &pos)) ;
	const __m256i delta1 = _mm256_castpd_si256 (_mm256_broadcast_sd ((const double *) (const void *) &delta)) ;
	const __m256i delta2 = _mm256_add_epi64 (delta1, delta1) ;
	const __m256i delta4 = _mm256_add_epi64 (delta2, delta2) ;
	const __m256i step = _mm256_add_epi64 (delta4, delta4) ;
	__m256i p0, p1 ;

	p0 = _mm256_add_epi64 (vpos, _mm256_blend_epi32 (_mm256_setzero_si256 (), delta1, 0xCC)) ;
	p0 = _mm256_add_epi64 (p0, _mm256_blend_epi32 (_mm256_setzero_si256 (), delta4, 0xF0)) ;
	p1 = _mm256_add_epi64 (p0, delta2) ;

	for (int k = 0 ; k < count ; k += 8)
	{	__m256 lo, hi, x, a, r0, r1, r2, r3, t0, t1, t2, t3, c0, c1, c2, c3 ;
		int indx [8] ;

		/* Split the positions into integer and fractional parts. */
		lo = _mm256_shuffle_ps (_mm256_castsi256_ps (p0), _mm256_castsi256_ps (p1), _MM_SHUFFLE (2, 0, 2, 0)) ;
		hi = _mm256_shuffle_ps (_mm256_castsi256_ps (p0), _mm256_castsi256_ps (p1), _MM_SHUFFLE (3, 1, 3, 1)) ;
		_mm256_storeu_si256 ((__m256i *) indx, _mm256_castps_si256 (hi)) ;
		x = _mm256_mul_ps (_mm256_cvtepi32_ps (_mm256_srli_epi32 (_mm256_castps_si256 (lo), 1)), frac_scale) ;

		/* Fetch the polynomial coefficients for each position. */
		r0 = _mm256_insertf128_ps (_mm256_castps128_ps256 (_mm_loadu_ps (table + 4 * indx [0])), _mm_loadu_ps (table + 4 * indx [4]), 1) ;
		r1 = _mm256_insertf128_ps (_mm256_castps128_ps256 (_mm_loadu_ps (table + 4 * indx [1])), _mm_loadu_ps (table + 4 * indx [5]), 1) ;
		r2 = _mm256_insertf128_ps (_mm256_castps128_ps256 (_mm_loadu_ps (table + 4 * indx [2])), _mm_loadu_ps (table + 4 * indx [6]), 1) ;
		r3 = _mm256_insertf128_ps (_mm256_castps128_ps256 (_mm_loadu_ps (table + 4 * indx [3])), _mm_loadu_ps (table + 4 * indx [7]), 1) ;

		/* Transpose them (within each half of the vectors). */
		t0 = _mm256_unpacklo_ps (r0, r1) ;
		t1 = _mm256_unpacklo_ps (r2, r3) ;
		t2 = _mm256_unpackhi_ps (r0, r1) ;
		t3 = _mm256_unpackhi_ps (r2, r3) ;
		c0 = _mm256_shuffle_ps (t0, t1, _MM_SHUFFLE (1, 0, 1, 0)) ;
		c1 = _mm256_shuffle_ps (t0, t1, _MM_SHUFFLE (3, 2, 3, 2)) ;
		c2 = _mm256_shuffle_ps (t2, t3, _MM_SHUFFLE (1, 0, 1, 0)) ;
		c3 = _mm256_shuffle_ps (t2, t3, _MM_SHUFFLE (3, 2, 3, 2)) ;

		a = _mm256_add_ps (_mm256_mul_ps (c3, x), c2) ;
		a = _mm256_add_ps (_mm256_mul_ps (a, x), c1) ;
		a = _mm256_add_ps (_mm256_mul_ps (a, x), c0) ;

		_mm256_storeu_pd (out + k, _mm256_cvtps_pd (_mm256_castps256_ps128 (a))) ;
		_mm256_storeu_pd (out + k + 4, _mm256_cvtps_pd (_mm256_extractf128_ps (a, 1))) ;

		p0 = _mm256_add_epi64 (p0, step) ;
		p1 = _mm256_add_epi64 (p1, step) ;
		} ;
} /* sinc_build_coeffs_avx2 */

static inline void
sinc_build_wing_forward_avx2 (const double *rows, int row_len, const double *weights, int count, double *out)
{	const __m256d w0 = _mm256_set1_pd (weights [0]), w1 = _mm256_set1_pd (weights [1]) ;
	const __m256d w2 = _mm256_set1_pd (weights [2]), w3 = _mm256_set1_pd (weights [3]) ;

	for (int k = 0 ; k < count ; k += 4)
	{	__m256d a, b ;

		a = _mm256_add_pd (_mm256_mul_pd (w0, _mm256_loadu_pd (rows + k)), _mm256_mul_pd (w1, _mm256_loadu_pd (rows + row_len + k))) ;
		b = _mm256_add_pd (_mm256_mul_pd (w2, _mm256_loadu_pd (rows + 2 * row_len + k)), _mm256_mul_pd (w3, _mm256_loadu_pd (rows + 3 * row_len + k))) ;
		_mm256_storeu_pd (out + k, _mm256_add_pd (a, b)) ;
		} ;
} /* sinc_build_wing_forward_avx2 */

static inline void
sinc_build_wing_reverse_avx2 (const double *rows, int row_len, const double *weights, int count, double *out)
{	const __m256d w0 = _mm256_set1_pd (weights [0]), w1 = _mm256_set1_pd (weights [1]) ;
	const __m256d w2 = _mm256_set1_pd (weights [2]), w3 = _mm256_set1_pd (weights [3]) ;

	for (int k = 0 ; k < count ; k += 4)
	{	__m256d a, b ;

		a = _mm256_add_pd (_mm256_mul_pd (w0, _mm256_loadu_pd (rows + k)), _mm256_mul_pd (w1, _mm256_loadu_pd (rows + row_len + k))) ;
		b = _mm256_add_pd (_mm256_mul_pd (w2, _mm256_loadu_pd (rows + 2 * row_len + k)), _mm256_mul_pd (w3, _mm256_loadu_pd (rows + 3 * row_len + k))) ;
		a = _mm256_add_pd (a, b) ;
		_mm256_storeu_pd (out - k - 3, _mm256_permute4x64_pd (a, _MM_SHUFFLE (0, 1, 2, 3))) ;
		} ;
} /* sinc_build_wing_reverse_avx2 */

static inline void
sinc_dot_mono_avx2 (const double *coeffs, const float *data, int count, double *acc)
{	__m256d s0, s1, s2, s3 ;
	double sum [4] ;
	int k ;

	s0 = s1 = s2 = s3 = _mm256_setzero_pd () ;
	for (k = 0 ; k + 16 <= count ; k += 16)
	{	s0 = _mm256_add_pd (s0, _mm256_mul_pd (_mm256_loadu_pd (coeffs + k), sinc_load4_pd_avx2 (data + k))) ;
		s1 = _mm256_add_pd (s1, _mm256_mul_pd (_mm256_loadu_pd (coeffs + k + 4), sinc_load4_pd_avx2 (data + k + 4))) ;
		s2 = _mm256_add_pd (s2, _mm256_mul_pd (_mm256_loadu_pd (coeffs + k + 8), sinc_load4_pd_avx2 (data + k + 8))) ;
		s3 = _mm256_add_pd (s3, _mm256_mul_pd (_mm256_loadu_pd (coeffs + k + 12), sinc_load4_pd_avx2 (data + k + 12))) ;
		} ;

	for ( ; k + 4 <= count ; k += 4)
		s0 = _mm256_add_pd (s0, _mm256_mul_pd (_mm256_loadu_pd (coeffs + k), sinc_load4_pd_avx2 (data + k))) ;

	_mm256_storeu_pd (sum, _mm256_add_pd (_mm256_add_pd (s0, s1), _mm256_add_pd (s2, s3))) ;

	for ( ; k < count ; k++)
		sum [0] += coeffs [k] * data [k] ;

	acc [0] += (sum [0] + sum [1]) + (sum [2] + sum [3]) ;
} /* sinc_dot_mono_avx2 */

static inline void
sinc_dot_stereo_avx2 (const double *coeffs, const float *data, int count, double *acc)
{	__m256d s0, s1, s2, s3 ;
	double sum [4] ;
	int k ;

	/* Each vector holds left, right, left, right. */
	s0 = s1 = s2 = s3 = _mm256_setzero_pd () ;
	for (k = 0 ; k + 8 <= count ; k += 8)
	{	const __m256d c0 = _mm256_loadu_pd (coeffs + k), c1 = _mm256_loadu_pd (coeffs + k + 4) ;

		s0 = _mm256_add_pd (s0, _mm256_mul_pd (_mm256_permute4x64_pd (c0, _MM_SHUFFLE (1, 1, 0, 0)), sinc_load4_pd_avx2 (data + 2 * k))) ;
		s1 = _mm256_add_pd (s1, _mm256_mul_pd (_mm256_permute4x64_pd (c0, _MM_SHUFFLE (3, 3, 2, 2)), sinc_load4_pd_avx2 (data + 2 * k + 4))) ;
		s2 = _mm256_add_pd (s2, _mm256_mul_pd (_mm256_permute4x64_pd (c1, _MM_SHUFFLE (1, 1, 0, 0)), sinc_load4_pd_avx2 (data + 2 * k + 8))) ;
		s3 = _mm256_add_pd (s3, _mm256_mul_pd (_mm256_permute4x64_pd (c1, _MM_SHUFFLE (3, 3, 2, 2)), sinc_load4_pd_avx2 (data + 2 * k + 12))) ;
		} ;

	_mm256_storeu_pd (sum, _mm256_add_pd (_mm256_add_pd (s0, s1), _mm256_add_pd (s2, s3))) ;

	sum [0] += sum [2] ;
	sum [1] += sum [3] ;
	for ( ; k < count ; k++)
	{	sum [0] += coeffs [k] * data [2 * k] ;
		sum [1] += coeffs [k] * data [2 * k + 1] ;
		} ;

	acc [0] += sum [0] ;
	acc [1] += sum [1] ;
} /* sinc_dot_stereo_avx2 */

/* Load two floats from each of two places (or one float from each) and convert them to double. */
static inline __m256d
sinc_load_pairs_avx2 (const float *first, const float *second, int single)
{	__m128i pairs ;

	if (single)
		pairs = _mm_unpacklo_epi64 (_mm_castps_si128 (_mm_load_ss (first)), _mm_castps_si128 (_mm_load_ss (second))) ;
	else
		pairs = _mm_unpacklo_epi64 (_mm_loadl_epi64 ((const __m128i *) first), _mm_loadl_epi64 ((const __m128i *) second)) ;

	return _mm256_cvtps_pd (_mm_castsi128_ps (pairs)) ;
} /* sinc_load_pairs_avx2 */

static inline void
sinc_dot_multi_avx2 (const double *coeffs, const float *data, int count, int channels, double *acc)
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
		__m256d s0, s1, s2, s3 ;
		double sum [4] ;

		s0 = s1 = s2 = s3 = _mm256_setzero_pd () ;
		for (k = 0 ; k + 8 <= count ; k += 8)
		{	s0 = _mm256_add_pd (s0, _mm256_mul_pd (_mm256_loadu_pd (dcoeffs + 2 * k), sinc_load_pairs_avx2 (ptr, ptr + channels, single))) ;
			s1 = _mm256_add_pd (s1, _mm256_mul_pd (_mm256_loadu_pd (dcoeffs + 2 * k + 4), sinc_load_pairs_avx2 (ptr + 2 * channels, ptr + 3 * channels, single))) ;
			s2 = _mm256_add_pd (s2, _mm256_mul_pd (_mm256_loadu_pd (dcoeffs + 2 * k + 8), sinc_load_pairs_avx2 (ptr + 4 * channels, ptr + 5 * channels, single))) ;
			s3 = _mm256_add_pd (s3, _mm256_mul_pd (_mm256_loadu_pd (dcoeffs + 2 * k + 12), sinc_load_pairs_avx2 (ptr + 6 * channels, ptr + 7 * channels, single))) ;

			ptr += 8 * channels ;
			} ;

		_mm256_storeu_pd (sum, _mm256_add_pd (_mm256_add_pd (s0, s1), _mm256_add_pd (s2, s3))) ;

		sum [0] += sum [2] ;
		sum [1] += sum [3] ;
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
} /* sinc_dot_multi_avx2 */

static inline void
sinc_dot_avx2 (const double *coeffs, const float *data, int count, int channels, double *acc)
{
	if (channels == 1)
		sinc_dot_mono_avx2 (coeffs, data, count, acc) ;
	else if (channels == 2)
		sinc_dot_stereo_avx2 (coeffs, data, count, acc) ;
	else
		sinc_dot_multi_avx2 (coeffs, data, count, channels, acc) ;
} /* sinc_dot_avx2 */

static const SINC_KERNEL sinc_kernel_avx2 =
{	sinc_build_coeffs_avx2,
	sinc_build_wing_forward_avx2,
	sinc_build_wing_reverse_avx2,
	sinc_dot_avx2
} ;

static void
calc_output_avx2 (SINC_FILTER *filter, int channels, double input_index, double scale, float *output)
{	sinc_calc_output (&sinc_kernel_avx2, filter, channels, input_index, scale, output) ;

	/*
	** Code built for older instruction sets may run next, which is slow while
	** the upper halves of the YMM registers are in use.
	*/
	_mm256_zeroupper () ;
} /* calc_output_avx2 */

LIBSAMPLERATE_DLL_PRIVATE const sinc_calc_output_t sinc_calc_output_avx2 = calc_output_avx2 ;

#else

/* Not built for AVX2 (see src_sinc.h). */
LIBSAMPLERATE_DLL_PRIVATE const sinc_calc_output_t sinc_calc_output_avx2 = NULL ;

#endif
