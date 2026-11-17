/*
** Copyright (c) 2026, Robert Kausch <robert.kausch@freac.org>
** All rights reserved.
**
** This code is released under 2-clause BSD license. Please see the
** file at : https://github.com/libsndfile/libsamplerate/blob/master/COPYING
*/

/*
**	NEON version of the inner loops of the sinc converters (see src_sinc.h).
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <assert.h>
#include <math.h>

#include "common.h"
#include "src_sinc.h"

#if (defined (__aarch64__) && defined (__AARCH64EL__)) || defined (_M_ARM64) || defined (_M_ARM64EC)

#include <arm_neon.h>

/*
**	All 64 bit ARM CPUs have NEON (including the double precision vectors used
**	here), so no run time check is needed.
*/

static inline void
sinc_build_coeffs_neon (const coeff_t *table, int64_t pos, int64_t delta, int count, double *out)
{	const int64x2_t step = vdupq_n_s64 (4 * delta) ;
	int64x2_t p0 = vcombine_s64 (vcreate_s64 ((uint64_t) pos), vcreate_s64 ((uint64_t) (pos + delta))) ;
	int64x2_t p1 = vaddq_s64 (p0, vdupq_n_s64 (2 * delta)) ;

	for (int k = 0 ; k < count ; k += 4)
	{	float32x4_t x, a, r0, r1, r2, r3, t0, t1, t2, t3, c0, c1, c2, c3 ;
		uint32x4_t lo ;
		int32x4_t hi ;

		/* Split the positions into integer and fractional parts. */
		lo = vuzp1q_u32 (vreinterpretq_u32_s64 (p0), vreinterpretq_u32_s64 (p1)) ;
		hi = vuzp2q_s32 (vreinterpretq_s32_s64 (p0), vreinterpretq_s32_s64 (p1)) ;
		x = vmulq_n_f32 (vcvtq_f32_u32 (vshrq_n_u32 (lo, 1)), 1.0f / 2147483648.0f) ;

		/* Fetch the polynomial coefficients for each position and transpose them. */
		r0 = vld1q_f32 (table + 4 * vgetq_lane_s32 (hi, 0)) ;
		r1 = vld1q_f32 (table + 4 * vgetq_lane_s32 (hi, 1)) ;
		r2 = vld1q_f32 (table + 4 * vgetq_lane_s32 (hi, 2)) ;
		r3 = vld1q_f32 (table + 4 * vgetq_lane_s32 (hi, 3)) ;

		t0 = vzip1q_f32 (r0, r2) ;
		t1 = vzip1q_f32 (r1, r3) ;
		t2 = vzip2q_f32 (r0, r2) ;
		t3 = vzip2q_f32 (r1, r3) ;
		c0 = vzip1q_f32 (t0, t1) ;
		c1 = vzip2q_f32 (t0, t1) ;
		c2 = vzip1q_f32 (t2, t3) ;
		c3 = vzip2q_f32 (t2, t3) ;

		a = vfmaq_f32 (c2, c3, x) ;
		a = vfmaq_f32 (c1, a, x) ;
		a = vfmaq_f32 (c0, a, x) ;

		vst1q_f64 (out + k, vcvt_f64_f32 (vget_low_f32 (a))) ;
		vst1q_f64 (out + k + 2, vcvt_high_f64_f32 (a)) ;

		p0 = vaddq_s64 (p0, step) ;
		p1 = vaddq_s64 (p1, step) ;
		} ;
} /* sinc_build_coeffs_neon */

static inline void
sinc_build_wing_forward_neon (const double *rows, int row_len, const double *weights, int count, double *out)
{	const float64x2_t w0 = vdupq_n_f64 (weights [0]), w1 = vdupq_n_f64 (weights [1]) ;
	const float64x2_t w2 = vdupq_n_f64 (weights [2]), w3 = vdupq_n_f64 (weights [3]) ;

	for (int k = 0 ; k < count ; k += 2)
	{	float64x2_t a, b ;

		a = vfmaq_f64 (vmulq_f64 (w0, vld1q_f64 (rows + k)), w1, vld1q_f64 (rows + row_len + k)) ;
		b = vfmaq_f64 (vmulq_f64 (w2, vld1q_f64 (rows + 2 * row_len + k)), w3, vld1q_f64 (rows + 3 * row_len + k)) ;
		vst1q_f64 (out + k, vaddq_f64 (a, b)) ;
		} ;
} /* sinc_build_wing_forward_neon */

static inline void
sinc_build_wing_reverse_neon (const double *rows, int row_len, const double *weights, int count, double *out)
{	const float64x2_t w0 = vdupq_n_f64 (weights [0]), w1 = vdupq_n_f64 (weights [1]) ;
	const float64x2_t w2 = vdupq_n_f64 (weights [2]), w3 = vdupq_n_f64 (weights [3]) ;

	for (int k = 0 ; k < count ; k += 2)
	{	float64x2_t a, b ;

		a = vfmaq_f64 (vmulq_f64 (w0, vld1q_f64 (rows + k)), w1, vld1q_f64 (rows + row_len + k)) ;
		b = vfmaq_f64 (vmulq_f64 (w2, vld1q_f64 (rows + 2 * row_len + k)), w3, vld1q_f64 (rows + 3 * row_len + k)) ;
		a = vaddq_f64 (a, b) ;
		vst1q_f64 (out - k - 1, vextq_f64 (a, a, 1)) ;
		} ;
} /* sinc_build_wing_reverse_neon */

static inline void
sinc_dot_mono_neon (const double *coeffs, const float *data, int count, double *acc)
{	float64x2_t s0, s1, s2, s3 ;
	double sum ;
	int k ;

	s0 = s1 = s2 = s3 = vdupq_n_f64 (0.0) ;
	for (k = 0 ; k + 8 <= count ; k += 8)
	{	const float32x4_t d0 = vld1q_f32 (data + k), d1 = vld1q_f32 (data + k + 4) ;

		s0 = vfmaq_f64 (s0, vld1q_f64 (coeffs + k), vcvt_f64_f32 (vget_low_f32 (d0))) ;
		s1 = vfmaq_f64 (s1, vld1q_f64 (coeffs + k + 2), vcvt_high_f64_f32 (d0)) ;
		s2 = vfmaq_f64 (s2, vld1q_f64 (coeffs + k + 4), vcvt_f64_f32 (vget_low_f32 (d1))) ;
		s3 = vfmaq_f64 (s3, vld1q_f64 (coeffs + k + 6), vcvt_high_f64_f32 (d1)) ;
		} ;

	sum = vaddvq_f64 (vaddq_f64 (vaddq_f64 (s0, s1), vaddq_f64 (s2, s3))) ;
	for ( ; k < count ; k++)
		sum += coeffs [k] * data [k] ;

	acc [0] += sum ;
} /* sinc_dot_mono_neon */

static inline void
sinc_dot_stereo_neon (const double *coeffs, const float *data, int count, double *acc)
{	float64x2_t s0, s1, s2, s3 ;
	int k ;

	/* Each vector holds left and right. */
	s0 = s1 = s2 = s3 = vdupq_n_f64 (0.0) ;
	for (k = 0 ; k + 4 <= count ; k += 4)
	{	const float32x4_t d0 = vld1q_f32 (data + 2 * k), d1 = vld1q_f32 (data + 2 * k + 4) ;

		s0 = vfmaq_n_f64 (s0, vcvt_f64_f32 (vget_low_f32 (d0)), coeffs [k]) ;
		s1 = vfmaq_n_f64 (s1, vcvt_high_f64_f32 (d0), coeffs [k + 1]) ;
		s2 = vfmaq_n_f64 (s2, vcvt_f64_f32 (vget_low_f32 (d1)), coeffs [k + 2]) ;
		s3 = vfmaq_n_f64 (s3, vcvt_high_f64_f32 (d1), coeffs [k + 3]) ;
		} ;

	for ( ; k < count ; k++)
		s0 = vfmaq_n_f64 (s0, vcvt_f64_f32 (vld1_f32 (data + 2 * k)), coeffs [k]) ;

	vst1q_f64 (acc, vaddq_f64 (vld1q_f64 (acc), vaddq_f64 (vaddq_f64 (s0, s1), vaddq_f64 (s2, s3)))) ;
} /* sinc_dot_stereo_neon */

static inline void
sinc_dot_multi_neon (const double *coeffs, const float *data, int count, int channels, double *acc)
{	int ch, k ;

	/* Handle two channels at a time. */
	for (ch = 0 ; ch + 2 <= channels ; ch += 2)
	{	const float *ptr = data + ch ;
		float64x2_t s0, s1, s2, s3 ;

		s0 = s1 = s2 = s3 = vdupq_n_f64 (0.0) ;
		for (k = 0 ; k + 4 <= count ; k += 4)
		{	s0 = vfmaq_n_f64 (s0, vcvt_f64_f32 (vld1_f32 (ptr)), coeffs [k]) ;
			s1 = vfmaq_n_f64 (s1, vcvt_f64_f32 (vld1_f32 (ptr + channels)), coeffs [k + 1]) ;
			s2 = vfmaq_n_f64 (s2, vcvt_f64_f32 (vld1_f32 (ptr + 2 * channels)), coeffs [k + 2]) ;
			s3 = vfmaq_n_f64 (s3, vcvt_f64_f32 (vld1_f32 (ptr + 3 * channels)), coeffs [k + 3]) ;

			ptr += 4 * channels ;
			} ;

		for ( ; k < count ; k++)
		{	s0 = vfmaq_n_f64 (s0, vcvt_f64_f32 (vld1_f32 (ptr)), coeffs [k]) ;
			ptr += channels ;
			} ;

		vst1q_f64 (acc + ch, vaddq_f64 (vld1q_f64 (acc + ch), vaddq_f64 (vaddq_f64 (s0, s1), vaddq_f64 (s2, s3)))) ;
		} ;

	/* And the last one alone if there is an odd number. */
	if (ch < channels)
	{	const float *ptr = data + ch ;
		double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0 ;

		for (k = 0 ; k + 4 <= count ; k += 4)
		{	s0 += coeffs [k] * ptr [0] ;
			s1 += coeffs [k + 1] * ptr [channels] ;
			s2 += coeffs [k + 2] * ptr [2 * channels] ;
			s3 += coeffs [k + 3] * ptr [3 * channels] ;

			ptr += 4 * channels ;
			} ;

		for ( ; k < count ; k++)
		{	s0 += coeffs [k] * ptr [0] ;
			ptr += channels ;
			} ;

		acc [ch] += (s0 + s1) + (s2 + s3) ;
		} ;
} /* sinc_dot_multi_neon */

static inline void
sinc_dot_neon (const double *coeffs, const float *data, int count, int channels, double *acc)
{
	if (channels == 1)
		sinc_dot_mono_neon (coeffs, data, count, acc) ;
	else if (channels == 2)
		sinc_dot_stereo_neon (coeffs, data, count, acc) ;
	else
		sinc_dot_multi_neon (coeffs, data, count, channels, acc) ;
} /* sinc_dot_neon */

static const SINC_KERNEL sinc_kernel_neon =
{	sinc_build_coeffs_neon,
	sinc_build_wing_forward_neon,
	sinc_build_wing_reverse_neon,
	sinc_dot_neon
} ;

static void
calc_output_neon (SINC_FILTER *filter, int channels, double input_index, double scale, float *output)
{	sinc_calc_output (&sinc_kernel_neon, filter, channels, input_index, scale, output) ;
} /* calc_output_neon */

LIBSAMPLERATE_DLL_PRIVATE const sinc_calc_output_t sinc_calc_output_neon = calc_output_neon ;

#else

/* Not built for 64 bit ARM (see src_sinc.h). */
LIBSAMPLERATE_DLL_PRIVATE const sinc_calc_output_t sinc_calc_output_neon = NULL ;

#endif
