
/* Copyright 2015 Haakan T. Johansson */

/*  This file is part of WIGXJPF.
 *
 *  WIGXJPF is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  WIGXJPF is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with WIGXJPF.  If not, see
 *  <http://www.gnu.org/licenses/>.
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>

#include "wigxjpf.h"

#include "wigxjpf_config.h"
#include "calc_minmax.h"

extern double factorial_log_precalc[FPSIMPLE_MAX_FACTORIAL+1];
extern int max_factorial_log_precalc;

#define CHECK_MAX_PRECALC_LOG_FACTORIAL(maxfact) do {			\
    if ((maxfact) > max_factorial_log_precalc) {			\
      fprintf (stderr, "Too large factorial (%d!).  Abort.\n", maxfact); \
      exit(1);								\
    }									\
  } while (0)

void fpsimple_log_delta_coeff(int two_a, int two_b, int two_c,
			      double *log_prefact)
{
  /* Check maximum factorial. */

  int max_factorial = (two_a + two_b + two_c) / 2 + 1;

  CHECK_MAX_PRECALC_LOG_FACTORIAL(max_factorial);

  /* Calculate. */

  const double *p_n1 =
    &factorial_log_precalc[(two_a + two_b - two_c) / 2];
  const double *p_n2 =
    &factorial_log_precalc[(two_a + two_c - two_b) / 2];
  const double *p_n3 =
    &factorial_log_precalc[(two_b + two_c - two_a) / 2];

  const double *p_d1 =
    &factorial_log_precalc[(two_a + two_b + two_c) / 2 + 1];

  *log_prefact += (*p_n1) + (*p_n2) + (*p_n3) - (*p_d1);
}

void fpsimple_log_3j(double *result,
		     int two_j1, int two_j2, int two_j3,
		     int two_m1, int two_m2, int two_m3)
{
  int kmin = max3(two_j1 + two_m2 - two_j3,
		  two_j2 - two_m1 - two_j3,
		  0) / 2;
  int kmax = min3(two_j2 + two_m2,
		  two_j1 - two_m1,
		  two_j1 + two_j2 - two_j3) / 2;

  /* Check maximum factorial. */

  int max_factorial = (two_j1 + two_j2 + two_j3) / 2 + 1;

  CHECK_MAX_PRECALC_LOG_FACTORIAL(max_factorial);

  /* Prefactor. */

  double log_prefact;

  {
    const double *p_n1 =
      &factorial_log_precalc[(  two_j1 + two_j2 - two_j3) / 2];
    const double *p_n2 =
      &factorial_log_precalc[(  two_j1 - two_j2 + two_j3) / 2];
    const double *p_n3 =
      &factorial_log_precalc[(- two_j1 + two_j2 + two_j3) / 2];

    const double *p_d1 =
      &factorial_log_precalc[(two_j1 + two_j2 + two_j3) / 2 + 1];

    const double *p_n4 =
      &factorial_log_precalc[(two_j1 - two_m1) / 2];
    const double *p_n5 =
      &factorial_log_precalc[(two_j1 + two_m1) / 2];
    const double *p_n6 =
      &factorial_log_precalc[(two_j2 - two_m2) / 2];
    const double *p_n7 =
      &factorial_log_precalc[(two_j2 + two_m2) / 2];
    const double *p_n8 =
      &factorial_log_precalc[(two_j3 - two_m3) / 2];
    const double *p_n9 =
      &factorial_log_precalc[(two_j3 + two_m3) / 2];

    log_prefact = ((*p_n1) + (*p_n2) + (*p_n3) +
		   (*p_n4) + (*p_n5) + (*p_n6) +
		   (*p_n7) + (*p_n8) + (*p_n9)) - (*p_d1);
  }

  log_prefact *= 0.5; /* sqrt */

  /* Do the loop. */

  int k;

  const double *p_d1 =
    &factorial_log_precalc[kmin];

  const double *p_d2 =
    &factorial_log_precalc[kmin + (two_j3 - two_j1 - two_m2) / 2];
  const double *p_d3 =
    &factorial_log_precalc[kmin + (two_j3 - two_j2 + two_m1) / 2];

  const double *p_d4 =
    &factorial_log_precalc[(two_j2 + two_m2) / 2 - kmin];
  const double *p_d5 =
    &factorial_log_precalc[(two_j1 - two_m1) / 2 - kmin];
  const double *p_d6 =
    &factorial_log_precalc[(two_j1 + two_j2 - two_j3) / 2 - kmin];
      
  int k_lim = kmax - kmin;

  double sum_prod = 0;

  int sign = kmin ^ ((two_j1 - two_j2 - two_m3) / 2);

  for (k = 0; k <= k_lim; k++)
    {
      double log_prod = log_prefact - (*(p_d1) + *(p_d2) + *(p_d3) +
				       *(p_d4) + *(p_d5) + *(p_d6));

      p_d1++;
      p_d2++;
      p_d3++;
      p_d4--;
      p_d5--;
      p_d6--;

      sum_prod += (1 - 2 * ((k ^ sign) & 1)) * exp(log_prod);
    }

  *result = sum_prod;
}

void fpsimple_log_6j(double *result,
		     int two_j1, int two_j2, int two_j3,
		     int two_j4, int two_j5, int two_j6)
{
  int two_a = two_j1;
  int two_b = two_j2;
  int two_c = two_j5;
  int two_d = two_j4;
  int two_e = two_j3;
  int two_f = two_j6;

  double log_prefact = 0;

  fpsimple_log_delta_coeff(two_a, two_b, two_e, &log_prefact);
  fpsimple_log_delta_coeff(two_c, two_d, two_e, &log_prefact);
  fpsimple_log_delta_coeff(two_a, two_c, two_f, &log_prefact);
  fpsimple_log_delta_coeff(two_b, two_d, two_f, &log_prefact);

  log_prefact *= 0.5; /* sqrt */

  double sum_prod;

  int kmin = max4(two_a + two_b + two_e,
		  two_c + two_d + two_e,
		  two_a + two_c + two_f,
		  two_b + two_d + two_f) / 2;
  int kmax = min3(two_a + two_b + two_c + two_d,
		  two_a + two_d + two_e + two_f,
		  two_b + two_c + two_e + two_f) / 2;

  /* Check maximum factorial. */

  int max_factorial = max4(kmax + 1,
			   (two_a + two_b + two_c + two_d) / 2,
			   (two_a + two_d + two_e + two_f) / 2,
			   (two_b + two_c + two_e + two_f) / 2);
  
  CHECK_MAX_PRECALC_LOG_FACTORIAL(max_factorial);

  /* Do the loop. */

  int k;

  const double *p_d1 =
    &factorial_log_precalc[kmin - (two_a + two_b + two_e) / 2];
  const double *p_d2 =
    &factorial_log_precalc[kmin - (two_c + two_d + two_e) / 2];
  const double *p_d3 =
    &factorial_log_precalc[kmin - (two_a + two_c + two_f) / 2];
  const double *p_d4 =
    &factorial_log_precalc[kmin - (two_b + two_d + two_f) / 2];

  const double *p_d5 =
    &factorial_log_precalc[(two_a + two_b + two_c + two_d) / 2 - kmin];
  const double *p_d6 =
    &factorial_log_precalc[(two_a + two_d + two_e + two_f) / 2 - kmin];
  const double *p_d7 =
    &factorial_log_precalc[(two_b + two_c + two_e + two_f) / 2 - kmin];
      
  const double *p_n1 =
    &factorial_log_precalc[kmin + 1];

  int k_lim = kmax - kmin;

  sum_prod = 0;

  for (k = 0; k <= k_lim; k++)
    {
      double log_prod = log_prefact + (*p_n1) -
	(*(p_d1) + *(p_d2) + *(p_d3) +
	 *(p_d4) + *(p_d5) + *(p_d6) + *(p_d7));

      p_n1++;
      p_d1++;
      p_d2++;
      p_d3++;
      p_d4++;
      p_d5--;
      p_d6--;
      p_d7--;

      sum_prod += (1 - 2 * ((k ^ kmin) & 1)) * exp(log_prod);
    }

  *result = sum_prod;
}

void fpsimple_log_9j(double *result,
		     int two_a, int two_b, int two_c,
		     int two_d, int two_e, int two_f,
		     int two_g, int two_h, int two_i)
{
  int two_kmin = max3(abs(two_h - two_d),
		      abs(two_b - two_f),
		      abs(two_a - two_i));
  int two_kmax = min3(two_h + two_d,
		      two_b + two_f,
		      two_a + two_i);
  int two_k;

  double sum_prod = 0;

  for (two_k = two_kmin; two_k <= two_kmax; two_k += 2)
    {
      double f1, f2, f3;

      fpsimple_log_6j(&f1, two_a, two_b, two_c, two_f, two_i, two_k);

      fpsimple_log_6j(&f2, two_f, two_d, two_e, two_h, two_b, two_k);
      
      fpsimple_log_6j(&f3, two_h, two_i, two_g, two_a, two_d, two_k);

      double factor = f1 * f2 * f3;

      /* And multiply by the coefficient 2*k+1. */

      double prod = (two_k + 1) * factor;

      sum_prod += (1 - 2 * ((two_k) & 1)) * prod;
   }

  *result = sum_prod;
}
