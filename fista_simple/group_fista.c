#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <cblas.h>
#include <sys/time.h>

#define INF INT_MAX
#define FREE_ARG char*

static double sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)

static double dmaxarg1,dmaxarg2;
#define DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b),(dmaxarg1) > (dmaxarg2) ?\
        (dmaxarg1) : (dmaxarg2))

static double signarg;
#define SIGN(a) ((signarg=(a)) == 0.0 ? 0.0 : (signarg < 0.0 ? -1.0 : 1.0))

void runerror(char error_text[])
/* Numerical Recipes standard error handler */
{
        fprintf(stdout,"Run-time error...\n");
        fprintf(stdout,"%s\n",error_text);
        fprintf(stderr,"...now exiting to system...\n");
        exit(1);
}

/* ALLOC */
double *vector(long n)
/* allocate a double vector with subscript range v[nl..nh] */
{
        double *v;

        v=(double *)malloc((size_t) ((n)*sizeof(double)));
        if (!v) runerror("allocation failure in vector()");
        return v;
}

/* FREE */
void free_vector(double *v)
/* free a double vector allocated with vector() */
{
        free((FREE_ARG) (v));
}

/* PRINT */
void dvprint(FILE *f, double *a, int a_els)
{
   for (int i=0; i < (a_els-1); i++)
      fprintf(f, "%g ", a[i]);
   fprintf(f, "%g\n",a[a_els-1]);
}

/* COPY */
void dvcopy(double *a, int a_els, double *y)
{
#ifdef _BLAS_
   cblas_dcopy(a_els, a, 1, y, 1);
#else
   for (int i=0; i < a_els; i++)
      y[i] = a[i];
#endif
}

/* ---------------- Vector-scalar functions */
double dvdot(double *a, int a_els, double *b)
{
#ifdef _BLAS_
   return cblas_ddot(a_els, a, 1, b, 1);
#else
   double y;

   y = 0.0;
   for (int i=0; i < a_els; i++) {
      y += a[i] * b[i];
   }
   return y;
#endif
}

double dv1norm(double *a, int a_els)
{
#ifdef _BLAS_
   return cblas_dasum(a_els, a, 1);
#else
   double y = 0.0;

   for (int i=0; i < a_els; i++)
      y += fabs(a[i]);

   return y;
#endif
}

double dv2norm(double *a, int a_els)
{

#ifdef _BLAS_
   return cblas_dnrm2(a_els, a, 1);
#else
   return sqrt(dvdot(a, a_els, a));
#endif
}

double dv21norm(double *a, int a_els, int *grp, int ngrp)
{
   double norm = 0;

   if (ngrp > 0)
      for(int i=0, j=0; i<ngrp; i++, j+=grp[i])
      {
         norm += dv2norm(&(a[j]), grp[i]);
      }
   else
      norm = dv1norm(a, a_els);
}

/* ---------------- Vector-vector functions */
void dvset(double *a, int a_els, double val)
{
   for (int i=0; i < a_els; i++)
      a[i] = val;
}

void dvscal(double *a, int a_els, double r)
{
#ifdef _BLAS_
   cblas_dscal(a_els, r, a, 1);
#else
   for (int i=0; i < a_els; i++)
      a[i] = a[i] * r;
#endif
}

/* y = ax + y */
void daxpy(double *a, int a_els, double r, double *y)
{
#ifdef _BLAS_
   cblas_daxpy(a_els, r, a, 1, y, 1);
#else
   for (int i=0; i < a_els; i++)
      y[i] = a[i] * r + y[i];
#endif
}

void soft_thresholding(double *x, int n, double lambda)
{
   #pragma omp parallel for
   for(int i=0; i<n; i++)
      x[i] = DMAX(fabs(x[i])-lambda,0)*SIGN(x[i]);
}

void vec_soft_thresholding(double *x, int n, double lambda)
{
   double norm = dv2norm(x, n);

   if (norm <= lambda)
      dvset(x, n, 0);
   else
      dvscal(x, n, (norm-lambda)/norm);
}

void l21_prox(double *x, int n, double lambda, int *grp, int ngrp)
{
   if (ngrp > 0)
      for(int i=0, j=0; i<ngrp; i++, j+=grp[i])
         vec_soft_thresholding(&(x[j]), grp[i], lambda);
   else
      soft_thresholding(x, n, lambda);
}

/* ---------------- Matrix-vector functions */
void dmvmult(double **a, int a_rows, int a_cols,
             double *b, int b_els, double *y)
{
#ifdef _BLAS_
   cblas_dgemv(CblasRowMajor, CblasNoTrans, a_rows, a_cols, 1, &a[0][0],
         a_cols, b, 1, 0, y, 1);
#else
   double sum;

   if (a_cols != b_els)
   {
      fprintf(stderr,"a_cols <> b_els (%d,%d): dmvmult\n", a_cols, b_els);
      exit(1);
   }

   for (int i=0; i < a_rows; i++) {
      sum = 0.0;
      for (int k=0; k < a_cols; k++) sum += a[i][k]*b[k];
      y[i] = sum;
   }
#endif
}

void dsymvmult(double **a, int n, double *b, double *y)
{
#ifdef _BLAS_
   cblas_dsymv(CblasRowMajor, CblasUpper, n, 1, &a[0][0], n, b, 1, 0, y, 1);
#else
   dmvmult(a, n, n, b, n, y);
#endif
}

#define EPS 1.0e-10
/*
 * Function that solves the problem
 *
 *     min 1/2 w^t G w  - c^t w + lambda_1 ||w||_2,1 + lambda_2 ||w||_2
 *
 * using the FISTA algorithm (Proximal Gradient Descent).
 *
 * Given an (n x d) matrix X an a (n x 1) whose columns are standardized
 * (mean 0 and standard deviation 1), vector y (also standardized) and
 *
 *     G = X^t X
 *     c = X^t y
 *
 * the previous problem is equivalent to Group Elastic Net
 *
 *     min 1/2 ||Xw - y||^2_2 + lambda_1 ||w||_2,1 + lambda_2 ||w||_2
 *
 * Note that Group Elastic Net reduces to Elastic Net if every group has
 * only 1 element (ngrp = 0).
 *
 * The group structure is given in a vector grp where the ith position
 * contains the number of elements in group i. Thus the coefficients for
 * the different groups have to be consecutive and not-overlapping.
 *
 * Input:
 *   - G, (dim x dim) matrix, see above
 *   - c, vector of size dim, see above
 *   - w, vector of size dim, see above (usually initialized to 0)
 *   - dim, size of the vector w (d)
 *   - *grp, vector of size ngrp with the group structure
 *   - ngrp, Number of groups
 *   - L, Lipchitz constant of the gradient, estimated as the largest
 *        eigenvalue of G
 *   - lambda_1, l_2,1-regularization parameter
 *   - lambda_2, l_2-regularization parameter
 *   - tol, tolerance for the stopping criterion
 *
 * Output:
 *   - w, final value for the weigths after the optimization
 *   - iter, final number of iterations
 *   - fret, final value of the objective function
 *
 ******************************************************************************/
void fista_gram(double **G, double *c, double *w, int dim, int *grp, int ngrp,
      double L, double lambda_1, double lambda_2, double tol, int *iter, double *fret)
{

   double *w_k, *w_km1, *v_k, *grad, *Gw;
   int keep_going = 1, max_iter = INF;
   double f, prev_f, t_k = 1, t_km1 = t_k;

   /* alloc memory */
   v_k = vector(dim);
   w_k = vector(dim);
   w_km1 = vector(dim);
   grad = vector(dim);
   Gw = vector(dim);

    /* v_k <- w;  w_k <- w */
   dvcopy(w, dim, v_k);
   dvcopy(w, dim, w_k);

   /* compute value of f at w_k */
   dsymvmult(G, dim, w_k, Gw);
   f = 0.5*dvdot(Gw, dim, w) - dvdot(c, dim, w) \
       + lambda_1*dv21norm(w_k, dim, grp, ngrp) \
       + lambda_2*SQR(dv2norm(w_k, dim));


   /* if tol == 0, we want to stop by iterations */
   if (!tol)
      max_iter = (*iter);

   struct timeval t0, t1;
   (*iter) = 0;
   while (keep_going && (*iter) < max_iter)
   {
      gettimeofday(&t0, 0);

      /* w_km1 = w_k */
      dvcopy(w_k, dim, w_km1);

      /* t_km1 = t_k */
      t_km1 = t_k;

      /* grad(v_k) = G*v_k - c + lambda_2*v_k */
      dsymvmult(G, dim, v_k, grad);
      daxpy(c, dim, -1, grad);
      daxpy(v_k, dim, lambda_2, grad);

      /* w_k = prox(v_k - (1/L)*grad(v_k), lambda/L) */
      dvcopy(v_k, dim, w_k);
      daxpy(grad, dim, -(1/L), w_k);
      l21_prox(w_k, dim, lambda_1/L, grp, ngrp);

      /* t_k = (1 + sqrt(1 + 4*t_km1^2)) / 2 */
      t_k = 0.5*(1+sqrt(1+4*SQR(t_km1)));

      /* v_k = w_k + ((t_km1-1)/t_k)*(w_k-w_km1) */
      dvcopy(w_k, dim, v_k);
      dvscal(v_k, dim, 1+(t_km1-1)/t_k);
      daxpy(w_km1, dim, -(t_km1-1)/t_k, v_k);

      /* prev_f = f */
      prev_f = f;

      /* compute value of f at w_k */
      dsymvmult(G, dim, w_k, Gw);
      f = 0.5*dvdot(Gw, dim, w) - dvdot(c, dim, w) \
          + lambda_1*dv21norm(w_k, dim, grp, ngrp) \
          + lambda_2*SQR(dv2norm(w_k, dim));

      //fprintf(stdout, "f=%g\n", f);
      if (tol)
         keep_going = (2.0*fabs(f - prev_f) > tol*(fabs(f)+fabs(prev_f)+EPS));

      (*iter)++;

      gettimeofday(&t1, 0);

#ifdef _DEBUG_
      fprintf(stderr, "%ld.%06ld %ld.%06ld ",
            t0.tv_sec, t0.tv_usec, t1.tv_sec, t1.tv_usec);
      dvprint(stderr, w_k, dim);
#endif
   }

   (*fret) = f;
   dvcopy(w_k, dim, w);

   free_vector(v_k);
   free_vector(w_k);
   free_vector(w_km1);
   free_vector(grad);
   free_vector(Gw);
}
