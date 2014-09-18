#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <limits.h>
#include <lapacke.h>
#include "nrutil.h"
#include "fista.h"

#define _DEBUG_
#define INF INT_MAX
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static char *line = NULL;
static int max_line_len;

void exit_input_error(int line_num)
{
   fprintf(stderr,"Wrong input format at line %d\n", line_num);
   exit(1);
}

void standarize(problem *prob, int is_train, double *mean, double *var)
{
   int i, j;
   int n = prob->n, dim = prob->dim;

   if (is_train)
   {
      dmcolmean(prob->X, n, dim, mean, var);
      dvmean(prob->y, n, &(mean[dim+1]), &(var[dim+1]));
   }

   for (i=1; i <= n; i++)
   {
      for (j=1; j <= dim; j++)
         prob->X[i][j] = (prob->X[i][j]-mean[j]) / sqrt(var[j]);

      prob->y[i] = (prob->y[i]-mean[dim+1]) / sqrt(var[dim+1]);
   }
}

double mse(double *y, int n, double *y_hat)
{
   int i;
   double sum = 0;

   for(i=1; i<=n; i++)
      sum += SQR(y_hat[i] - y[i]);

   return sum/n;
}

double mae(double *y, int n, double *y_hat)
{
   int i;
   double sum = 0;

   for(i=1; i<=n; i++)
      sum += fabs(y_hat[i] - y[i]);

   return sum/n;
}

double rss(problem *prob, double *w)
{
   int i;
   double sum = 0.0, *y_hat = dvector(1, prob->n);

   fista_predict(prob, w, y_hat);

/* for(i=1; i<prob->n; i++)
      y_hat[i] = dvdot(prob->X[i], prob->dim, w);*/

   for(i=1; i<=prob->n; i++)
      sum += SQR(y_hat[i] - prob->y[i]);
   free_dvector(y_hat, 1, prob->n);

   return sum;
}

void soft_thresholding(double *x, int n, double lambda)
{
   int i;

   #pragma omp parallel for
   for(i=1; i<=n; i++)
         x[i] = DMAX(fabs(x[i])-lambda,0)*SIGN(x[i]);
}

#ifdef _BLAS_
double eigs(double **X, int n)
{
   double y, z;
   float abstol, vl, vu;
   int info, m, ifail;
   double **A = dmatrix(1, n, 1, n);

   dmcopy(X, n, n, A);
   info = LAPACKE_dsyevx(LAPACK_ROW_MAJOR, 'N', 'I', 'U', n, &A[1][1], n,
         vl, vu, n, n, abstol, &m, &y, &z, 1, &ifail);

   if (info)
   {
      fprintf(stderr, "Eigenvalues failed to converge\n");
      exit(1);
   }

   free_dmatrix(A, 1, n, 1, n);
   return y;
}
#else
double eigs(double **X, int n)
{
   double max, *e, *d, **A;

   e = dvector(1, n);
   d = dvector(1, n);
   A = dmatrix(1, n, 1, n);

   dmcopy(X, n, n, A);

   tred2(A, n, d, e);
   tqli(d, e, n, A);
   dvmax(d, n, &max);

   free_dvector(e, 1, n);
   free_dvector(d, 1, n);
   free_dmatrix(A, 1, n, 1, n);

   return max;
}
#endif

void fista_predict(problem *prob, double *w, double *y)
{
   dmvmult(prob->X, prob->n, prob->dim, w, prob->dim, y);
}

#define EPS 1.0e-10
void fista(problem *prob, double *w, double lambda_1, double lambda_2,
           double tol, int verbose_flag, int *iter, double *fret)
{
   double **G;
   double f, prev_f;
   double *w_k, *w_km1;
   double *v_k, *c, *grad;
   double L, t_k = 1, t_km1 = t_k;
   int keep_going = 1, max_iter = INF;
   /* not in use until other stopping criterions (apart from iterations)
    * are implemented */
   /*stop_crit_t stopping_criterion = OBJECTIVE;*/

   /* alloc memory */
   v_k = dvector(1, prob->dim);
   w_k = dvector(1, prob->dim);
   w_km1 = dvector(1, prob->dim);

   c = dvector(1, prob->dim);
   G = dmatrix(1, prob->dim, 1, prob->dim);
   grad = dvector(1, prob->dim);

   /* G = trans(X)*X */
   dmtransmult (prob->X, prob->n, prob->dim, G);

   /* c = trans(X)*y */
   dmvtransmult(prob->X, prob->n, prob->dim, prob->y, prob->n, c);

   /* scale lambda, compute Lipstchiz constant */
   lambda_1 = (lambda_1 * prob->n); /* / prob->dim;*/
   lambda_2 = (lambda_2 * prob->n); /* / prob->dim;*/
   L = eigs(G, prob->dim);

    /* v_k = w */
   dvcopy(w, prob->dim, v_k);

   /* w_k = v_k */
   dvcopy(w, prob->dim, w_k);

   /* compute value of f at w_k */
   f = 0.5*rss(prob, w_k) + lambda_1*dvnorm(w_k, prob->dim, 1) \
                          + lambda_2*SQR(dvnorm(w_k, prob->dim, 2));

   /* if tol == 0, we want to stop by iterations */
   if (!tol)
      max_iter = (*iter);

   (*iter) = 0;
   while (keep_going && (*iter) < max_iter)
   {
      /* w_km1 = w_k */
      dvcopy(w_k, prob->dim, w_km1);

      /* t_km1 = t_k */
      t_km1 = t_k;

      /* prev_f = f */
      prev_f = f;

      /* grad(v_k) = G*v_k - c + lambda_2*v_k */
#ifdef _BLAS_
      dsymvmult(G, prob->dim, 1, v_k, grad);
      daxpy(c, prob->dim, -1, grad);
      daxpy(v_k, prob->dim, lambda_2, grad);
#else
      dmvmult(G, prob->dim, prob->dim, v_k, prob->dim, grad);
      dvsub(grad, prob->dim, c, grad);
      dvpiv(v_k, prob->dim, lambda_2, grad, grad);
#endif

      /* w_k = soft_thresholding(v_k - (1/L)*grad(v_k), lambda/L) */
#ifdef _BLAS_
      dvcopy(v_k, prob->dim, w_k);
      daxpy(grad, prob->dim, -(1/L), w_k);
#else
      dvsmy(grad, prob->dim, (1/L), w_k);
      dvsub(v_k, prob->dim, w_k, w_k);
#endif
      soft_thresholding(w_k, prob->dim, lambda_1/L);

      /* t_k = (1 + sqrt(1 + 4*t_km1^2)) / 2 */
      t_k = 0.5*(1+sqrt(1+4*SQR(t_km1)));

      /* v_k = w_k + ((t_km1-1)/t_k)*(w_k-w_km1) */
#ifdef _BLAS_
      dvcopy(w_k, prob->dim, v_k);
      dscal(v_k, prob->dim, 1+(t_km1-1)/t_k);
      daxpy(w_km1, prob->dim, -(t_km1-1)/t_k, v_k);
#else
      dvsub(w_k, prob->dim, w_km1, v_k);
      dvpiv(v_k, prob->dim, (t_km1-1)/t_k, w_k, v_k);
#endif
      /* compute value of f at w_k */
      f = 0.5*rss(prob, w_k) + lambda_1*dvnorm(w_k, prob->dim, 1) \
                             + lambda_2*SQR(dvnorm(w_k, prob->dim, 2));

      if (tol)
         keep_going = (2.0*fabs(f - prev_f) > tol*(fabs(f)+fabs(prev_f)+EPS));
      /* if (stopping_criterion == OBJECTIVE)
       *    keep_going = (fabs(f - prev_f) > tol);*/
      (*iter)++;

#ifdef _DEBUG_
      if (verbose_flag)
         dvprint(stderr, w_k, prob->dim);
#endif
   }

   (*fret) = f;
   dvcopy(w_k, prob->dim, w);

   free_dvector(v_k, 1, prob->dim);
   free_dvector(w_k, 1, prob->dim);
   free_dvector(w_km1, 1, prob->dim);

   free_dvector(c, 1, prob->dim);
   free_dmatrix(G, 1, prob->dim, 1, prob->dim);

   free_dvector(grad, 1, prob->dim);
}
#undef EPS


void fista_backtrack(problem *prob, double *w, double lambda_1, double lambda_2,
                     double ftol, int *iter, double *fret)
{
   double **G;
   double *w_k, *w_km1, *w_kp1;
   double *v_k, *s_k, *c, *grad, *temp;
   double t_k = 1, t_km1 = 1, t_kp1 = 1;
   double prev_f = 0, f = 0, temp1 = 0, temp2 = 0;
   double alpha, L, L0 = 1, beta = 0.95, eta = 1.5;
   int keep_going = 1, stop_backtrack = 0, max_iter = 10000;
   stop_crit_t stopping_criterion = OBJECTIVE;

   lambda_1 = (lambda_1 * prob->n) / prob->dim;
   lambda_2 = (lambda_2 * prob->n) / prob->dim;

   v_k = dvector(1, prob->dim);
   w_k = dvector(1, prob->dim);
   s_k = dvector(1, prob->dim);
   w_km1 = dvector(1, prob->dim);
   w_kp1 = dvector(1, prob->dim);
   c = dvector(1, prob->dim);
   G = dmatrix(1, prob->dim, 1, prob->dim);

   grad = dvector(1, prob->dim);
   temp = dvector(1, prob->dim);

   /* w_k = w */
   dvcopy(w, prob->dim, w_k);

   /* G = trans(X)*X */
   dmtransmult(prob->X, prob->n, prob->dim, G);

   /* c = trans(X)*y */
   dmvtransmult(prob->X, prob->n, prob->dim, prob->y, prob->n, c);

   /* L = L0 */
   L = L0;

   /* alpha = 0.5*L0*norm(c, inf) */
   alpha = 0.5*L0*dvnorm(c, prob->dim, INF);

   /* w_km1 = w_k */
   dvcopy(w_k, prob->dim, w_km1);

   /* compute value of f at w_k */
   f = 0.5*rss(prob, w_k) + lambda_1*dvnorm(w_k, prob->dim, 1) \
                          + lambda_2*SQR(dvnorm(w_k, prob->dim, 2));

   (*iter) = 0;
   while (keep_going && (*iter) < max_iter)
   {
      ++(*iter);

#ifdef _DEBUG_
      fprintf(stderr, "%d %g %g\n", *iter, rss(prob, w_k), lambda_1*dvnorm(w_k, prob->dim, 1));
#endif

      /* v_k = w_k + ((t_km1-1)/t_k)*(w_k-w_km1) */
#ifdef _BLAS_
      dvcopy(w_k, prob->dim, v_k);
      dscal(v_k, prob->dim, 1+(t_km1-1)/t_k);
      daxpy(w_km1, prob->dim, -(t_km1-1)/t_k, v_k);
#else
      dvsub(w_k, prob->dim, w_km1, v_k);
      dvpiv(v_k, prob->dim, (t_km1-1)/t_k, w_k, v_k);
#endif

      /* grad(v_k) = G*v_k - c + lambda_2*v_k */
#ifdef _BLAS_
      dsymvmult(G, prob->dim, 1, v_k, grad);
      daxpy(c, prob->dim, -1, grad);
      daxpy(v_k, prob->dim, lambda_2, grad);
#else
      dmvmult(G, prob->dim, prob->dim, v_k, prob->dim, grad);
      dvsub(grad, prob->dim, c, grad);
      dvpiv(v_k, prob->dim, lambda_2, grad, grad);
#endif

      while (!stop_backtrack)
      {
         /* w_kp1 = soft_thresholding(v_k - (1/L)*grad(v_k), alpha/L) */
#ifdef _BLAS_
         dvcopy(v_k, prob->dim, w_kp1);
         daxpy(grad, prob->dim, -(1/L), w_kp1);
#else
         dvsmy(grad, prob->dim, (1/L), w_kp1);
         dvsub(v_k, prob->dim, w_kp1, w_kp1);
#endif
         soft_thresholding(w_kp1, prob->dim, alpha/L);

         /* temp  = w_kp1 - v_k
            temp1 = f(w_kp1)
            temp2 = f(v_k) + <temp, grad(v_k)> + (L/2)*||temp||_2^2 */
#ifdef _BLAS_
         dvcopy(w_kp1, prob->dim, temp);
         daxpy(v_k, prob->dim, -1, temp);
#else
         dvsub(w_kp1, prob->dim, v_k, temp);
#endif
         temp1 = 0.5*rss(prob, w_kp1);
         temp2 = 0.5*rss(prob, v_k) \
                 + dvdot(temp, prob->dim, grad) \
                 + (L/2)*SQR(dvnorm(temp, prob->dim, 2));

         if (temp1 <= temp2)
            stop_backtrack = 1;
         else
            L *= eta;
      }

      stop_backtrack = 0;

      /* alpha = max(beta*alpha, lambda) */
      alpha = DMAX(beta*alpha, lambda_1);

      /* stopping criterion */
      switch (stopping_criterion)
      {
         case SUBGRADIENT:
            /* sk = L*(vk-wkp1) + G*(wkp1-vk); */
#ifdef _BLAS_
            dsymvmult(G, prob->dim, 1, temp, s_k);
            daxpy(temp, prob->dim, -L, s_k);
#else
            dmvmult(G, prob->dim, prob->dim, temp, prob->dim, s_k);
            dvsmy(temp, prob->dim, -L, temp);
            dvsub(temp, prob->dim, s_k, s_k);
#endif
            keep_going = (dv2norm(s_k, prob->dim) > ftol*L*DMAX(1,dv2norm(w_kp1, prob->dim)));
            break;
         case OBJECTIVE:
            prev_f = f;
            f = 0.5*rss(prob, w_kp1) + lambda_1*dvnorm(w_k, prob->dim, 1) \
                                     + lambda_2*SQR(dvnorm(w_k, prob->dim, 2));
            keep_going = (fabs(f - prev_f)/(prev_f) > ftol);
            break;
         default: return;
      }

      /* t_kp1 = (1 + sqrt(1 + 4*t_k^2)) / 2 */
      t_kp1 = 0.5*(1+sqrt(1+4*SQR(t_k)));

      /* update t_k and w_k values */
      t_km1 = t_k;
      t_k = t_kp1;

      /*w_km1 = w_k;
      w_k = w_kp1;*/
      dvcopy(w_k, prob->dim, w_km1);
      dvcopy(w_kp1, prob->dim, w_k);
   }

   (*fret) = f;
   dvcopy(w_k, prob->dim, w);

   free_dvector(s_k, 1, prob->dim);
   free_dvector(v_k, 1, prob->dim);
   free_dvector(w_k, 1, prob->dim);
   free_dvector(w_km1, 1, prob->dim);
   free_dvector(w_kp1, 1, prob->dim);

   free_dvector(c, 1, prob->dim);
   free_dmatrix(G, 1, prob->dim, 1, prob->dim);

   free_dvector(grad, 1, prob->dim);
   free_dvector(temp, 1, prob->dim);
}


void fista_nocov(problem *prob, double *w, double lambda_1, double lambda_2, 
                 double ftol, int *iter, double *fret)
{
   double **G;
   double *w_k, *w_km1, *w_kp1;
   double *v_k, *s_k, *c, *grad, *temp;
   double t_k = 1, t_km1 = 1, t_kp1 = 1;
   double prev_f = 0, f = 0, temp1 = 0, temp2 = 0;
   double alpha, L, L0 = 1, beta = 0.95, eta = 1.5;
   int keep_going = 1, stop_backtrack = 0, max_iter = 100000;
   stop_crit_t stopping_criterion = OBJECTIVE;

   lambda_1 = (lambda_1 * prob->n) / prob->dim;
   lambda_2 = (lambda_2 * prob->n) / prob->dim;

   v_k = dvector(1, prob->dim);
   w_k = dvector(1, prob->dim);
   s_k = dvector(1, prob->dim);
   w_km1 = dvector(1, prob->dim);
   w_kp1 = dvector(1, prob->dim);

   c = dvector(1, prob->n);

   grad = dvector(1, prob->dim);
   temp = dvector(1, prob->dim);

   dvcopy(w, prob->dim, w_k);

   /* c = trans(X)*y */
   dmvtransmult(prob->X, prob->n, prob->dim, prob->y, prob->n, temp);

   /* L = L0 */
   L = L0;

   /* alpha = 0.5*L0*norm(c, inf) */
   alpha = 0.5*L0*dvnorm(temp, prob->dim, INF);

   /* w_km1 = w_k */
   dvcopy(w_k, prob->dim, w_km1);

   /* compute value of f at w_k */
   f = 0.5*rss(prob, w_k) + lambda_1*dvnorm(w_k, prob->dim, 1) \
                          + lambda_2*SQR(dvnorm(w_k, prob->dim, 2));

   (*iter) = 0;
   while (keep_going && (*iter) < max_iter)
   {
      ++(*iter);

      /* v_k = w_k + ((t_km1-1)/t_k)*(w_k-w_km1) */
#ifdef _BLAS_
      dvcopy(w_k, prob->dim, v_k);
      dscal(v_k, prob->dim, 1+(t_km1-1)/t_k);
      daxpy(w_km1, prob->dim, -(t_km1-1)/t_k, v_k);
#else
      dvsub(w_k, prob->dim, w_km1, v_k);
      dvpiv(v_k, prob->dim, (t_km1-1)/t_k, w_k, v_k);
#endif

      /* grad(v_k) = X^t(X*v_k - y) + lambda_2*v_k*/
      dmvmult(prob->X, prob->n, prob->dim, v_k, prob->dim, c);
#ifdef _BLAS_
      daxpy(prob->y, prob->n, -1, c);
#else
      dvsub(c, prob->n, prob->y, c);
#endif
      dmvtransmult(prob->X, prob->n, prob->dim, c, prob->n, grad);
#ifdef _BLAS_
      daxpy(v_k, prob->dim, lambda_2, grad);
#else
      dvpiv(v_k, prob->dim, lambda_2, grad, grad);
#endif

      while (!stop_backtrack)
      {
         /* w_kp1 = soft_thresholding(v_k - (1/L)*grad(v_k), alpha/L) */
#ifdef _BLAS_
         dvcopy(v_k, prob->dim, w_kp1);
         daxpy(grad, prob->dim, -(1/L), w_kp1);
#else
         dvsmy(grad, prob->dim, (1/L), w_kp1);
         dvsub(v_k, prob->dim, w_kp1, w_kp1);
#endif
         soft_thresholding(w_kp1, prob->dim, alpha/L);

         /* temp  = w_kp1 - v_k
            temp1 = f(w_kp1)
            temp2 = f(v_k) + <temp, grad(v_k)> + (L/2)*||temp||_2^2 */
#ifdef _BLAS_
         dvcopy(w_kp1, prob->dim, temp);
         daxpy(v_k, prob->dim, -1, temp);
#else
         dvsub(w_kp1, prob->dim, v_k, temp);
#endif
         temp1 = 0.5*rss(prob, w_kp1);
         temp2 = 0.5*rss(prob, v_k) \
                 + dvdot(temp, prob->dim, grad) \
                 + (L/2)*SQR(dvnorm(temp, prob->dim, 2));

#ifdef _DEBUG_
         dvprint(stderr, w_kp1, prob->dim);
#endif

         if (temp1 <= temp2)
            stop_backtrack = 1;
         else
            L *= eta;
      }

      stop_backtrack = 0;

      /* alpha = max(beta*alpha, lambda_1) */
      alpha = DMAX(beta*alpha, lambda_1);

      /* stopping criterion */
      switch (stopping_criterion)
      {
         case OBJECTIVE:
            prev_f = f;
            f = 0.5*rss(prob, w_kp1) + lambda_1*dvnorm(w_k, prob->dim, 1) \
                                     + lambda_2*SQR(dvnorm(w_k, prob->dim, 2));
            keep_going = (fabs(f - prev_f)/(prev_f) > ftol);
            break;
         default: return;
      }

      /* t_kp1 = (1 + sqrt(1 + 4*t_k^2)) / 2 */
      t_kp1 = 0.5*(1+sqrt(1+4*SQR(t_k)));

      /* update t_k and w_k values */
      t_km1 = t_k;
      t_k = t_kp1;

      /*w_km1 = w_k;
      w_k = w_kp1;*/
      dvcopy(w_k, prob->dim, w_km1);
      dvcopy(w_kp1, prob->dim, w_k);
   }

   (*fret) = f;
   dvcopy(w_k, prob->dim, w);

   free_dvector(v_k, 1, prob->dim);
   free_dvector(w_k, 1, prob->dim);
   free_dvector(w_km1, 1, prob->dim);
   free_dvector(w_kp1, 1, prob->dim);
   free_dvector(c, 1, prob->n);

   free_dvector(grad, 1, prob->dim);
   free_dvector(temp, 1, prob->dim);
}

inline void swap(int *x, int *y)
{
   int temp;

   temp = *x;
   *x = *y;
   *y = temp;
}

#undef _DEBUG_
void cross_validation(problem *prob, double *w, double lambda_1, double lambda_2, int nr_fold, double *target)
{
   int i, iter;
   int n = prob->n;
   int *fold_start = Malloc(int,nr_fold+1);
   int *perm = Malloc(int,n);
   double fret, tol = 1e-9;

   srand(time(NULL));

   for(i=0;i<n;i++) perm[i]=i+1;
   for(i=0;i<n;i++)
   {
      int j = i+rand()%(n-i);
      swap(&perm[i],&perm[j]);
   }

   for(i=0;i<=nr_fold;i++)
      fold_start[i]=i*n/nr_fold;

   for(i=0;i<nr_fold;i++)
   {
      int begin = fold_start[i];
      int end = fold_start[i+1];
      int j,k;
      problem subprob;

      subprob.dim = prob->dim;
      subprob.n = n-(end-begin);
#ifdef _BLAS_
      subprob.X = dmatrix(1, subprob.n, 1, subprob.dim);
#else
      subprob.X = pvector(1, subprob.n);
#endif
      subprob.y = dvector(1, subprob.n);

      /* nr_fold-1 partitions are used for training */
      k=1;
      for(j=0;j<begin;j++)
      {
#ifdef _BLAS_
         dvcopy(prob->X[perm[j]], subprob.dim, subprob.X[k]);
#else
         subprob.X[k] = prob->X[perm[j]];
#endif
         subprob.y[k] = prob->y[perm[j]];
         ++k;
      }
      for(j=end;j<n;j++)
      {
#ifdef _BLAS_
         dvcopy(prob->X[perm[j]], subprob.dim, subprob.X[k]);
#else
         subprob.X[k] = prob->X[perm[j]];
#endif
         subprob.y[k] = prob->y[perm[j]];
         ++k;
      }

      fista(&subprob, w, lambda_1, lambda_2, tol, 0, &iter, &fret);

      /* the other partition is used for test */
      for(j=begin;j<end;j++)
         target[perm[j]] = dvdot(prob->X[perm[j]], prob->dim, w);

#ifdef _BLAS_
      free_dmatrix(subprob.X, 1, subprob.n, 1, subprob.dim);
#else
      free_pvector(subprob.X,1,subprob.n);
#endif
      free_dvector(subprob.y,1,subprob.n);
   }
   free(fold_start);
   free(perm);

}

/************************************************************************
 * readline
 *
 * Reads line from input file. If the line is bigger than max_line_len,
 * more memory is allocated
 *
 ***********************************************************************/
static char* readline(FILE *input)
{
   int len;

   if(fgets(line,max_line_len,input) == NULL)
      return NULL;

   while(strrchr(line,'\n') == NULL)
   {
      max_line_len *= 2;
      line = (char *) realloc(line,max_line_len);
      len = (int) strlen(line);
      if(fgets(line+len,max_line_len-len,input) == NULL)
         break;
   }
   return line;
}

/************************************************************************
 * write_problem
 *
 * Writes problem to file
 *
 ***********************************************************************/
void write_problem(problem *prob, char *filename)
{
   int i, j;
   FILE *f = fopen(filename, "w");

   if (f == NULL) return;

   for(i=1; i<=prob->n; i++)
   {
      fprintf(f, "%g", prob->y[i]);
      for(j=1; j<=prob->dim; j++)
         fprintf(f, " %g", prob->X[i][j]);
      fprintf(f, "\n");
   }

   fclose(f);
}

/************************************************************************
 * read_problem
 *
 * Reads problem from file
 *
 ***********************************************************************/
problem* read_problem(const char *filename)
{
   problem *prob;
   int elements, i, j;
   FILE *fp = fopen(filename,"r");
   char *endptr;
   char *val, *label;

   if(fp == NULL)
   {
      fprintf(stderr,"can't open input file %s\n",filename);
      exit(1);
   }

   prob = Malloc(problem, 1);

   prob->n = 0;
   prob->dim = 0;

   max_line_len = 1024;
   line = Malloc(char,max_line_len);
   while(readline(fp)!=NULL)
   {
      char *p = strtok(line," \t"); // label
      elements = 0;
      // features
      while(1)
      {
         p = strtok(NULL," \t");
         if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
            break;
         ++elements;
      }

      if (prob->n && elements != prob->dim)
         exit_input_error(prob->n);

      prob->dim = elements;
      ++prob->n;
   }
   rewind(fp);

   prob->y = dvector(1, prob->n);
   prob->X = dmatrix(1, prob->n, 1, prob->dim);

   for(i=1;i<=prob->n;i++)
   {
      readline(fp);
      label = strtok(line," \t\n");
      if(label == NULL) // empty line
         exit_input_error(i+1);

      prob->y[i] = strtod(label,&endptr);
      if(endptr == label || *endptr != '\0')
         exit_input_error(i+1);

      j=1;
      while(1)
      {
         val = strtok(NULL," \t");

         if(val == NULL)
            break;

         errno = 0;
         prob->X[i][j] = strtod(val,&endptr);
         if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
            exit_input_error(i+1);

         ++j;
      }
   }

   free(line);
   fclose(fp);
   return prob;
}
