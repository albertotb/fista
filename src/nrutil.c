/***********************************************************************
   nrutil.c

   Numerica Recipes functions

***********************************************************************/
#if defined(__STDC__) || defined(ANSI) || defined(NRANSI) /* ANSI */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <cblas.h>
#include "nrutil.h"

#define INF INT_MAX
#define NR_END 1
#define FREE_ARG char*
#define EPSILON_ZERO 1e-15
#define IS_ZERO(a) (((a) < EPSILON_ZERO && (a) > -EPSILON_ZERO) ? 1 : 0)

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
        fprintf(stdout,"Numerical Recipes run-time error...\n");
        fprintf(stdout,"%s\n",error_text);
        fprintf(stderr,"...now exiting to system...\n");
        exit(1);
}

double *vector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
        double *v;

        v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
        if (!v) nrerror("allocation failure in vector()");
        return v-nl+NR_END;
}

double **pvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
        double **v;

        v=(double **)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double *)));
        if (!v) nrerror("allocation failure in pvector()");
        return v-nl+NR_END;
}

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
        int *v;

        v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
        if (!v) nrerror("allocation failure in ivector()");
        return v-nl+NR_END;
}

unsigned char *cvector(long nl, long nh)
/* allocate an unsigned char vector with subscript range v[nl..nh] */
{
        unsigned char *v;

        v=(unsigned char *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(unsigned char)));
        if (!v) nrerror("allocation failure in cvector()");
        return v-nl+NR_END;
}

unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
        unsigned long *v;

        v=(unsigned long *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(long)));
        if (!v) nrerror("allocation failure in lvector()");
        return v-nl+NR_END;
}

double *dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
        double *v;

        v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
        if (!v) nrerror("allocation failure in dvector()");
        return v-nl+NR_END;
}

double **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
        long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
        double **m;

        /* allocate pointers to rows */
        m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
        if (!m) nrerror("allocation failure 1 in matrix()");
        m += NR_END;
        m -= nrl;

        /* allocate rows and set pointers to them */
        m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
        if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
        m[nrl] += NR_END;
        m[nrl] -= ncl;

        for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

        /* return pointer to array of pointers to rows */
        return m;
}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
        long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
        double **m;

        /* allocate pointers to rows */
        m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
        if (!m) nrerror("allocation failure 1 in matrix()");
        m += NR_END;
        m -= nrl;

        /* allocate rows and set pointers to them */
        m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
        if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
        m[nrl] += NR_END;
        m[nrl] -= ncl;

        for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

        /* return pointer to array of pointers to rows */
        return m;
}

int **imatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
        long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
        int **m;

        /* allocate pointers to rows */
        m=(int **) malloc((size_t)((nrow+NR_END)*sizeof(int*)));
        if (!m) nrerror("allocation failure 1 in matrix()");
        m += NR_END;
        m -= nrl;


        /* allocate rows and set pointers to them */
        m[nrl]=(int *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
        if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
        m[nrl] += NR_END;
        m[nrl] -= ncl;

        for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

        /* return pointer to array of pointers to rows */
        return m;
}

double **submatrix(double **a, long oldrl, long oldrh, long oldcl, long oldch,
        long newrl, long newcl)
/* point a submatrix [newrl..][newcl..] to a[oldrl..oldrh][oldcl..oldch] */
{
        long i,j,nrow=oldrh-oldrl+1,ncol=oldcl-newcl;
        double **m;

        /* allocate array of pointers to rows */
        m=(double **) malloc((size_t) ((nrow+NR_END)*sizeof(double*)));
        if (!m) nrerror("allocation failure in submatrix()");
        m += NR_END;
        m -= newrl;

        /* set pointers to rows */
        for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+ncol;

        /* return pointer to array of pointers to rows */
        return m;
}

double **convert_matrix(double *a, long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix m[nrl..nrh][ncl..nch] that points to the matrix
declared in the standard C manner as a[nrow][ncol], where nrow=nrh-nrl+1
and ncol=nch-ncl+1. The routine should be called with the address
&a[0][0] as the first argument. */
{
        long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1;
        double **m;

        /* allocate pointers to rows */
        m=(double **) malloc((size_t) ((nrow+NR_END)*sizeof(double*)));
        if (!m) nrerror("allocation failure in convert_matrix()");
        m += NR_END;
        m -= nrl;

        /* set pointers to rows */
        m[nrl]=a-ncl;
        for(i=1,j=nrl+1;i<nrow;i++,j++) m[j]=m[j-1]+ncol;
        /* return pointer to array of pointers to rows */
        return m;
}

double ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate a double 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
        long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
        double ***t;

        /* allocate pointers to pointers to rows */
        t=(double ***) malloc((size_t)((nrow+NR_END)*sizeof(double**)));
        if (!t) nrerror("allocation failure 1 in f3tensor()");
        t += NR_END;
        t -= nrl;

        /* allocate pointers to rows and set pointers to them */
        t[nrl]=(double **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double*)));
        if (!t[nrl]) nrerror("allocation failure 2 in f3tensor()");
        t[nrl] += NR_END;
        t[nrl] -= ncl;

        /* allocate rows and set pointers to them */
        t[nrl][ncl]=(double *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(double)));
        if (!t[nrl][ncl]) nrerror("allocation failure 3 in f3tensor()");
        t[nrl][ncl] += NR_END;
        t[nrl][ncl] -= ndl;

        for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
        for(i=nrl+1;i<=nrh;i++) {
                t[i]=t[i-1]+ncol;
                t[i][ncl]=t[i-1][ncl]+ncol*ndep;
                for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
        }

        /* return pointer to array of pointers to rows */
        return t;
}

void dmprint(FILE *f, double **a, int a_rows, int a_cols)
{
   int i, j;

   for (i=1; i <= a_rows; i++)
   {
      for (j=1; j < a_cols; j++)
         fprintf(f, "%g ", a[i][j]);
      fprintf(f, "%g\n", a[i][a_cols]);
   }
}

void dvprint(FILE *f, double *a, int a_els)
{
   int i;

   for (i=1; i < a_els; i++)
      fprintf(f, "%.15f ", a[i]);
   fprintf(f, "%.15f\n",a[a_els]);
}

void dmadd( double **a, int a_rows, int a_cols, double **b, double **y)
/* add two matrices a, b, result in y. y can be same as a or b */
{
   int i, j;

   for (i=0; i<a_rows; i++)
      for (j=0; j<a_cols; j++) {
         y[i][j] = a[i][j] + b[i][j];
      }
}

void dmmult(double **a, int a_rows, int a_cols,
            double **b, int b_rows, int b_cols, double **y)
/* multiply two matrices a, b, result in y. y must not be same as a or b */
{
   int i, j, k;
   double sum;

   if (a_cols != b_rows)
   {
      fprintf(stderr,"a_cols <> b_rows (%d,%d): dmmult\n", a_cols, b_rows);
      exit(1);
   }

   for (i=0; i < a_rows; i++)
      for (j=0; j < b_cols; j++)
      {
         sum = 0.0;
         for ( k=0; k < a_cols; k++ ) sum += a[i][k]*b[k][j];
         y[i][j] = sum;
      }
}

#ifdef _BLAS_
void dmtransmult(double **a, int a_rows, int a_cols, double **y)
{
   cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
         a_cols, a_rows, 1, &a[1][1], a_cols, 0, &y[1][1], a_cols);
}
#else
void dmtransmult (double **a, int a_rows, int a_cols, double **y)
{
   int i, j, k;
   double sum;

   for (i=1; i <= a_cols; i++)
      for (j=i; j <= a_cols; j++)
      {
         //printf("y[%d][%d]=", i,j);
         sum = 0.0;
         for (k=1; k <= a_rows; k++)
         {
           // if (k == a_rows)
           //    printf("a[%d][%d]a[%d][%d]\n",k,i,k,j);
           // else
           //    printf("a[%d][%d]a[%d][%d]+",k,i,k,j);
            sum += a[k][i] * a[k][j];
         }

        y[i][j] = sum;
      }
   /* Copiamos lo que nos falta */
   for (i = 1; i <= a_cols; i++)
      for (j = 1; j < i; j++)
         y[i][j] = y[j][i];
}
#endif

#ifdef _BLAS_
void dsymvmult(double **a, int n, double alpha, double *b,  double *y)
{
   cblas_dsymv(CblasRowMajor, CblasUpper, n, alpha, &a[1][1], n, &b[1], 1, 0, &y[1], 1);
}
#endif

#ifdef _BLAS_
void dmvmult(double **a, int a_rows, int a_cols,
             double *b, int b_els, double *y)
{
   cblas_dgemv(CblasRowMajor, CblasNoTrans, a_rows, a_cols, 1, &a[1][1],
         a_cols, &b[1], 1, 0, &y[1], 1);
}
#else
void dmvmult(double **a, int a_rows, int a_cols,
             double *b, int b_els, double *y)
/* multiply a matrix a by vector b, result in y. y can be same as b */
{
   int i, k;
   double sum;

   if (a_cols != b_els)
   {
      fprintf(stderr,"a_cols <> b_els (%d,%d): dmvmult\n", a_cols, b_els);
      exit(1);
   }

   for (i=1; i <= a_rows; i++) {
      sum = 0.0;
      for (k=1; k <= a_cols; k++) sum += a[i][k]*b[k];
      y[i] = sum;
   }
}
#endif

#ifdef _BLAS_
void dmvtransmult(double **a, int a_rows, int a_cols,
             double *b, int b_els, double *y)
{
   cblas_dgemv(CblasRowMajor, CblasTrans, a_rows, a_cols, 1, &a[1][1], 
         a_cols, &b[1], 1, 0, &y[1], 1);
}
#else
void dmvtransmult(double **a, int a_rows, int a_cols,
                  double *b, int b_els, double *y)
/* multiply a matrix a' by vector b, result in y. y can be same as b */
{
   int j, k;
   double sum;

   if (a_rows != b_els)
   {
      fprintf(stderr,"a_rows <> b_els (%d,%d): dmvtransmult\n", a_rows, b_els);
      exit(1);
   }

   for (j=1; j <= a_cols; j++) {
      sum = 0.0;
      for (k=1; k <= a_rows; k++) sum += a[k][j]*b[k];
      y[j] = sum;
   }
}
#endif

void dmtranspose(double **a, int a_rows, int a_cols, double **y)
/* transpose matrix a, result in y. y must not be same as a */
{
   int i, j;

   for (i=1; i <= a_rows; i++)
      for (j=1; j <= a_cols; j++)
         y[j][i] = a[i][j];
}

void dvmax(double *a, int a_els, double *max)
{
   int i;

   (*max) = DBL_MIN;

   for (i=1; i <= a_els; i++)
      if (a[i] > (*max))
         (*max) = a[i];
}

void dvset(double *a, int a_els, double val)
{
   int i;

   for (i=1; i <= a_els; i++)
      a[i] = val;
}

#ifdef _BLAS_
void dvcopy(double *a, int a_els, double *y)
{
   cblas_dcopy(a_els, &a[1], 1, &y[1], 1);
}
#else
void dvcopy(double *a, int a_els, double *y)
{
   int i;

   for (i=1; i <= a_els; i++)
      y[i] = a[i];
}
#endif

#ifdef _BLAS_
void dmcopy(double **a, int a_rows, int a_cols, double **y)
{
   int i;
   /* bugged!! does not copy first value */
   /*dvcopy(&a[1][1], a_rows*a_cols, &y[1][1]);*/
   for(i=1; i<=a_rows; i++)
      dvcopy(a[i], a_cols, y[i]);

}
#else
void dmcopy(double **a, int a_rows, int a_cols, double **y)
{
   int i;
   for(i=1; i<=a_rows; i++)
      dvcopy(a[i], a_cols, y[i]);
}
#endif

#ifdef _BLAS_
void dscal(double *a, int a_els, double r)
{
   cblas_dscal(a_els, r, &a[1], 1);
}

/* y = ax + y */
void daxpy(double *a, int a_els, double r, double *y)
{
   cblas_daxpy(a_els, r, &a[1], 1, &y[1], 1);
}
#endif

void dvsmy( double *a, int a_els, double r, double *y)
{
   int i;

   for (i=1; i <= a_els; i++)
      y[i] = a[i] * r;
}

void dvpiv( double *a, int a_els, double r, double *b, double *y)
{
   int i;

   for (i=1; i <= a_els; i++)
      y[i] = a[i] * r + b[i];
}

void dvadd(double *a, int a_els, double *b, double *y)
{
   dvpiv(a, a_els, 1, b, y);
}

void dvsub(double *a, int a_els, double *b, double *y)
{
   dvpiv(b, a_els, -1, a, y);
}

#ifdef _BLAS_
double dvdot(double *a, int a_els, double *b)
{
   return cblas_ddot(a_els, &a[1], 1, &b[1], 1);
}
#else
double dvdot(double *a, int a_els, double *b)
{
   int i;
   double y;

   y = 0.0;
   for (i=1; i <= a_els; i++) {
      y += a[i] * b[i];
   }
   return y;
}
#endif

double dvinfnorm(double *a, int a_els)
{
   int i;
   double max = 0.0;

   for (i=1; i <= a_els; i++)
      if (fabs(a[i]) > max)
         max = fabs(a[i]);

   return max;
}

#ifdef _BLAS_
double dv2norm(double *a, int a_els)
{
   return cblas_dnrm2(a_els, &a[1], 1);
}
#else
double dv2norm(double *a, int a_els)
{
   return sqrt(dvdot(a, a_els, a));
}
#endif

#ifdef _BLAS_
double dv1norm(double *a, int a_els)
{
   return cblas_dasum(a_els, &a[1], 1);
}
#else
double dv1norm(double *a, int a_els)
{
   int i;
   double y = 0.0;

   for (i=1; i <= a_els; i++)
      y += fabs(a[i]);

   return y;
}
#endif

double dvpnorm(double *a, int a_els, int p)
{
   int i;
   double y = 0.0;

   for (i=1; i <= a_els; i++)
      y += pow(fabs(a[i]), p);
   return pow(y, 1/p);
}

double dvnorm(double *a, int a_els, int p)
{
   switch(p)
   {
      case 1:   return dv1norm(a, a_els);
      case 2:   return dv2norm(a, a_els);
      case INF: return dvinfnorm(a, a_els);
      default:  return dvpnorm(a, a_els, p);
   }
}

int dvnotzero(double *a, int a_els)
{
   int i, sum = 0;

   for (i=1; i <= a_els; i++)
      sum += !IS_ZERO(a[i]);

   return sum;
}

void dvmean(double *a, int a_els, double *mean, double *var)
{
   int i;

   (*mean) = 0.0;
   for (i=1; i <= a_els; i++) 
      (*mean) += a[i];
   (*mean) /= a_els;
   (*var) = 0.0;
   for (i=1; i <= a_els; i++) 
      (*var) += DSQR(a[i]- (*mean));
   (*var) /= (a_els-1);
}

void dmcolmean(double **a, int a_rows, int a_cols, double *mean, double *var)
{
   int i, j;

   for(j=1; j <= a_cols; j++)
   {
      mean[j] = 0.0;
      for (i=1; i <= a_rows; i++) 
         mean[j] += a[i][j];
      mean[j] /= a_rows;
      var[j] = 0.0;
      for (i=1; i <= a_rows; i++) 
         var[j] += DSQR(a[i][j]-mean[j]);
      var[j] /= (a_rows-1);
   }
}

void free_vector(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

void free_pvector(double **v, long nl, long nh)
/* free a double vector allocated with vector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

void free_cvector(unsigned char *v, long nl, long nh)
/* free an unsigned char vector allocated with cvector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

void free_lvector(unsigned long *v, long nl, long nh)
/* free an unsigned long vector allocated with lvector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

void free_matrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by matrix() */
{
        free((FREE_ARG) (m[nrl]+ncl-NR_END));
        free((FREE_ARG) (m+nrl-NR_END));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
        free((FREE_ARG) (m[nrl]+ncl-NR_END));
        free((FREE_ARG) (m+nrl-NR_END));
}

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
/* free an int matrix allocated by imatrix() */
{
        free((FREE_ARG) (m[nrl]+ncl-NR_END));
        free((FREE_ARG) (m+nrl-NR_END));
}

void free_submatrix(double **b, long nrl, long nrh, long ncl, long nch)
/* free a submatrix allocated by submatrix() */
{
        free((FREE_ARG) (b+nrl-NR_END));
}

void free_convert_matrix(double **b, long nrl, long nrh, long ncl, long nch)
/* free a matrix allocated by convert_matrix() */
{
        free((FREE_ARG) (b+nrl-NR_END));
}

void free_f3tensor(double ***t, long nrl, long nrh, long ncl, long nch,
        long ndl, long ndh)
/* free a double f3tensor allocated by f3tensor() */
{
        free((FREE_ARG) (t[nrl][ncl]+ndl-NR_END));
        free((FREE_ARG) (t[nrl]+ncl-NR_END));
        free((FREE_ARG) (t+nrl-NR_END));
}

#endif /* ANSI */
