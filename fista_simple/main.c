#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <sys/time.h>
#include <string.h>
#include <ctype.h>
#include <cblas.h>
#include "group_fista.h"
#include "eigen.h"

#define ARGC_MIN 1
#define ARGC_MAX 1
#define PROG     0

#define FREE_ARG char*

static double sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef struct {
   int n, dim;
   double *y;
   double **X;
} problem;

static char *line = NULL;
static int max_line_len;

void exit_input_error(int line_num)
{
   fprintf(stderr,"Wrong input format at line %d\n", line_num);
   exit(1);
}

void exit_without_help(char*prog)
{
   fprintf(stderr, "Try `%s --help' for more information.\n", prog);
   exit(1);
}

void exit_with_help(char *prog)
{
   fprintf(stdout, "Usage: %s [OPTION]... TRAIN\n", prog);
   fprintf(stdout, "Build an Elastic Net model using FISTA algorithm.\n\n");
   fprintf(stdout, "Options:\n");
   fprintf(stdout, "  -h, --help                  show help and exit\n");
   fprintf(stdout, "  -v, --verbose               verbose\n");
   fprintf(stdout, "  -o, --original              do not standarize data to 0 mean and unit variance\n");
   fprintf(stdout, "  -e, --tolerance=TOL         tolerance parameter [default: 1e-9]\n");
   fprintf(stdout, "  -1, --l1=LAMBDA_1           l1 norm parameter [default: 1e-6]\n");
   fprintf(stdout, "  -2, --l2=LAMBDA_2           l2 norm parameter [default: 0]\n");
   fprintf(stdout, "  -g, --group=NGRP1,NGRP2,... Group Elastic Net [default: no groups]\n");
   exit(1);
}

double **matrix(long nrow, long ncol)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
        double **m;

        /* allocate pointers to rows */
        m=(double **) malloc((size_t)((nrow)*sizeof(double*)));
        if (!m) nrerror("allocation failure 1 in matrix()");

        /* allocate rows and set pointers to them */
        m[0]=(double *) malloc((size_t)((ncol)*(nrow)*sizeof(double)));
        if (!m[0]) nrerror("allocation failure 2 in matrix()");

        for(long i=1;i<nrow;i++) m[i]=m[i-1]+ncol;

        /* return pointer to array of pointers to rows */
        return m;
}

void free_matrix(double **m)
/* free a double matrix allocated by matrix() */
{
        free((FREE_ARG) (m[0]));
        free((FREE_ARG) (m));
}

void dmprint(FILE *f, double **a, int a_rows, int a_cols)
{
   for (int i=0; i < a_rows; i++)
   {
      for (int j=0; j < (a_cols-1); j++)
         fprintf(f, "%g ", a[i][j]);
      fprintf(f, "%g\n", a[i][a_cols-1]);
   }
}

void dmcopy(double **a, int a_rows, int a_cols, double **y)
{
   for(int i=0; i < a_rows; i++)
      for(int j=0; j < a_cols; j++)
         y[i][j] = a[i][j];
}

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

   prob->y = vector(prob->n);
   prob->X = matrix(prob->n, prob->dim);

   for(i=0; i<prob->n; i++)
   {
      readline(fp);
      label = strtok(line," \t\n");
      if(label == NULL) // empty line
         exit_input_error(i+1);

      prob->y[i] = strtod(label,&endptr);
      if(endptr == label || *endptr != '\0')
         exit_input_error(i+1);

      j=0;
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

void dmtransmult(double **a, int a_rows, int a_cols, double **y)
{
#ifdef _BLAS_
   cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
         a_cols, a_rows, 1, &a[0][0], a_cols, 0, &y[0][0], a_cols);
#else
   int i, j, k;
   double sum;

   for (i=0; i < a_cols; i++)
      for (j=0; j < a_cols; j++)
      {
         sum = 0.0;
         for (k=0; k < a_rows; k++)
            sum += a[k][i] * a[k][j];

        y[i][j] = sum;
      }
   /* Copiamos lo que nos falta */
   for (i = 0; i < a_cols; i++)
      for (j = 0; j < i-1; j++)
         y[i][j] = y[j][i];
#endif
}

void dmvtransmult(double **a, int a_rows, int a_cols,
             double *b, int b_els, double *y)
{
#ifdef _BLAS_
   cblas_dgemv(CblasRowMajor, CblasTrans, a_rows, a_cols, 1, &a[0][0], 
         a_cols, b, 1, 0, y, 1);
#else
   int j, k;
   double sum;

   if (a_rows != b_els)
   {
      fprintf(stderr,"a_rows <> b_els (%d,%d): dmvtransmult\n", a_rows, b_els);
      exit(1);
   }

   for (j=0; j < a_cols; j++) {
      sum = 0.0;
      for (k=0; k < a_rows; k++) sum += a[k][j]*b[k];
      y[j] = sum;
   }
#endif
}

void dvmean(double *a, int a_els, double *mean, double *var)
{
   int i;

   (*mean) = 0.0;
   for (i=0; i < a_els; i++) 
      (*mean) += a[i];
   (*mean) /= a_els;
   (*var) = 0.0;
   for (i=0; i < a_els; i++) 
      (*var) += SQR(a[i]- (*mean));
   (*var) /= (a_els-1);
}

void dmcolmean(double **a, int a_rows, int a_cols, double *mean, double *var)
{
   int i, j;

   for(j=0; j < a_cols; j++)
   {
      mean[j] = 0.0;
      for (i=0; i < a_rows; i++) 
         mean[j] += a[i][j];
      mean[j] /= a_rows;
      var[j] = 0.0;
      for (i=0; i < a_rows; i++) 
         var[j] += SQR(a[i][j]-mean[j]);
      var[j] /= (a_rows-1);
   }
}

void standarize(problem *prob, int is_train, double *mean, double *var)
{
   int n = prob->n, dim = prob->dim;

   if (is_train)
   {
      dmcolmean(prob->X, n, dim, mean, var);
      dvmean(prob->y, n, &(mean[dim]), &(var[dim]));
   }

   for (int i=0; i < n; i++)
   {
      for (int j=0; j < dim; j++)
         prob->X[i][j] = (prob->X[i][j]-mean[j]) / sqrt(var[j]);

      prob->y[i] = (prob->y[i]-mean[dim]) / sqrt(var[dim]);
   }
}

int main(int argc, char *argv[])
{
   char c, *s;
   problem *train;
   double **Q, *q, *w, *y_hat, *mean, *var;
   int i, iter, verbose_flag = 0, ngrp = 0, *grp = NULL;
   double L, lambda_1 = 1e-6, lambda_2 = 0, tol = 1e-9, epsilon, fret;

   while (1)
   {
      static struct option long_options[] =
      {
         /* These options don't set a flag.
          We distinguish them by their indices. */
         {"help",                   no_argument, 0, 'h'},
         {"verbose",                no_argument, 0, 'v'},
         {"l1",               required_argument, 0, '1'},
         {"l2",               required_argument, 0, '2'},
         {"tolerance       ", required_argument, 0, 'e'},
         {"max-iters",        required_argument, 0, 'i'},
         {"group",            required_argument, 0, 'g'},
         {0, 0, 0, 0}
      };

      int option_index = 0;

      c = getopt_long (argc, argv, "vh1:2:g:e:i:", long_options, &option_index);

      /* Detect the end of the options. */
      if (c == -1)
         break;

      switch(c)
      {
         case 'h':
            exit_with_help(argv[PROG]);
            break;

         case 'v':
            verbose_flag = 1;
            break;

         case 'e':
            if (sscanf(optarg, "%lf", &tol) != 1)
            {
               fprintf(stderr, "%s: option -e requires a double\n", argv[PROG]);
               exit_without_help(argv[PROG]);
            }
            break;

         case '1':
            if (sscanf(optarg, "%lf", &lambda_1) != 1)
            {
               fprintf(stderr, "%s: option -l requires a float\n", argv[PROG]);
               exit_without_help(argv[PROG]);
            }
            break;

         case '2':
            if (sscanf(optarg, "%lf", &lambda_2) != 1)
            {
               fprintf(stderr, "%s: option -r requires a float\n", argv[PROG]);
               exit_without_help(argv[PROG]);
            }
            break;

         case 'g':
            /* count number of groups: number of ',' plus 1 */
            for (ngrp=0, s=optarg; s[ngrp]; s[ngrp]==',' ? ngrp++ : *s++);
            ngrp += 1;

            /* alloc memory */
            grp = Malloc(int, ngrp);

            /* tokenize group sizes */
            i = 0;
            for (i=0, s = strtok(optarg, ","); s != NULL; s = strtok(NULL, ","), i++)
            {
               if (sscanf(s, "%d", &grp[i]) != 1)
               {
                  fprintf(stderr, "%s: incorrect group structure\n", argv[PROG]);
                  exit_without_help(argv[PROG]);
               }
               printf("%d ", grp[i]);
            }
            printf("\n");

            break;

         case '?':
            /* getopt_long already printed an error message. */
            exit_without_help(argv[PROG]);
            break;

         default:
            printf("?? getopt returned character code 0%o ??\n", c); 
      }
   }

   if ((argc - optind) < ARGC_MIN || (argc - optind) > ARGC_MAX)
   {
      fprintf(stderr, "%s: missing file operand\n", argv[PROG]);
      exit_without_help(argv[PROG]);
   }

   train = read_problem(argv[optind]);

   fprintf(stdout, "n:%d dim:%d\n", train->n, train->dim);

   /* alloc vector for means and variances, plus 1 for output */
   //fprintf(stdout, "Standarizing train set...\n");
   //mean = vector(train->dim+1);
   //var = vector(train->dim+1);
   //standarize(train, 1, mean, var);

   fprintf(stdout, "lambda_1: %g\n", lambda_1);
   fprintf(stdout, "lambda_2: %g\n", lambda_2);

   /* initialize weight vector to 0 */
   w = vector(train->dim);
   memset(w, 0, train->dim*sizeof(*w));

   fprintf(stdout, "Training model...\n");
   q = vector(train->dim);
   Q = matrix(train->dim, train->dim);

   /* Q = trans(X)*X */
   dmtransmult(train->X, train->n, train->dim, Q);
   dmprint(stdout, Q, train->dim, train->dim);

   /* q = trans(X)*y */
   dmvtransmult(train->X, train->n, train->dim, train->y, train->n, q);
   dvprint(stdout, q, train->dim);

   /* L = max(eigs(Q)) */
   double **A = matrix(train->dim, train->dim);
   dmcopy(Q, train->dim, train->dim, A);
   L = max_eig(A, train->dim);
   free_matrix(A);

   printf("L: %g\n", L);

   /* optimize */
   fista_gram(Q, q, w, train->dim, grp, ngrp, L, lambda_1, lambda_2, tol, &iter, &fret);

   /* solution */
   fprintf(stdout, "Iterations: %d\n", iter);
   fprintf(stdout, "Fret: %g\n", fret);
   fprintf(stdout, "Weights: ");
   dvprint(stdout, w, train->dim);

   /* free memory */
   free_vector(q);
   free_matrix(Q);
   free_vector(w);
   free_vector(mean);
   free_vector(var);
   free_vector(train->y);
   free_matrix(train->X);
   free(train);

   return 0;
}
