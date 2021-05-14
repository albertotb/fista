#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <string.h>
#include "nrutil.h"
#include "fista.h"
#include "eigen.h"

#define ARGC_MIN 1
#define ARGC_MAX 1
#define PROG     0

#define INF INT_MAX
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_line(int length)
{
   int i;
   for(i=0; i<length; i++) fprintf(stdout, "-");
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
   fprintf(stdout, "  -t, --test=FILE             test file\n");
   fprintf(stdout, "  -e, --tolerance=TOL         tolerance parameter [default: 1e-3]\n");
   fprintf(stdout, "  -1, --l1=LAMBDA_1           l1 norm parameter [default: 1e-6]\n");
   fprintf(stdout, "  -2, --l2=LAMBDA_2           l2 norm parameter [default: 0]\n");
   fprintf(stdout, "  -g, --group=NGRP1,NGRP2,... Group Elastic Net [default: no groups]\n");
   fprintf(stdout, "  -p, --precompute            precompute the Gram matrix X^tX\n");
   fprintf(stdout, "  -i, --max-iters=ITERS       maximum number of iterations [Default: 1000 if CRITERION=2, INF otherwise]\n");
   fprintf(stdout, "  -r, --regpath=NVAL          regularization path for lambda_1\n");
   //fprintf(stdout, "  -s, --stop=CRITERION        stopping criterion for FISTA [Default: OBJECTIVE]\n");
   //fprintf(stdout, "      0: SUBGRADIENT\n");
   //fprintf(stdout, "      1: OBJECTIVE\n");
   //fprintf(stdout, "      2: ITERATIONS\n");
   //fprintf(stdout, "  -c, --cross-validation=K    do K-fold cross-validation\n");
   exit(1);
}

int timeval_subtract (struct timeval *x, struct timeval *y, struct timeval *result)
{
   /* Perform the carry for the later subtraction by updating y. */
   if (x->tv_usec < y->tv_usec) {
      int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
      y->tv_usec -= 1000000 * nsec;
      y->tv_sec += nsec;
   }
   if (x->tv_usec - y->tv_usec > 1000000) {
      int nsec = (x->tv_usec - y->tv_usec) / 1000000;
      y->tv_usec += 1000000 * nsec;
      y->tv_sec -= nsec;
   }

   /* Compute the time remaining to wait.
      tv_usec is certainly positive. */
   result->tv_sec = x->tv_sec - y->tv_sec;
   result->tv_usec = x->tv_usec - y->tv_usec;

   /* Return 1 if result is negative. */
   return x->tv_sec < y->tv_sec;
}

int main(int argc, char *argv[])
{
   char *ftest = NULL;
   struct timeval t0, t1, diff;
   problem *train, *test;
   int regpath_flag = 0, precomp_flag = 0, std_flag = 1, verbose_flag = 0;
   int iter = 1000, c, crossval_flag = 0, nr_folds = 10, nval = 100, nzerow;
   double **Q, *q, *w, *y_hat, *mean, *var;
   double lambda_1 = 1e-3, lambda_2 = 0, tol = 1e-6, epsilon, fret;
   int ngrp = 0, *grp = NULL;
   int i;
   char *s;

   while (1)
   {
      static struct option long_options[] =
      {
         /* These options don't set a flag.
          We distinguish them by their indices. */
         {"help",                   no_argument, 0, 'h'},
         {"verbose",                no_argument, 0, 'v'},
         {"original",               no_argument, 0, 'o'},
         {"precompute",             no_argument, 0, 'p'},
         {"test",             required_argument, 0, 't'},
         {"l1",               required_argument, 0, '1'},
         {"l2",               required_argument, 0, '2'},
         {"tolerance       ", required_argument, 0, 'e'},
         {"max-iters",        required_argument, 0, 'i'},
         {"group",            required_argument, 0, 'g'},
         {"regpath",          optional_argument, 0, 'r'},
         /*{"stop",             optional_argument, 0, 's'},*/
         /*{"cross-validation", optional_argument, 0, 'c'},*/
         {0, 0, 0, 0}
      };

      int option_index = 0;

      c = getopt_long (argc, argv, "vhopt:1:2:g:e:s:i:r::", long_options, &option_index);

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

         case 'o':
            std_flag = 0;
            break;

         case 'p':
            precomp_flag = 1;
            break;

         case 't':
            ftest = optarg;
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
            while (s = strsep(&optarg, ","))
            {
               if (sscanf(s, "%d", &grp[i]) != 1)
               {
                  fprintf(stderr, "%s: incorrect group structure\n", argv[PROG]);
                  exit_without_help(argv[PROG]);
               }
               i++;
            }

            break;

         case 'i':
            if (sscanf(optarg, "%d", &iter) != 1)
            {
               fprintf(stderr, "%s: option -i requires an int\n", argv[PROG]);
               exit_without_help(argv[PROG]);
            }
            break;

         case 'r':
            regpath_flag = 1;
            if (optarg)
               if (sscanf(optarg, "%d", &nval) != 1)
               {
                  fprintf(stderr, "%s: option -r requires an int\n", argv[PROG]);
                  exit_without_help(argv[PROG]);
               }
            break;
         //case 's':
         //   search_flag = 1;
         //   if (optarg)
         //      if (sscanf(optarg, "%lf:%d:%lf", &lmax, &nval, &lmin) != 3)
         //      {
         //         printf("%s\n", optarg);
         //         fprintf(stderr, "%s: option -s requires a range in the format MAX:NVAL:MIN\n", argv[PROG]);
         //         exit_without_help(argv[PROG]);
         //      }
         //   break;

         //case 'c':
         //   crossval_flag = 1;
         //   if (optarg)
         //      if (sscanf(optarg, "%d", &nr_folds) != 1)
         //      {
         //         fprintf(stderr, "%s: option -c requires an int\n", argv[PROG]);
         //         exit_without_help(argv[PROG]);
         //      }
         //   break;

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

   /* start time */
   gettimeofday(&t0, 0);

   train = read_problem(argv[optind]);

   fprintf(stdout, "n:%d dim:%d\n", train->n, train->dim);

   /* alloc vector for means and variances, plus 1 for output */
   if (std_flag)
   {
      fprintf(stdout, "Standarizing train set...\n");
      mean = dvector(1, train->dim+1);
      var = dvector(1, train->dim+1);
      standarize(train, 1, mean, var);
   }

   if (ftest)
   {
      test = read_problem(ftest);
      if (std_flag)
         standarize(test, 0, mean, var);
   }

   if (regpath_flag)
   {
      fprintf(stdout, "Regularization path...\n");
      /* in glmnet package they use 0.0001 instead of 0.001 ? */
      epsilon = train->n > train->dim ? 0.001 : 0.01;
      lambda_1 = regularization_path(train, epsilon, nval, lambda_2);
   }

   fprintf(stdout, "lambda_1: %g\n", lambda_1);
   fprintf(stdout, "lambda_2: %g\n", lambda_2);

   /* initialize weight vector to 0 */
   w = dvector(1, train->dim);
   dvset(w, train->dim, 0);

   fprintf(stdout, "Training model...\n");
   if (precomp_flag)
   {
      q = dvector(1, train->dim);
      Q = dmatrix(1, train->dim, 1, train->dim);

      /* Q = trans(X)*X */
      dmtransmult(train->X, train->n, train->dim, Q);

      /* q = trans(X)*y */
      dmvtransmult(train->X, train->n, train->dim, train->y, train->n, q);

      lambda_1 = lambda_1 * train->n;
      lambda_2 = lambda_2 * train->n;
      fista_gram(Q, q, w, train->dim, grp, ngrp, lambda_1, lambda_2, tol, &iter, &fret);

      free_dvector(q, 1, train->dim);
      free_dmatrix(Q, 1, train->dim, 1, train->dim);
   }
   else
   {
      fista(train, w, lambda_1, lambda_2, tol, &iter, &fret);
   }

   y_hat = dvector(1, train->n);
   fista_predict(train, w, y_hat);

   nzerow = dvnotzero(w, train->dim);

   fprintf(stdout, "Iterations: %d\n", iter);
   fprintf(stdout, "Fret: %g\n", fret);
   fprintf(stdout, "Active weights: %d/%d\n", nzerow, train->dim);
   if (std_flag)
      fprintf(stdout, "MAE train: %g\n", var[train->dim+1]*mae(train->y, train->n, y_hat));
   fprintf(stdout, "MAE train (standarized): %g\n", mae(train->y, train->n, y_hat));

   free_dvector(y_hat, 1, train->n);

   //if (crossval_flag)
   //{
   //   dvset(w, train->dim, 0);
   //   y_hat = dvector(1, train->n);
   //   cross_validation(train, w, lambda_1, lambda_2, nr_folds, y_hat, precomp_flag);
   //   fprintf(stdout, "MAE cross-validation: %lf\n",
   //           mae(train->y, train->n, y_hat));
   //   free_dvector(y_hat, 1, train->n);
   //}

   if (ftest)
   {
      /* we alloc memory again since test size is different from train size */
      y_hat = dvector(1, test->n);
      fista_predict(test, w, y_hat);
      fprintf(stdout, "MAE test: %g\n", mae(test->y, test->n, y_hat));
      free_dvector(y_hat, 1, test->n);
   }

   /* stop time */
   gettimeofday(&t1, 0);
   timeval_subtract(&t1, &t0, &diff);
   fprintf(stdout, "Time(h:m:s.us): %02d:%02d:%02d.%06ld\n",
           diff.tv_sec/3600, (diff.tv_sec/60), diff.tv_sec%60, diff.tv_usec);

   if (verbose_flag)
   {
      fprintf(stdout, "Weights: ");
      dvprint(stdout, w, train->dim);
   }

   free_dvector(w, 1, train->dim);

   if (std_flag)
   {
      free_dvector(mean, 1, train->dim+1);
      free_dvector(var, 1, train->dim+1);
   }

   if (ftest)
   {
      free_dvector(test->y, 1, test->n);
      free_dmatrix(test->X, 1, test->n, 1, test->dim);
      free(test);
   }

   free_dvector(train->y, 1, train->n);
   free_dmatrix(train->X, 1, train->n, 1, train->dim);
   free(train);

   return 0;
}
