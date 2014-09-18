#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
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
   fprintf(stdout, "  -t, --test=FILE             test file\n");
   fprintf(stdout, "  -e, --tolerance=TOL         tolerance parameter [default: 1e-9]\n");
   fprintf(stdout, "  -l, --l1=LAMBDA_1           l1 norm parameter [default: 1e-6]\n");
   fprintf(stdout, "  -r, --l2=LAMBDA_2           l2 norm parameter [default: 0]\n");
   fprintf(stdout, "  -b, --backtracking          use backtracking in FISTA\n");
   fprintf(stdout, "  -o, --original              do not standarize data to 0 mean and unit variance\n");
   fprintf(stdout, "  -p, --regpath=NVAL          regularization path for lambda_1\n");
   /*fprintf(stdout, "  -c, --cross-validation=K    do K-fold cross-validation\n");
   fprintf(stdout, "  -s, --stop=CRITERION        stopping criterion for FISTA [Default: OBJECTIVE]\n");
   fprintf(stdout, "      0: SUBGRADIENT\n");
   fprintf(stdout, "      1: OBJECTIVE\n");
   fprintf(stdout, "      2: ITERATIONS\n");*/
   fprintf(stdout, "  -i, --max-iters=ITERS       maximum number of iterations [Default: 1000 if CRITERION=2, INF otherwise]\n");
   fprintf(stdout, "  -v, --verbose               verbose\n");
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

double regularization_path(problem *prob, double epsilon, int nval)
{
   int nr_folds = 5;
   double llog, error, best_error = DBL_MAX, lambda, best_lambda;
   double lmax, lmin, lstep;
   double *y_hat = dvector(1, prob->n);
   double *w = dvector(1, prob->dim);

  /* compute maximum lambda for which all weights are 0 (Osborne et al. 1999)
    * lambda_max = ||X'y||_inf. According to scikit-learn source code, you can
    * divide by npatterns and it still works */
   dmvtransmult(prob->X, prob->n, prob->dim, prob->y, prob->n, w);
   lmax = dvnorm(w, prob->dim, INF) / prob->n;
   lmin = epsilon*lmax;
   lstep = (log2(lmax)-log2(lmin))/nval;

   fprintf(stdout, "lmax=%g lmin=%g epsilon=%g nval=%d\n",
           lmax, lmin, epsilon, nval);

   /* warm-starts: weights are set to 0 only at the begining */
   dvset(w, prob->dim, 0);
   for(llog=log2(lmax); llog >= log2(lmin); llog -= lstep)
   {
      lambda = pow(2, llog);
      /*cross_validation(prob, w, lambda, 0, nr_folds, y_hat);*/

      /*******************************************************/
      int iter = 1000; double tol = 0, fret;
      fista(prob, w, lambda, 0, tol, 0, &iter, &fret);
      fista_predict(prob, w, y_hat);
      /*******************************************************/

      error = mae(prob->y, prob->n, y_hat);
      fprintf(stdout, "   lambda %10.6lf   MAE %7.6lf   active weights %d/%d\n",
              lambda, error, dvnotzero(w, prob->dim), prob->dim);

      dvprint(stdout, w, prob->dim);

      if (error < best_error)
      {
         best_error = error;
         best_lambda = lambda;
      }
   }

   free_dvector(y_hat, 1, prob->n);
   free_dvector(w, 1, prob->dim);

   print_line(60);
   fprintf(stdout, "\nBest: lambda=%g MAE=%g active weights=%d/%d\n",
           best_lambda, best_error, dvnotzero(w, prob->dim), prob->dim);

   return best_lambda;
}

int main(int argc, char *argv[])
{
   char *ftest = NULL;
   struct timeval t0, t1, diff;
   problem *train, *test;
   int regpath_flag = 0, backtracking_flag = 0, std_flag = 1, verbose_flag = 0;
   int iter = 1000, c, crossval_flag = 0, nr_folds = 10, nval = 100, nzerow;
   double *w, *y_hat, *mean, *var;
   double lambda_1 = 1e-6, lambda_2 = 0, tol = 1e-9, epsilon, fret;

   while (1)
   {
      static struct option long_options[] =
      {
         /* These options don't set a flag.
          We distinguish them by their indices. */
         {"help",                   no_argument, 0, 'h'},
         {"verbose",                no_argument, 0, 'v'},
         {"backtracking",           no_argument, 0, 'b'},
         {"original",               no_argument, 0, 'o'},
         {"test",             required_argument, 0, 't'},
         {"l1",               required_argument, 0, 'l'},
         {"l2",               required_argument, 0, 'r'},
         {"cross-validation", optional_argument, 0, 'c'},
         {"tolerance       ", optional_argument, 0, 'e'},
         {"regpath",          optional_argument, 0, 'p'},
         /*{"stop",             optional_argument, 0, 's'},*/
         {"max-iters",        optional_argument, 0, 'i'},
         {0, 0, 0, 0}
      };

      int option_index = 0;

      c = getopt_long (argc, argv, "vhbot:r:l:p::c::e::s::i::", long_options, &option_index);

      /* Detect the end of the options. */
      if (c == -1)
         break;

      switch(c)
      {
         case 'h':
            exit_with_help(argv[PROG]);
            break;

         case 'b':
            backtracking_flag = 1;
            break;

         case 'v':
            verbose_flag = 1;
            break;

         case 'o':
            std_flag = 0;
            break;

         case 't':
            ftest = optarg;
            break;

         case 'c':
            crossval_flag = 1;
            if (optarg)
               if (sscanf(optarg, "%d", &nr_folds) != 1)
               {
                  fprintf(stderr, "%s: option -c requires an int\n", argv[PROG]);
                  exit_without_help(argv[PROG]);
               }
            break;

         case 'e':
            if (optarg)
               if (sscanf(optarg, "%lf", &tol) != 1)
               {
                  fprintf(stderr, "%s: option -e requires a double\n", argv[PROG]);
                  exit_without_help(argv[PROG]);
               }
            break;

         case 'p':
            regpath_flag = 1;
            if (optarg)
               if (sscanf(optarg, "%d", &nval) != 1)
               {
                  fprintf(stderr, "%s: option -p requires an int\n", argv[PROG]);
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

         case 'l':
            if (sscanf(optarg, "%lf", &lambda_1) != 1)
            {
               fprintf(stderr, "%s: option -l requires a float\n", argv[PROG]);
               exit_without_help(argv[PROG]);
            }
            break;

         case 'r':
            if (sscanf(optarg, "%lf", &lambda_2) != 1)
            {
               fprintf(stderr, "%s: option -r requires a float\n", argv[PROG]);
               exit_without_help(argv[PROG]);
            }
            break;

         case 'i':
            if (optarg)
               if (sscanf(optarg, "%d", &iter) != 1)
               {
                  fprintf(stderr, "%s: option -i requires an int\n", argv[PROG]);
                  exit_without_help(argv[PROG]);
               }
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
      lambda_1 = regularization_path(train, epsilon, nval);
   }

   fprintf(stdout, "lambda_1: %g\n", lambda_1);
   fprintf(stdout, "lambda_2: %g\n", lambda_2);

   /* initialize weight vector to 0 */
   w = dvector(1, train->dim);
   dvset(w, train->dim, 0);

   fprintf(stdout, "Training model...\n");
   if (backtracking_flag)
      /*fista_backtrack(train, w, lambda_1, lambda_2, tol, &iter, &fret);*/
      fista_nocov(train, w, lambda_1, lambda_2, tol, &iter, &fret);
   else
      fista(train, w, lambda_1, lambda_2, tol, verbose_flag, &iter, &fret);

   y_hat = dvector(1, train->n);
   fista_predict(train, w, y_hat);

   nzerow = dvnotzero(w, train->dim);

   fprintf(stdout, "Iterations: %d\n", iter);
   fprintf(stdout, "Active weights: %d/%d\n", nzerow, train->dim);
   if (std_flag)
      fprintf(stdout, "MAE train: %g\n", var[train->dim+1]*mae(train->y, train->n, y_hat));
   fprintf(stdout, "MAE train (standarized): %g\n", mae(train->y, train->n, y_hat));

   free_dvector(y_hat, 1, train->n);

   if (crossval_flag)
   {
      dvset(w, train->dim, 0);
      y_hat = dvector(1, train->n);
      cross_validation(train, w, lambda_1, lambda_2, nr_folds, y_hat);
      fprintf(stdout, "MAE cross-validation: %lf\n",
              mae(train->y, train->n, y_hat));
      free_dvector(y_hat, 1, train->n);
   }

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
