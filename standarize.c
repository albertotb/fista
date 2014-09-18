#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>

#define NR_END 1
#define FREE_ARG char*
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static double dsqrarg;
#define DSQR(a) ((dsqrarg=(a)) == 0.0 ? 0.0 : dsqrarg*dsqrarg)

static char *line = NULL;
static int max_line_len;

typedef struct {
   int n, dim;
   double *y;
   double **X;
} problem;

void exit_input_error(int line_num)
{
   fprintf(stderr,"Wrong input format at line %d\n", line_num);
   exit(1);
}

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
        fprintf(stdout,"Numerical Recipes run-time error...\n");
        fprintf(stdout,"%s\n",error_text);
        fprintf(stderr,"...now exiting to system...\n");
        exit(1);
}

double *dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
        double *v;

        v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
        if (!v) nrerror("allocation failure in dvector()");
        return v-nl+NR_END;
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

void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{
        free((FREE_ARG) (v+nl-NR_END));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by matrix() */
{
        free((FREE_ARG) (m[nrl]+ncl-NR_END));
        free((FREE_ARG) (m+nrl-NR_END));
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
void write_problem(FILE *f, problem *prob)
{
   int i, j;

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


int main(int argc, char *argv[])
{
   problem *train;
   double *mean, *var;

   train = read_problem(argv[1]);
   mean = dvector(1, train->dim+1);
   var = dvector(1, train->dim+1);
   standarize(train, 1, mean, var);
   write_problem(stdout, train);

   free_dvector(mean, 1, train->dim+1);
   free_dvector(var, 1, train->dim+1);
   free_dvector(train->y, 1, train->n);
   free_dmatrix(train->X, 1, train->n, 1, train->dim);
   free(train);
   return 0;
}
