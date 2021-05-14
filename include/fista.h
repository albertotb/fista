/***********************************************************************
    fista.h

    FISTA functions

************************************************************************/
#ifndef _FISTA_H_
#define _FISTA_H_

typedef struct {
   int n, dim;
   double *y;
   double **X;
} problem;

typedef enum {SUBGRADIENT, OBJECTIVE, ITERATIONS} stop_crit_t;

problem *read_problem(const char *filename);
void write_problem(problem *prob, char *filename);
double mse(double *y, int n, double *y_hat);
double mae(double *y, int n, double *y_hat);
void standarize(problem *prob, int is_train, double *mean, double *var);
void fista(problem *prob, double *w, double lambda_1, double lambda_2, double ftol, int *iter, double *fret);
void fista_gram(double **G, double *c, double *w, int dim, int *grp, int ngrp, double lambda1, double lambda_2, double ftol, int *iter, double *fret);
void fista_predict(problem *prob, double *w, double *y);
double regularization_path(problem *prob, double epsilon, int nval, double lambda_2);

void l21_prox(double *x, int n, double lambda, int *grp, int ngrp);

#endif /* _FISTA_H_ */
