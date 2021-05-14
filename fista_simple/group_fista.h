#pragma once

double *vector(long n);
void free_vector(double *v);
void dvprint(FILE *f, double *a, int a_els);
void fista_gram(double **G, double *c, double *w, int dim, int *grp, int ngrp, double L, double lambda1, double lambda_2, double ftol, int *iter, double *fret);
