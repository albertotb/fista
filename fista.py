#!/bin/env python

import sys
import os
import numpy as np
import numpy.ctypeslib as npct
from ctypes import *

from sklearn.preprocessing import StandardScaler

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=c_int, ndim=1, flags='CONTIGUOUS')
doublepp = npct.ndpointer(dtype=np.uintp, ndim=1, flags='C')

libfista = npct.load_library('libfista.so', '.')

libfista.fista_gram.argtypes = (doublepp, array_1d_double, array_1d_double,
        c_int, POINTER(c_int), c_int, c_double, c_double, c_double, c_double,
        POINTER(c_int), POINTER(c_double))

libfista.fista_gram.restype = None

def fista_gram(Q, q, lam1, lam2=0, group=None):

    tol = 1e-10

    if group is None:
        ngrp = 0
    else:
        ngrp = len(group)
        group = (c_int * ngrp)(*group)

    d = len(q)

    print "Q:"
    print Q
    print "q:"
    print q

    e, _ = np.linalg.eig(Q)
    L = np.amax(e)

    print "L:", L

    Q = np.ascontiguousarray(Q, dtype=np.double)
    Qpp = (Q.__array_interface__['data'][0] +
            np.arange(d)*Q.strides[0]).astype(np.uintp)

    q = np.ascontiguousarray(q, dtype=np.double)

    w = np.zeros(d, dtype=np.double)
    iter = c_int()
    obj = c_double()

    libfista.fista_gram(Qpp, q, w, d, group, ngrp, L, lam1, lam2, tol, byref(iter), byref(obj))

    print w
    print iter.value
    print obj.value


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "usage: {0} FILE".format(sys.argv[0])
        sys.exit(1)

    data = np.loadtxt(sys.argv[1])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    X = data[:, 1:]
    y = data[:, 0]

    Q = np.dot(X.T, X)
    q = np.dot(X.T, y)

    print Q.shape
    print q.shape

    fista_gram(Q, q, 1e-6)
