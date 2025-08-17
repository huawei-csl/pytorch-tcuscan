#!/bin/bash

# Extract sparse matrix names

cut -d, -f 1 < sparse_suite_matrices.csv | awk '{print "\"" $0 "\","}' > stress_test.csv
