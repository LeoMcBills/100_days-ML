// Computes the final capital with interest and compound interest.
// Formula: capital = k0 * (1.0 + p/100)n
// where k0 = start capital, p = rate, n = run time

#include <math.h>

double capital( double k0, double p, double n)
{
    return (k0 * pow(1.0+p/100, n));
}