#include <stdio.h>
#include "npy_math.h"


main (void)
{
    double x, y;

    x = 2.0;
    y = npy_sqrt(x);
    printf("sqrt(2) = %f\n", y);

    return 0;
}
