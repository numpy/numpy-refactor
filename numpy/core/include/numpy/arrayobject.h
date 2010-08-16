#ifndef _ARRAYOBJECT_H_
#define _ARRAYOBJECT_H_

/* This expects the following variables to be defined (besides
   the usual ones from pyconfig.h

   NPY_SIZEOF_LONG_DOUBLE -- sizeof(long double) or sizeof(double) if no
                             long double is present on platform.
   CHAR_BIT           --     number of bits in a char (usually 8)
                             (should be in limits.h)
*/

#include "ndarraytypes.h"
#include "ndarrayobject.h"
#ifdef NPY_NO_PREFIX
#include "noprefix.h"
#endif

#include "numpy/interrupt.h"

#endif /* _ARRAYOBJECT_H_ */
