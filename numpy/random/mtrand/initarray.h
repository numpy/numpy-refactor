#include "randomkit.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void
init_by_array(rk_state *self, unsigned long init_key[],
              unsigned long key_length);

#ifdef __cplusplus
}
#endif