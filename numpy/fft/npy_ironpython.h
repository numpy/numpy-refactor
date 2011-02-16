#ifndef _NPY_IRONPYTHON_H_
#define _NPY_IRONPYTHON_H_

#include "npy_object.h"

#define Npy_INTERFACE_OBJECT(a) (System::Runtime::InteropServices::GCHandle::FromIntPtr((System::IntPtr)Npy_INTERFACE(a)).Target)

#endif
