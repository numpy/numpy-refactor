using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using IronPython.Runtime;

namespace NumpyDotNet
{
    [PythonType]
    public class flagsobj
    {
        internal flagsobj(ndarray arr) {
            if (arr == null) {
                flags = NpyDefs.NPY_CONTIGUOUS | NpyDefs.NPY_OWNDATA | NpyDefs.NPY_FORTRAN | NpyDefs.NPY_ALIGNED;
            } else {
                flags = Marshal.ReadInt32(arr.Array, NpyCoreApi.ArrayOffsets.off_flags);
            }
            array = arr;
         }

        private bool ChkFlags(int check) {
            return (flags&check) == check;
        }

        public int num { get { return flags; } }

        public bool this[string name] {
            get {
                if (name != null) {
                    switch (name.Length) {
                        case 1:
                            switch (name[0]) {
                                case 'C':
                                    return contiguous;
                                case 'F':
                                    return fortran;
                                case 'W':
                                    return writeable;
                                case 'B':
                                    return behaved;
                                case 'O':
                                    return owndata;
                                case 'A':
                                    return aligned;
                                case 'U':
                                    return updateifcopy;
                            }
                            break;
                        case 2:
                            switch (name) {
                                case "CA":
                                    return carray;
                                case "FA":
                                    return farray;
                            }
                            break;
                        case 3:
                            switch (name) {
                                case "FNC":
                                    return fnc;
                            }
                            break;
                        case 5:
                            switch (name) {
                                case "FORC":
                                    return forc;
                            }
                            break;
                        case 6:
                            switch (name) {
                                case "CARRAY":
                                    return carray;
                                case "FARRAY":
                                    return farray;
                            }
                            break;
                        case 7:
                            switch (name) {
                                case "FORTRAN":
                                    return fortran;
                                case "BEHAVED":
                                    return behaved;
                                case "OWNDATA":
                                    return owndata;
                                case "ALIGNED":
                                    return aligned;
                            }
                            break;
                        case 9:
                            switch (name) {
                                case "WRITEABLE":
                                    return writeable;
                            }
                            break;
                        case 10:
                            switch (name) {
                                case "CONTIGUOUS":
                                    return contiguous;
                            }
                            break;
                        case 12:
                            switch (name) {
                                case "UPDATEIFCOPY":
                                    return updateifcopy;
                                case "C_CONTIGUOUS":
                                    return c_contiguous;
                                case "F_CONTIGUOUS":
                                    return f_contiguous;
                            }
                            break;
                    }
                }
                throw new System.Collections.Generic.KeyNotFoundException("Unknown flag");
            }
            set {
                if (name != null) {
                    if (name == "W" || name == "WRITEABLE") {
                        writeable = value;
                        return;
                    } else if (name == "A" || name == "ALIGNED") {
                        aligned = value;
                        return;
                    } else if (name == "U" || name == "UPDATEIFCOPY") {
                        updateifcopy = value;
                        return;
                    }
                }
                throw new System.Collections.Generic.KeyNotFoundException("Unknown flag");
            }
        }

        private string ValueLine(string key, bool includeNewline=true) {
            if (includeNewline) {
                return String.Format("  {0} : {1}\n", key, this[key]);
            } else {
                return String.Format("  {0} : {1}", key, this[key]);
            }
        }

        public virtual string __repr__() {
            return ToString();
        }

        public virtual string __str__() {
            return ToString();
        }

        public override string ToString() {
            return ValueLine("C_CONTIGUOUS") + 
                ValueLine("F_CONTIGUOUS") +
                ValueLine("OWNDATA") +
                ValueLine("WRITEABLE") +
                ValueLine("ALIGNED") +
                ValueLine("UPDATEIFCOPY", includeNewline:false);
        }

        // Get only flags
        public bool contiguous { get { return ChkFlags(NpyDefs.NPY_CONTIGUOUS); } }
        public bool c_contiguous { get { return ChkFlags(NpyDefs.NPY_CONTIGUOUS); } }
        public bool f_contiguous { get { return ChkFlags(NpyDefs.NPY_FORTRAN); } }
        public bool fortran { get { return ChkFlags(NpyDefs.NPY_FORTRAN); } }
        public bool owndata { get { return ChkFlags(NpyDefs.NPY_OWNDATA); } }
        public bool fnc { get { return f_contiguous && !c_contiguous; } }
        public bool forc { get { return f_contiguous || c_contiguous; } }
        public bool behaved { get { return ChkFlags(NpyDefs.NPY_BEHAVED); } }
        public bool carray { get { return ChkFlags(NpyDefs.NPY_CARRAY); } }
        public bool farray { get { return ChkFlags(NpyDefs.NPY_FARRAY) && !c_contiguous; } }

        // get/set flags
        public bool aligned {
            get {
                return ChkFlags(NpyDefs.NPY_ALIGNED);
            }
            set {
                if (array == null) {
                    throw new ArgumentException("Cannet set flags on array scalars");
                }
                array.setflags(null, value, null);
            }
        }

        public bool updateifcopy {
            get {
                return ChkFlags(NpyDefs.NPY_UPDATEIFCOPY);
            }
            set {
                if (array == null) {
                    throw new ArgumentException("Cannot set flags on array scalars");
                }
                array.setflags(null, null, value);
            }
        }

        public bool writeable {
            get {
                return ChkFlags(NpyDefs.NPY_WRITEABLE);
            }
            set {
                if (array == null) {
                    throw new ArgumentException("Cannot set flags on array scalars");
                }
                array.setflags(value, null, null);
            }
        }


        private int flags;
        private ndarray array;
    }
}
