/*
  "before" is 16 bytes to ensure there's no padding between it and "x".
   We're not expecting any "long double" bigger than 16 bytes or with
   alignment requirements stricter than 16 bytes.
*/
typedef long double test_type;

struct {
        char         before[16];
        test_type    x;
        char         after[8];
}

foo = {
    { '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
      'a', 'R', '8', 'q', 'y', 'b', '1', 'W' },
    -123456789.0,
    { 'z', '7', 'p', 'L', 'C', '3', 'S', 'i' }
};
