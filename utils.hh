#ifndef UTILS_HH
#define UTILS_HH

#include <cutil_inline.h>

// Delete and nullify a variable
#define ZAP(x) { \
        if (x != NULL) {  \
            delete x;     \
            x = NULL;     \
        } }


CUTBoolean CUTIL_API cutCheckErrorGL(const char* file, const int line);

#ifdef _DEBUG
#define CUT_CHECK_ERROR_GL()                                     \
        if( CUTFalse == cutCheckErrorGL( __FILE__, __LINE__)) {  \
            exit(EXIT_FAILURE);                                  \
        }
#else
#define CUT_CHECK_ERROR_GL()
#endif // _DEBUG

#endif // UTILS_HH
