/* CUda UTility Library */

// includes, system
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#undef min
#undef max
#endif // _WIN32


////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL error
//! @return CUTTrue if no GL error has been encountered, otherwise 0
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
//! @note The GL error is listed on stderr
//! @note This function should be used via the CHECK_ERROR_GL() macro
////////////////////////////////////////////////////////////////////////////
CUTBoolean CUTIL_API cutCheckErrorGL(const char* file, const int line) {
    CUTBoolean ret_val = CUTTrue;

    // check for error
    GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR) {
        fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
        fprintf(stderr, "%s\n", gluErrorString(gl_error));
        ret_val = CUTFalse;
    }
    return ret_val;
}

