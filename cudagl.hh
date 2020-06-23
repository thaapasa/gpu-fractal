#ifndef CUDAGL_HH
#define CUDAGL_HH

// Graphics includes
#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif

class CudaGL {
private:
    size_t width;
    size_t height;

    // The fractal is rendered to this buffer
    GLuint glBuffer;
    // The completed fractal is copied to this texture and then rendered
    GLuint glResultTexture;
    struct cudaGraphicsResource* cudaBuffer;

    bool created;
    bool mapped;

public:
    CudaGL(size_t width, size_t height);
    ~CudaGL();

    void resize(size_t width, size_t height);

    unsigned int *mapCudaBuffer();
    void unmapCudaBuffer();

    void drawBufferGL();

private:
    void createBuffers();
    void clearBuffers();
};

#endif // CUDAGL_HH
