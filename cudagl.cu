#include "cudagl.hh"
#include "utils.hh"

#include <cutil_inline.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <cassert>

using namespace std;

CudaGL::CudaGL(size_t width_, size_t height_) :
    width(width_), height(height_), glBuffer(0), glResultTexture(0), cudaBuffer(0), created(false), mapped(false) {
    createBuffers();
}

CudaGL::~CudaGL() {
    clearBuffers();
}

void CudaGL::createBuffers() {
    size_t nTexels = width * height;
    size_t bufferSize = sizeof(GLubyte) * nTexels * 4;

    // Create buffer object
    glGenBuffers(1, &glBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, glBuffer);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_DYNAMIC_DRAW);

    // Register this buffer object with CUDA
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cudaBuffer, glBuffer,
            cudaGraphicsMapFlagsNone));
    CUT_CHECK_ERROR_GL();
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Create the texture that we use to visualise the drawn result
    glGenTextures(1, &glResultTexture);
    glBindTexture(GL_TEXTURE_2D, glResultTexture);

    // Set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Buffer data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    created = true;
    CUT_CHECK_ERROR_GL();
}

void CudaGL::clearBuffers() {
    if (!created)
        return;
    assert(!mapped);

    CUT_CHECK_ERROR_GL();

    cutilSafeCall(cudaGraphicsUnregisterResource(cudaBuffer));
    cudaBuffer = NULL;
    // cutilSafeCall(cudaGLUnregisterBufferObject(glBuffer));
    glBindBuffer(GL_ARRAY_BUFFER, glBuffer);
    glDeleteBuffers(1, &glBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBuffer = 0;

    // glBindBuffer(GL_ARRAY_BUFFER, glResultTexture);
    glBindTexture(GL_TEXTURE_2D, glResultTexture);
    glDeleteBuffers(1, &glResultTexture);
    glBindTexture(GL_TEXTURE_2D, 0);
    glResultTexture = 0;
    created = false;

    CUT_CHECK_ERROR_GL();
}

void CudaGL::resize(size_t width_, size_t height_) {
    glViewport(0, 0, width, height);

    if (width == width_ && height == height_)
        return;

    width = width_;
    height = height_;

    clearBuffers();
    createBuffers();
}


unsigned int* CudaGL::mapCudaBuffer() {
    unsigned int* out;
    size_t numBytes;
    assert(!mapped);
    // Map the GL buffer to CUDA
    cutilSafeCall(cudaGraphicsMapResources(1, &cudaBuffer, 0));
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&out,
            &numBytes, cudaBuffer));

    mapped = true;
    assert(numBytes == width * height * 4);
    return out;
}

void CudaGL::unmapCudaBuffer() {
    // Unmap the GL buffer from CUDA
    assert(mapped);
    cutilSafeCall(cudaGraphicsUnmapResources(1, &cudaBuffer, 0));
    mapped = false;
}

void CudaGL::drawBufferGL() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, glBuffer);
    glBindTexture(GL_TEXTURE_2D, glResultTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA,
            GL_UNSIGNED_BYTE, NULL);

    CUT_CHECK_ERROR_GL();

    // Render a screen sized quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

    // Unbind
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    CUT_CHECK_ERROR_GL();
}

