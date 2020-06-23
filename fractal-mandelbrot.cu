#include "fractal.hh"
#include "utils.hh"

#include <cutil_inline.h>
#include <cuda_gl_interop.h>
#include <cutil_gl_inline.h>
#include <cutil_math.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_functions.h>

#include <iostream>
#include <iomanip>

using namespace std;

#include "kernel-utils.hh"

__global__ void drawMandelbrotKernel(unsigned int *out,
        float centerX, float centerY, float scale, size_t width, size_t height,
        size_t maxIters) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Position of this pixel
    float x0 = centerX + (x - (width * 0.5f)) * scale;
    float y0 = centerY + (y - (height * 0.5f)) * scale;

    int iters = 0;
    float cx = 0, cy = 0;
    // Squared values
    float scx = 0, scy = 0;
    while ((scx + scy) < 4 && iters < maxIters) {
        float xt = scx - scy + x0;
        cy = 2 * cx * cy + y0;
        cx = xt;
        // Calculate squared values
        scx = cx * cx;
        scy = cy * cy;
        iters++;
    }

    unsigned int color = getColor(iters, maxIters);
    out[y * width + x] = color;
}

MandelbrotRenderer::MandelbrotRenderer(size_t width, size_t height) :
    FractalRenderer(width, height) {
}

MandelbrotRenderer::~MandelbrotRenderer() {
}

void MandelbrotRenderer::calculateFractal(unsigned int* target) {
    size_t iters = 4.0f / pow(scale, 0.6);
    iters = min((unsigned int) iters, 10000);
    iters *= iterScale;
    cout << "Rendering Mandelbrot at " << fixed << setprecision(5)
            << x << "," << y << " with scale " << scale << ", max iters " << iters << endl;

    drawMandelbrotKernel<<<*dimGrid, *dimBlock>>>(target, x, y, scale, width, height, iters);
    changed = false;
}
