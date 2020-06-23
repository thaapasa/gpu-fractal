#ifndef KERNEL_UTILS_HH
#define KERNEL_UTILS_HH

////////////////////////////////////////////////////////////////////////////
// Kernel utilities
////////////////////////////////////////////////////////////////////////////

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    // notice switch red and blue to counter the GL_BGRA
    return (int(0xff) << 24) | (int(r) << 16) | (int(g) << 8) | int(b);
}

__device__ int getColor(size_t iters, size_t maxIters) {
    if (iters == maxIters)
        return 0;
    float s = ((float) iters) / maxIters;
    if (s < 0.1f) {
        s *= 10.0f;
        return rgbToInt(SLIDE(R0, R1, s), SLIDE(G0, G1, s), SLIDE(B0, B1, s));
    } else if (s < 0.5f) {
        s -= 0.1f;
        s *= (1.0f/0.4f);
        return rgbToInt(SLIDE(R1, R2, s), SLIDE(G1, G2, s), SLIDE(B1, B2, s));
    } else if (s < 0.5f) {
        s -= 0.5f;
        s *= 4.0f;
        return rgbToInt(SLIDE(R2, R3, s), SLIDE(G2, G3, s), SLIDE(B2, B3, s));
    } else {
        s -= 0.75f;
        s *= 4.0f;
        return rgbToInt(SLIDE(R3, R4, s), SLIDE(G3, G4, s), SLIDE(B3, B4, s));
    }
}

#endif // KERNEL_UTILS_HH
