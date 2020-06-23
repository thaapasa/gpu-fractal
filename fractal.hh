#ifndef FRACTAL_HH
#define FRACTAL_HH

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

#include <iostream>
#include <iomanip>

using namespace std;

#define SLIDE(c1, c2, s) ((((c2) - (c1)) * (s)) + (c1))

// Colors used
#define R0 0.0f
#define G0 0.0f
#define B0 0.0f

#define R1 64.0f
#define G1 64.0f
#define B1 256.0f

#define R2 256.0f
#define G2 128.0f
#define B2 64.0f

#define R3 186.0f
#define G3 0.0f
#define B3 0.0f

#define R4 128.0f
#define G4 64.0f
#define B4 256.0f


// Utilities and system includes

typedef struct {
    float *fractal_val;
    size_t pitch;
} cudaparams_t;

class dim3;

class FractalRenderer {
protected:
    size_t width;
    size_t height;
    float x;
    float y;
    float scale;
    cudaparams_t h_params;
    dim3 *dimBlock;
    dim3 *dimGrid;

    bool changed;
    float iterScale;

public:
    FractalRenderer(size_t width, size_t height);
    virtual ~FractalRenderer();

    virtual void calculateFractal(unsigned int *out) = 0;
    void setCenter(float x_, float y_) { x = x_; y = y_; changed = true; }
    void setX(float x_) { x = x_; changed = true; }
    void setY(float y_) { y = y_; changed = true; }
    float getX() const { return x; }
    float getY() const { return y; }
    void setPixelScale(float scale_) { scale = scale_; changed = true; }
    float getPixelScale() const { return scale; }

    void resize(size_t width, size_t height);
    void zoom(int x1, int y1, int x2, int y2);

    bool isChanged() const { return changed; }

    float screenXtoReal(int sx) { return x - (width * scale / 2.0) + sx * scale; }
    float screenYtoReal(int sy) { return y + (height * scale / 2.0) - sy * scale; }

    void setIterScale(float scale) { iterScale = scale; changed = true; }
    float getIterScale() const { return iterScale; }

};

class MandelbrotRenderer : public virtual FractalRenderer {
public:
    MandelbrotRenderer(size_t width, size_t height);
    ~MandelbrotRenderer();
    virtual void calculateFractal(unsigned int *out);
};

class JuliaRenderer : public virtual FractalRenderer {
protected:
    float cx, cy;
public:
    JuliaRenderer(size_t width, size_t height, float cx, float cy);
    ~JuliaRenderer();
    virtual void calculateFractal(unsigned int *out);
};

#endif // FRACTAL_HH
