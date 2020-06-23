#include "fractal.hh"
#include "utils.hh"
#include "gui.hh"

#include <cassert>

int main(int argc, char *argv[]) {
    GUI* gui = new GUI(800, 600, argc, argv);

    gui->start();

    // This should not be reached
    return EXIT_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////
// Public interface                                                        /
////////////////////////////////////////////////////////////////////////////
FractalRenderer::FractalRenderer(size_t width_, size_t height_) :
    width(0), height(0), x(0), y(0), scale(4.0f / width_), dimBlock(0), dimGrid(0), changed(true), iterScale(1.0f) {
    resize(width_, height_);
}

// Destructor: release the device and other structures
FractalRenderer::~FractalRenderer() {
    delete dimBlock;
    delete dimGrid;
}

void FractalRenderer::resize(size_t width_, size_t height_) {
    if (width == width_ && height == height_)
        return;
    float oldsize = width * height;
    width = width_;
    height = height_;
    float newsize = width * height;
    if (oldsize != 0) {
        scale *= oldsize / newsize;
    }
    ZAP(dimBlock);
    ZAP(dimGrid);

    dimBlock = new dim3(BLOCK_WIDTH, BLOCK_HEIGHT);
    dimGrid = new dim3((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);
    changed = true;
}

void FractalRenderer::zoom(int x1, int y1, int x2, int y2) {
    assert(x2 >= x1);
    assert(y2 >= y1);

    if (x1 == x2 || y1 == y2) {
        int cx = (x2 - x1) / 2 + x1;
        int cy = (y2 - y1) / 2 + y1;

        int w = max((int)(width / 4), 1);
        int h = max((int)(height / 4), 1);
        zoom(cx - w, cy - h, cx + w, cy + h);
        return;
    }
    float rx1 = screenXtoReal(x1);
    float rx2 = screenXtoReal(x2);

    float ry2 = screenYtoReal(y1);
    float ry1 = screenYtoReal(y2);
    cout << "Zooming to  " << x1 << ", " << y1 << " - " << x2 << ", " << y2 << endl;
    cout << "... in real " << rx1 << ", " << ry1 << " - " << rx2 << ", " << ry2 << endl;

    // Change center coordinate
    x = (rx2 - rx1) / 2.0 + rx1;
    y = (ry2 - ry1) / 2.0 + ry1;

    // New ranges
    float nxr = rx2 - rx1;
    float nyr = ry2 - ry1;

    // Current ranges
    float cxr = width * scale;
    float cyr = height * scale;
    cout << "Scales are " << (nxr / cxr) << " and " << (nyr / cyr) << endl;

    float ratio = max(nxr / cxr, nyr / cyr);
    scale *= ratio;

    changed = true;
}
