#include "gui.hh"
#include "cudagl.hh"
#include "fractal.hh"
#include "utils.hh"
#include "glutils.hh"

// Graphics includes
#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cutil_inline.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <iomanip>

using namespace std;

// Global GUI instance
static GUI * gui = NULL;

// Forward definitions for global glutMainLoop functions
void _display();
void _key(unsigned char key, int x, int y);
void _special(int k, int x, int y);
void _reshape(int width, int height);
void _mouse(int button, int state, int x, int y);
void _motion(int x, int y);
void _idle();
void _cleanup();

////////////////////////////////////////////////////////////////////////////
// The GUI
////////////////////////////////////////////////////////////////////////////

GUI::GUI(size_t width_, size_t height_, int argc, char *argv[]) :
        width(0), height(0),
        fractal(0), cudaGL(0)
//        msPerFrame(0), lastFrameTime(0), lastFPSShownTime(0), frame(0)
{
    if (gui != NULL) {
        cerr << "More than one GUI instance created" << endl;
        exit(-1);
    }
    size_t size = min(width_, height_);
    width = size;
    height = size;
    gui = this;

    int device = cutGetMaxGflopsDeviceId();

    // Explicitly set device
    // cudaSetDevice(device);
    cudaGLSetGLDevice(device);

    // Initialise OpenGL and GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Fractal");

    glewInit();
    if (!glewIsSupported(
            "GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.\n");
        exit(-1);
    }

#if defined (_WIN32)
    if (wglewIsSupported("WGL_EXT_swap_control")) {
        // Disable vertical sync
        wglSwapIntervalEXT(0);
    }
#endif

    cudaGL = new CudaGL(width, height);

    fractal = new MandelbrotRenderer(width, height);
//    fractal->setCenter(-0.77005, 0.099);
    fractal->setPixelScale(min(4.0/width, 4.0/height));

    resetParams();
}

GUI::~GUI() {
    // Delete fractal renderer
    ZAP(fractal);
    // .. and cudaGL
    ZAP(cudaGL);
}

void GUI::start() {
    glutDisplayFunc(_display);
    glutReshapeFunc(_reshape);
    glutMouseFunc(_mouse);
    glutMotionFunc(_motion);
    glutKeyboardFunc(_key);
    glutSpecialFunc(_special);
    glutIdleFunc(_idle);
    atexit(_cleanup);

    glutMainLoop();
}

void GUI::resetParams() {
    fractal->setIterScale(1);
}

void GUI::mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        mouseState.buttonState |= 1 << button;
        mouseState.x1 = x;
        mouseState.y1 = y;
        mouseState.x2 = x;
        mouseState.y2 = y;
        mouseState.ox = fractal->getX();
        mouseState.oy = fractal->getY();
        cout << fixed << setprecision(3) << "Clicked at position " << x << ", " << y << " that corresponds to "
                << fractal->screenXtoReal(x) << ", " << fractal->screenYtoReal(y) << endl;
    }
    else if (state == GLUT_UP) {
        mouseState.buttonState &= ((-1) ^ (1 << button));
        if (button == 0 && fractal != NULL) {
            // Zoom!
            fractal->zoom(min(mouseState.x1, mouseState.x2), min(mouseState.y1, mouseState.y2),
                    max(mouseState.x1, mouseState.x2), max(mouseState.y1, mouseState.y2));
        }
    }
    glutPostRedisplay();
}

void GUI::motion(int x, int y) {
    mouseState.x2 = x;
    mouseState.y2 = y;

    if (mouseState.isPressed(2)) {
        int dx = mouseState.x1 - x;
        int dy = mouseState.y1 - y;
        fractal->setCenter(mouseState.ox + fractal->getPixelScale() * dx, mouseState.oy - fractal->getPixelScale() * dy);
    }

    glutPostRedisplay();
}

void GUI::idle() {
    // TODO: Idle stuff?
    glutPostRedisplay();
}

void GUI::display() {
    // Uncomment if screen clearing is needed
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (fractal->isChanged()) {
        // Map GL buffer to CUDA
        unsigned int* out = cudaGL->mapCudaBuffer();
        // Execute kernel to calculate fractal
        fractal->calculateFractal(out);
        // Unmap GL buffer
        cudaGL->unmapCudaBuffer();
    }

    // Draw current fractal
    cudaGL->drawBufferGL();

    // Draw other stuff
    GLUtils::setup2D(width, height);

    if (mouseState.isPressed(0)) {
        // Draw selection rectangle
        glColor4f(0.0f, 0.8f, 1.0f, 0.2f);
        GLUtils::drawRectangle(mouseState.x1, mouseState.y1, mouseState.x2, mouseState.y2);
    }

    glutSwapBuffers();
    // frame++;
}

// commented out to remove unused parameter warnings in Linux
void GUI::key(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {
    case '\033':
    case 'q':
        exit(0);
        break;
    case '+':
        fractal->setIterScale(fractal->getIterScale() * 1.1);
        break;
    case '-':
        fractal->setIterScale(fractal->getIterScale() / 1.1);
        break;
    case '=':
        resetParams();
        break;
    case 'j': {
        float x = fractal->getX();
        float y = fractal->getY();
        cout << "Switching to Julia fractal located at " << fixed << setprecision(3) << x << ", " << y << endl;
        delete fractal;
        fractal = new JuliaRenderer(width, height, x, y);
    }
    }

    glutPostRedisplay();
}


// commented out to remove unused parameter warnings in Linux
void GUI::special(int key, int /*x*/, int /*y*/) {
    switch (key) {
    case GLUT_KEY_PAGE_UP:
        fractal->setPixelScale(fractal->getPixelScale() / 1.5);
        break;
    case GLUT_KEY_PAGE_DOWN:
        fractal->setPixelScale(fractal->getPixelScale() * 1.5);
        break;
    case GLUT_KEY_UP:
        fractal->setY(fractal->getY() + fractal->getPixelScale() * 10);
        break;
    case GLUT_KEY_DOWN:
        fractal->setY(fractal->getY() - fractal->getPixelScale() * 10);
        break;
    case GLUT_KEY_LEFT:
        fractal->setX(fractal->getX() - fractal->getPixelScale() * 10);
        break;
    case GLUT_KEY_RIGHT:
        fractal->setX(fractal->getX() + fractal->getPixelScale() * 10);
        break;
    }
    glutPostRedisplay();
}

void GUI::resize(size_t width_, size_t height_) {
    width = width_;
    height = height_;
    if (fractal) {
        fractal->resize(width, height);
    }
    if (cudaGL) {
        cudaGL->resize(width, height);
    }
}



////////////////////////////////////////////////////////////////////////////
// Global functions for glutMainLoop
////////////////////////////////////////////////////////////////////////////

void _display() {
    gui->display();
}

void _key(unsigned char key, int x, int y) {
    gui->key(key, x, y);
}

void _special(int key, int x, int y) {
    gui->special(key, x, y);
}

void _reshape(int width, int height) {
    gui->resize(width, height);
}

void _mouse(int button, int state, int x, int y) {
    gui->mouse(button, state, x, y);
}

void _motion(int x, int y) {
    gui->motion(x, y);
}

void _idle() {
    gui->idle();
}

void _cleanup() {
    ZAP(gui);
    // Stop CUDA
    cudaThreadExit();
}

