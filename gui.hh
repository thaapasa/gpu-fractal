#ifndef GUI_HH
#define GUI_HH

class CudaGL;
class FractalRenderer;

class MouseState {
public:
    int x1, y1;
    int x2, y2;
    float ox, oy;
    int buttonState;

    MouseState() : x1(0), y1(0), x2(0), y2(0), ox(0), oy(0), buttonState(0) {}
    bool isPressed(int button) { return (buttonState & (1 << button)) != 0; }
};

class GUI {
private:
    size_t width;
    size_t height;

    FractalRenderer *fractal;
    CudaGL *cudaGL;

//    float msPerFrame;
//    int lastFrameTime;
//    int lastFPSShownTime;
//    long unsigned int frame;

    MouseState mouseState;

public:
    GUI(size_t width, size_t height, int argc, char *argv[]);
    ~GUI();

    void start();
    void display();
    void key(unsigned char key, int x, int y);
    void special(int key, int x, int y);
    void resize(size_t width, size_t height);
    void idle();
    void mouse(int button, int state, int x, int y);
    void motion(int x, int y);

    void resetParams();
};


#endif // GUI_HH
