#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include <GL/gl.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#define WIDTH 800
#define HEIGHT 600
#define FOV 45
#define Z_NEAR 1.0f
#define Z_FAR 500.0f

#define KEY_ESC 27

int count = 0;

void initialize () {
    unsigned char tex_data[64*3];
    for (int i = 0; i < 64*3; ++i) {
        tex_data[i] = i;
    }

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data);
    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_PROJECTION);
    glOrtho(0, WIDTH, 0, HEIGHT, -1.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);

}

void display () {
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, 1);

    unsigned char tex_data[64*3];
    for (int i = 0; i < 64*3; ++i) {
        tex_data[i] = (i + count) % 256;
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 8, 8, GL_RGB, GL_UNSIGNED_BYTE, tex_data);

    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);

    glTexCoord2i(0, 0); glVertex2i(0, 0);
    glTexCoord2i(0, 1); glVertex2i(0, HEIGHT);
    glTexCoord2i(1, 1); glVertex2i(WIDTH, HEIGHT);
    glTexCoord2i(1, 0); glVertex2i(WIDTH, 0);

    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    count += 1;
}

void keyboard (unsigned char key, int mousePositionX, int mousePositionY) {
    switch (key) {
        case KEY_ESC:
            exit(0);
            break;
        default:
            break;
    }
}

int main (int argc, char *argv[]) {

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("CUDA Simulation");
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutKeyboardFunc(keyboard);
    initialize();
    glutMainLoop();

    return 0;
}
