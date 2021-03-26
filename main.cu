#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include <GL/gl.h>
#include <GL/glut.h>

#define WIDTH 1000
#define HEIGHT 1000
#define FOV 45
#define Z_NEAR 1.0f
#define Z_FAR 500.0f

#define KEY_ESC 27

void display () {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.0f);

    glBegin(GL_TRIANGLES);
    glColor3f(0.0f,0.0f,1.0f);
    glVertex3f( 0.0f, 1.0f, 0.0f);
    glColor3f(0.0f,1.0f,0.0f);
    glVertex3f(-1.0f,-1.0f, 0.0f);
    glColor3f(1.0f,0.0f,0.0f);
    glVertex3f( 1.0f,-1.0f, 0.0f);
    
    glEnd();

    glutSwapBuffers();
}



void initialize () {
    glMatrixMode(GL_PROJECTION);
    glViewport(0, 0, WIDTH, HEIGHT);
    glLoadIdentity();
    glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("CUDA Simulation");
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutKeyboardFunc(keyboard);
    initialize();
    glutMainLoop();

    return 0;
}
