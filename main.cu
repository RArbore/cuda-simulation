#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <GL/gl.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WIDTH 1920
#define HEIGHT 1080
#define NUM_AGENTS 100000
#define AGENTS_ARRAY_SIZE NUM_AGENTS * 3
#define PIXELS WIDTH * HEIGHT
#define BYTES_PER_PIXEL 4
#define PIXELS_SIZE PIXELS * BYTES_PER_PIXEL
#define AGENTS_DIV 10
#define PIXEL_DIV 256
#define TOTAL_STEPS 100000
#define KERNEL_R 1
#define DECAY 0.99f
#define TRAVEL_SPEED 1
#define SAMPLE_DIST 4.0f
#define SAMPLE_ANGLE M_PI / 12.0f
#define TURN_SPEED 0.3f
#define RANDOM_STREN 0.05f
#define KEY_ESC 27

float *host_agents;
uint8_t *host_image;
float *agents;
uint8_t *image;

GLuint tex;
cudaGraphicsResource_t cuda_resource;

__global__
void update_agents (float *agents, uint8_t *image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_AGENTS) {
        float x = agents[3 * i];
        float y = agents[3 * i + 1];
        float angle = agents[3 * i + 2];
        float sense_left = 0.0f;
        float sense_center = 0.0f;
        float sense_right = 0.0f;
        float sample_x_left = x + SAMPLE_DIST * cos(angle + SAMPLE_ANGLE);
        float sample_y_left = y + SAMPLE_DIST * sin(angle + SAMPLE_ANGLE);
        float sample_x_center = x + SAMPLE_DIST * cos(angle);
        float sample_y_center = y + SAMPLE_DIST * sin(angle);
        float sample_x_right = x + SAMPLE_DIST * cos(angle - SAMPLE_ANGLE);
        float sample_y_right = y + SAMPLE_DIST * sin(angle - SAMPLE_ANGLE);
        float rsx, rsy;
        
        for (float sx = sample_x_left - KERNEL_R; sx <= sample_x_left + KERNEL_R; sx++) {
            for (float sy = sample_y_left - KERNEL_R; sy <= sample_y_left + KERNEL_R; sy++) {
                rsx = sx;
                rsy = sy;
                while (rsx < 0) rsx += WIDTH;
                while (rsy < 0) rsy += HEIGHT;
                while (rsx >= WIDTH) rsx -= WIDTH;
                while (rsy >= HEIGHT) rsy -= HEIGHT;
                sense_left += (float) image[BYTES_PER_PIXEL * (((int) (rsx)) + ((int) (rsy)) * WIDTH)];
            }
        }
        for (float sx = sample_x_center - KERNEL_R; sx <= sample_x_center + KERNEL_R; sx++) {
            for (float sy = sample_y_center - KERNEL_R; sy <= sample_y_center + KERNEL_R; sy++) {
                rsx = sx;
                rsy = sy;
                while (rsx < 0) rsx += WIDTH;
                while (rsy < 0) rsy += HEIGHT;
                while (rsx >= WIDTH) rsx -= WIDTH;
                while (rsy >= HEIGHT) rsy -= HEIGHT;
                sense_center += (float) image[BYTES_PER_PIXEL * (((int) (rsx)) + ((int) (rsy)) * WIDTH)];
            }
        }
        for (float sx = sample_x_right - KERNEL_R; sx <= sample_x_right + KERNEL_R; sx++) {
            for (float sy = sample_y_right - KERNEL_R; sy <= sample_y_right + KERNEL_R; sy++) {
                rsx = sx;
                rsy = sy;
                while (rsx < 0) rsx += WIDTH;
                while (rsy < 0) rsy += HEIGHT;
                while (rsx >= WIDTH) rsx -= WIDTH;
                while (rsy >= HEIGHT) rsy -= HEIGHT;
                sense_right += (float) image[BYTES_PER_PIXEL * (((int) (rsx)) + ((int) (rsy)) * WIDTH)];
            }
        }

        uint state = (uint) (x * y * angle + sense_left + sense_center + sense_right);
        state ^= 2747636419u;
        state *= 2654435769u;
        state ^= state >> 16;
        state *= 2654435769u;
        state ^= state >> 16;
        state *= 2654435769u;

        float turn_stren = ((float) (state % 1000)) / 1000.0f;

        if (sense_center > sense_left && sense_center > sense_right) {

        }
        else if (sense_center < sense_left && sense_center < sense_right) {
            angle += 2.0f * (turn_stren - 0.5f) * TURN_SPEED * RANDOM_STREN;
        }
        else if (sense_left > sense_right) {
            angle += pow(turn_stren, RANDOM_STREN) * TURN_SPEED;
        }
        else if (sense_right > sense_left) {
            angle -= pow(turn_stren, RANDOM_STREN) * TURN_SPEED;
        }

        //float center_angle = atan((y - 539) / (x - 959));
        //if (x >= 960) {
        //    center_angle -= M_PI;
        //}

        //float center_dist_2 = (y - 539) * (y - 539) + (x - 959) * (x - 959);

        //center_angle += M_PI/2 + 0.01f;

        float vx = cos(angle);
        float vy = sin(angle);

        //float vx_c = cos(center_angle + M_PI / 2) * 0.1f;
        //float vy_c = sin(center_angle + M_PI / 2) * 0.1f;

        //float mult = (-cos(4.0f * angle) + 1.2f) / 2.0f;
        //float mult = (cos(-center_angle + angle) + 4.0f) / (center_dist_2 / 200000.0f + 4.0f);
        float mult = 1.0f;

        vx = vx * mult;
        vy = vy * mult;

        for (int iter = 0; iter < TRAVEL_SPEED; iter++) {
            x += vx;
            y += vy;

            while (x < 0) x += WIDTH;
            while (y < 0) y += HEIGHT;
            while (x >= WIDTH) x -= WIDTH;
            while (y >= HEIGHT) y -= HEIGHT;

            image[BYTES_PER_PIXEL * (((int) x) + ((int) y) * WIDTH)] = 255;
        }

        agents[3 * i] = x;
        agents[3 * i + 1] = y;
        agents[3 * i + 2] = angle;

        image[BYTES_PER_PIXEL * (((int) x) + ((int) y) * WIDTH)] = 127;
        image[BYTES_PER_PIXEL * (((int) x) + ((int) y) * WIDTH) + 1] = 0;
        image[BYTES_PER_PIXEL * (((int) x) + ((int) y) * WIDTH) + 2] = 255;
    }
}

__global__
void update_image (float *agents, uint8_t *image) {
    const float KERNEL[3][3] = {{0.005, 0.12, 0.005},
                                {0.12, 0.5, 0.12},
                                {0.005, 0.12, 0.005}};
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < PIXELS) {
        int x = i % WIDTH;
        int y = i / WIDTH;
        float total = 0;
        int count = 0;
        int left = x - KERNEL_R;
        int right = x + KERNEL_R;
        int up = y - KERNEL_R;
        int down = y + KERNEL_R;
        if (left < 0) left = 0;
        if (right > WIDTH - 1) right = WIDTH - 1;
        if (up < 0) up = 0;
        if (down > HEIGHT - 1) down = HEIGHT - 1;
        for (int ix = left; ix <= right; ix++) {
            for (int iy = up; iy <= down; iy++) {
                total += ((float) image[BYTES_PER_PIXEL * (ix + iy * WIDTH)]) * KERNEL[iy-y+KERNEL_R][ix-x+KERNEL_R];
                count++;
            }
        }
        image[BYTES_PER_PIXEL * i] = (int) (total * DECAY);
        image[BYTES_PER_PIXEL * i + 1] = (int) (total * DECAY) * 0;
        image[BYTES_PER_PIXEL * i + 2] = (int) (total * DECAY) * 1.5;
    }
}

__global__
void update_surface(cudaSurfaceObject_t cuda_surface, uint8_t *image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < PIXELS_SIZE) {
        int x = i % (BYTES_PER_PIXEL * WIDTH);
        int y = i / (BYTES_PER_PIXEL * WIDTH);
        surf2Dwrite<uint8_t>(image[i], cuda_surface, x, y);
    }
}

void invokeRenderingKernel(cudaSurfaceObject_t cuda_surface) {
    update_image<<<PIXELS_SIZE/PIXEL_DIV, PIXEL_DIV>>>(agents, image);
    update_agents<<<NUM_AGENTS/AGENTS_DIV, AGENTS_DIV>>>(agents, image);
    update_surface<<<PIXELS_SIZE/PIXEL_DIV, PIXEL_DIV>>>(cuda_surface, image);
}

void initializeGL () {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    cudaGraphicsGLRegisterImage(&cuda_resource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

void displayGL() {
    cudaGraphicsMapResources(1, &cuda_resource);
    cudaArray_t cuda_array;
    cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
    cudaResourceDesc cuda_array_resource_desc;
    cuda_array_resource_desc.resType = cudaResourceTypeArray;
    cuda_array_resource_desc.res.array.array = cuda_array;
    cudaSurfaceObject_t cuda_surface;
    cudaCreateSurfaceObject(&cuda_surface, &cuda_array_resource_desc);
    invokeRenderingKernel(cuda_surface);
    cudaDestroySurfaceObject(cuda_surface);
    cudaGraphicsUnmapResources(1, &cuda_resource);

    glBindTexture(GL_TEXTURE_2D, tex);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glutSwapBuffers();
}

void keyboardGL (unsigned char key, int mousePositionX, int mousePositionY) {
    switch (key) {
        case KEY_ESC:
            exit(0);
            break;
        default:
            break;
    }
}

int main (int argc, char *argv[]) {

    srand(time(0));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("CUDA Simulation");
    glutDisplayFunc(displayGL);
    glutIdleFunc(displayGL);
    glutKeyboardFunc(keyboardGL);
    initializeGL();

    int i;
    
    host_agents = (float *) malloc(AGENTS_ARRAY_SIZE * sizeof(float));
    host_image = (uint8_t *) malloc(PIXELS_SIZE * sizeof(uint8_t));

    cudaMalloc(&agents, AGENTS_ARRAY_SIZE * sizeof(float));
    cudaMalloc(&image, PIXELS_SIZE * sizeof(uint8_t));

    float angle;
    for (i = 0; i < NUM_AGENTS; i++) {

        host_agents[3 * i] = (float) (rand() % WIDTH);
        host_agents[3 * i + 1] = (float) (rand() % HEIGHT);

        host_agents[3 * i] = (float) (WIDTH / 2);
        host_agents[3 * i + 1] = (float) (HEIGHT / 2);

        angle = (float) (rand() % 360);
        angle *= M_PI/180.0f;

        host_agents[3 * i + 2] = angle;
    }

    for (i = 0; i < PIXELS_SIZE; i++) {
        host_image[i] = 0;
    }

    cudaMemcpy(agents, host_agents, AGENTS_ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(image, host_image, PIXELS_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);

    glutMainLoop(); 
    
    return 0;
}
