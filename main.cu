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
#define NUM_AGENTS 50000
#define AGENTS_ARRAY_SIZE NUM_AGENTS * 3
#define PIXELS WIDTH * HEIGHT
#define TOTAL_STEPS 100000
#define KERNEL_R 1
#define DECAY 0.98f
#define TRAVEL_SPEED 2
#define SAMPLE_DIST 5.0f
#define SAMPLE_ANGLE M_PI / 8.0f
#define TURN_SPEED 0.4f
#define RANDOM_STREN 0.0f
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
        
        for (float sx = sample_x_left - KERNEL_R; sx <= sample_x_left + KERNEL_R; sx++) {
            for (float sy = sample_y_left - KERNEL_R; sy <= sample_y_left + KERNEL_R; sy++) {
                if (sx >= 0  && sy >= 0 && sx < WIDTH && sy < HEIGHT) {
                    sense_left += (float) image[((int) (sx)) + ((int) (sy)) * WIDTH];
                }
            }
        }
        for (float sx = sample_x_center - KERNEL_R; sx <= sample_x_center + KERNEL_R; sx++) {
            for (float sy = sample_y_center - KERNEL_R; sy <= sample_y_center + KERNEL_R; sy++) {
                if (sx >= 0  && sy >= 0 && sx < WIDTH && sy < HEIGHT) {
                    sense_center += (float) image[((int) (sx)) + ((int) (sy)) * WIDTH];
                }
            }
        }
        for (float sx = sample_x_right - KERNEL_R; sx <= sample_x_right + KERNEL_R; sx++) {
            for (float sy = sample_y_right - KERNEL_R; sy <= sample_y_right + KERNEL_R; sy++) {
                if (sx >= 0  && sy >= 0 && sx < WIDTH && sy < HEIGHT) {
                    sense_right += (float) image[((int) (sx)) + ((int) (sy)) * WIDTH];
                }
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

        float vx = cos(angle);
        float vy = sin(angle);

        for (int iter = 0; iter < TRAVEL_SPEED; iter++) {
            x += vx;
            y += vy;

            if (x < 0) {
                x = 0;
                angle = turn_stren * 2 * M_PI;
                float vx = cos(angle);
                float vy = sin(angle);
            }
            if (y < 0) {
                y = 0;
                angle = turn_stren * 2 * M_PI;
                float vx = cos(angle);
                float vy = sin(angle);
            }
            if (x >= WIDTH) {
                x = WIDTH - 1;
                angle = turn_stren * 2 * M_PI;
                float vx = cos(angle);
                float vy = sin(angle);
            }
            if (y >= HEIGHT) {
                y = HEIGHT - 1;
                angle = turn_stren * 2 * M_PI;
                float vx = cos(angle);
                float vy = sin(angle);
            }

            image[((int) x) + ((int) y) * WIDTH] = 255;
        }

        if (x < 0) {
            x = 0;
            angle = turn_stren * 2 * M_PI;
        }
        if (y < 0) {
            y = 0;
            angle = turn_stren * 2 * M_PI;
        }
        if (x >= WIDTH) {
            x = WIDTH - 1;
            angle = turn_stren * 2 * M_PI;
        }
        if (y >= HEIGHT) {
            y = HEIGHT - 1;
            angle = turn_stren * 2 * M_PI;
        }

        agents[3 * i] = x;
        agents[3 * i + 1] = y;
        agents[3 * i + 2] = angle;

        image[((int) x) + ((int) y) * WIDTH] = 255;
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
                total += ((float) image[ix + iy * WIDTH]) * KERNEL[iy-y+KERNEL_R][ix-x+KERNEL_R];
                count++;
            }
        }
        image[i] = (int) (total * DECAY);
    }
}

__global__
void update_surface(cudaSurfaceObject_t cuda_surface, uint8_t *image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < PIXELS) {
        int x = i % WIDTH;
        int y = i / WIDTH;
        surf2Dwrite<uint8_t>(image[i], cuda_surface, x, y);
    }
}

void invokeRenderingKernel(cudaSurfaceObject_t cuda_surface) {
    update_image<<<PIXELS/10, 10>>>(agents, image);
    update_agents<<<NUM_AGENTS/10, 10>>>(agents, image);
    update_surface<<<PIXELS/10, 10>>>(cuda_surface, image);
}

void initializeGL () {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, WIDTH, HEIGHT, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
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
    cudaStreamSynchronize(0);
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
    host_image = (uint8_t *) malloc(PIXELS * sizeof(uint8_t));

    cudaMalloc(&agents, AGENTS_ARRAY_SIZE * sizeof(float));
    cudaMalloc(&image, PIXELS * sizeof(uint8_t));

    float angle;
    for (i = 0; i < NUM_AGENTS; i++) {
        host_agents[3 * i] = (float) (rand() % WIDTH);
        host_agents[3 * i + 1] = (float) (rand() % HEIGHT);
        angle = (float) (rand() % 360);
        angle *= M_PI/180.0f;
        host_agents[3 * i + 2] = angle;
    }

    for (i = 0; i < PIXELS; i++) {
        host_image[i] = 0;
    }

    cudaMemcpy(agents, host_agents, AGENTS_ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(image, host_image, PIXELS * sizeof(uint8_t), cudaMemcpyHostToDevice);

    glutMainLoop(); 
    
    for (i = 0; i < TOTAL_STEPS; i++) {
        update_image<<<PIXELS/100, 100>>>(agents, image);
        update_agents<<<NUM_AGENTS/100, 100>>>(agents, image);

        if (i % 1000 == 0) printf("%d\n", i);
    }

    //cudaMemcpy(host_agents, agents, AGENTS_ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_image, image, PIXELS * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    stbi_write_png("output.png", WIDTH, HEIGHT, 1, host_image, WIDTH * 1);

    return 0;
}
