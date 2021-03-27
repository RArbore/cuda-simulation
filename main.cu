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

#define WIDTH 300
#define HEIGHT 300
#define NUM_AGENTS 500
#define AGENTS_ARRAY_SIZE NUM_AGENTS * 3
#define PIXELS WIDTH * HEIGHT
#define TOTAL_STEPS 100000
#define KERNEL_R 1
#define DECAY 0.92
#define TRAVEL_SPEED 3
#define SAMPLE_DIST 4.0f
#define SAMPLE_ANGLE M_PI / 12.0f
#define TURN_SPEED 1.2
#define KEY_ESC 27

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
            angle += 2.0f * (turn_stren - 0.5f) * TURN_SPEED;
        }
        else if (sense_left > sense_right) {
            angle += turn_stren * TURN_SPEED;
        }
        else if (sense_right > sense_left) {
            angle -= turn_stren * TURN_SPEED;
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
        int y = i / HEIGHT;
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

int main (int argc, char *argv[]) {

    int i;

    float *host_agents;
    uint8_t *host_image;
    host_agents = (float *) malloc(AGENTS_ARRAY_SIZE * sizeof(float));
    host_image = (uint8_t *) malloc(PIXELS * sizeof(uint8_t));

    float *agents;
    uint8_t *image;

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
