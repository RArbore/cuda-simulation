#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WIDTH 1000
#define HEIGHT 1000
#define NUM_AGENTS 10000
#define AGENTS_ARRAY_SIZE NUM_AGENTS * 3
#define PIXELS WIDTH * HEIGHT
#define TOTAL_STEPS 1000000
#define KERNEL_R 1
#define DECAY 0.97
#define SAMPLE_DIST 2
#define SAMPLE_ANGLE M_PI / 6.0f
#define TURN_SPEED 0.1

__global__
void update_agents (float *agents, uint8_t *image) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_AGENTS) {
        float x = agents[3 * i];
        float y = agents[3 * i + 1];
        float angle = agents[3 * i + 2];

        float sense_left = image[((int) (x + SAMPLE_DIST * cos(angle + SAMPLE_ANGLE))) + ((int) (y + SAMPLE_DIST * sin(angle + SAMPLE_ANGLE))) * WIDTH];
        float sense_center = image[((int) (x + SAMPLE_DIST * cos(angle))) + ((int) (y + SAMPLE_DIST * sin(angle))) * WIDTH];
        float sense_right = image[((int) (x + SAMPLE_DIST * cos(angle - SAMPLE_ANGLE))) + ((int) (y + SAMPLE_DIST * sin(angle - SAMPLE_ANGLE))) * WIDTH];

        uint state = (uint) (x * y * angle + sense_left + sense_center + sense_right);
        state ^= 2747636419u;
        state *= 2654435769u;
        state ^= state >> 16;
        state *= 2654435769u;
        state ^= state >> 16;
        state *= 2654435769u;

        float turn_stren = ((float) (state % 1000)) / 1000.0f;

        if (sense_center < sense_left && sense_center < sense_right) {
            angle += 2.0f * (turn_stren - 0.5f) * TURN_SPEED;
        }
        else if (sense_left > sense_right) {
            angle += turn_stren * TURN_SPEED;
        }
        else {
            angle -= turn_stren * TURN_SPEED;
        }

        float vx = cos(angle);
        float vy = sin(angle);

        x += vx;
        y += vy;

        if (x < 0) {
            x = 0;
            angle = M_PI - angle;
        }
        if (y < 0) {
            y = 0;
            angle *= -1;
        }
        if (x >= WIDTH) {
            x = WIDTH - 1;
            angle = M_PI - angle;
        }
        if (y >= HEIGHT) {
            y = HEIGHT - 1;
            angle *= -1;
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

        cudaMemcpy(host_image, image, PIXELS * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        stbi_write_png("output.png", WIDTH, HEIGHT, 1, host_image, WIDTH * 1);

        if (i % 1000 == 0) printf("%d\n", i);
    }

    //cudaMemcpy(host_agents, agents, AGENTS_ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);


    return 0;
}
