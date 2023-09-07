/*
    Particle Simulation
    Ewan Burnett - 2023
*/
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cfloat>
#include <chrono>
#include <random>
#include <vector>
#include <thread>
#include <barrier>
#include <condition_variable>
#include <mutex>
#include "../include/Particle.h"
#include "../include/CUDAUpdate.cuh"
#include <raylib.h>

using namespace Simulation;
#define MAX_BATCH_ELEMENTS 8096 

bool g_UseCuda = false;

constexpr uint32_t DEFAULT_NUM_PARTICLES = 1u << 20u;

//Initialisation Defaults
const float INITIAL_SPEED_MIN = 100.0f;
const float INITIAL_SPEED_MAX = 1000.0f;
const float INITIAL_SIZE_MIN = 1.0f;
const float INITIAL_SIZE_MAX = 10.0f;
const Simulation::Vector3 INITIAL_POS_MIN = { 250.0f, 200.0f, -100.0f };
const Simulation::Vector3 INITIAL_POS_MAX = { 750.0f, 600.0f, 100.0f };


//Generates a random number between min and max. 
float RandomRange(float min, float max) {
    thread_local std::mt19937 generator(std::random_device{}());  //Generate random numbers using a Mersenne Twister
    std::uniform_real_distribution<> dist(min, max);

    return static_cast<float>(dist(generator));
}


//Initialize the Particle data. 
void InitParticles(Particles& particles, const size_t startIdx, const size_t count) {
    //Iterate over each particle, and assign a random Position, Velocity, Speed and Mass. 
    for (size_t i = startIdx; i < (startIdx + count); i++) {
        
        particles.speeds[i] = RandomRange(INITIAL_SPEED_MIN, INITIAL_SPEED_MAX);
        particles.masses[i] = RandomRange(INITIAL_SIZE_MIN, INITIAL_SIZE_MAX);

        particles.positions[i].x = RandomRange(INITIAL_POS_MIN.x, INITIAL_POS_MAX.x);
        particles.positions[i].y = RandomRange(INITIAL_POS_MIN.y, INITIAL_POS_MAX.y);
        particles.positions[i].z = RandomRange(INITIAL_POS_MIN.z, INITIAL_POS_MAX.z);

        //Velocity is clamped between -1 and 1, as it represents a direction vector. 
        particles.velocities[i].x = RandomRange(-1.0f, 1.0f); 
        particles.velocities[i].y = RandomRange(-1.0f, 1.0f);
        particles.velocities[i].z = RandomRange(-1.0f, 1.0f);

        uint8_t r = (uint8_t)(RandomRange(0.0f, 1.0f) * 255.99f);
        uint8_t g = (uint8_t)(RandomRange(0.0f, 1.0f) * 255.99f);
        uint8_t b = (uint8_t)(RandomRange(0.0f, 1.0f) * 255.99f);

        Color c;
        c.r = r; 
        c.g = g;
        c.b = b;
        c.a = 255;
        particles.colours[i] = c;
        int a = 1;
    }
}


//Update all particle states
void UpdateParticles(Particles& particles, const size_t startIdx, const size_t count, const float deltaTime) {
    //Iterate over each particle, and update their positions. 
    for (size_t i = startIdx; i < (startIdx + count); i++) {
        (particles.masses[i] < 0.0f) ? particles.masses[i] = FLT_EPSILON : (float)NULL;   //Ensure the size is greater than 0. 
        const float k = (1.0f / particles.masses[i]);      //Particles with more mass move slower than those with less, even at the same speed. 
        
        particles.positions[i].x += (particles.velocities[i].x * particles.speeds[i] * k * deltaTime);
        particles.positions[i].y += (particles.velocities[i].y * particles.speeds[i] * k * deltaTime);
        particles.positions[i].z += (particles.velocities[i].z * particles.speeds[i] * k * deltaTime);
    }
}


void DrawParticles(Particles& particles, const size_t numParticles){

    BeginDrawing();
    ClearBackground(BLACK);

    for(int i = 0; i < numParticles; i++){
        DrawPixel(particles.positions[i].x, particles.positions[i].y, particles.colours[i]);
    }

    EndDrawing();

}

//Launch the application
//  -p / -particles : How many particles to simulate
//  -t / -threads : How many threads to launch
int main(int argc, const char** argv){
    
    const int windowWidth = 1000;
    const int windowHeight = 800;

    InitWindow(windowWidth, windowHeight, "Particle Simulation");

    uint32_t num_particles = DEFAULT_NUM_PARTICLES;
    uint32_t num_threads = std::thread::hardware_concurrency();     //Use as many threads as possible by default.

    //Check if any CMD arguments were passed through
    if (argc > 1) { //argv[0] is the name of the program. 
        for (int i = 1; i < argc; i++) {
            //Parse the Argument
            if(std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "-particles") == 0) {
                num_particles = std::atoi(argv[i + 1]); //The caller specified how many particles to use, so retrieve them.
            }

            if (std::strcmp(argv[i], "-t") == 0 || std::strcmp(argv[i], "-threads") == 0) {
                num_threads = std::atoi(argv[i + 1]);   //The caller specified how many threads to run, so retrieve them. 
            }
        }
    }

    printf("Simulating %u Particles utilizing %d threads.\nCTRL + C to exit.\n", num_particles, num_threads);

    std::vector<std::thread> threads(num_threads - 1);
    const uint32_t block_size = (num_particles / num_threads);  //How many particles each thread updates
    const uint32_t block_remainder = num_particles % num_threads;   //Any remaining particles left out of the evenly-sized blocks. 
    
    printf("%d Particles Split into blocks of %d (+ %d)\n",num_particles, block_size, block_remainder);


    //Create the number of particles specified
    Particles particles(num_particles);
    {
        //Dispatch n - 1 threads to initialise the particles, as the process already owns 1 thread. 
        for (size_t i = 0; i < (num_threads - 1); i++) {
            threads[i] = std::thread(InitParticles, std::ref(particles), block_size * i, block_size);
        }

        const auto init_start = std::chrono::steady_clock::now();
        
        InitParticles(particles, block_size * (num_threads - 1), (block_size + block_remainder));   //The main thread works on its own block, including any remaining particles not updated. 
        
        //Wait for all threads to complete. 
        for (auto& t : threads) {
            t.join();
        }

        //Initialize CUDA as well. 
        CUDAInit(&particles, num_particles);
        const auto init_end = std::chrono::steady_clock::now();
        const float initTime = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count() / 1000.0f;  //Compute how long particle initialization took in ms. 

        printf("Particle Initialization Complete in %fms.\n", initTime);
    }
 
    float updateTime = 0.0f;
    while(true)
    {
        const auto update_start = std::chrono::steady_clock::now();   //Finish the current frame
        CUDAUpdate(&particles, num_particles, updateTime);        
        const auto update_end = std::chrono::steady_clock::now();   //Finish the current frame
        updateTime = std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_start).count() / 1000.0f; //Delta Time is in Milliseconds

    printf("\r[CUDA] Updated %d particles in %fms", num_particles, updateTime);   
    DrawParticles(particles, num_particles);
    }

    //Simulate Particles across our Threads.
    float deltaTime = 0.0f;
    float elapsedTime = 0.0f; 
    auto update_start = std::chrono::steady_clock::now();

    //Create synchronisation primitives to control our threads. 
    std::mutex lCuda;
    std::unique_lock lkCuda(lCuda);
    std::condition_variable cvCuda;

    std::barrier particle_sync(num_threads, [&] () noexcept {   //When all threads "arrive" at the barrier, this code will execute. 
        const auto update_end = std::chrono::steady_clock::now();   //Finish the current frame

        deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_start).count() / 1000.0f; //Delta Time is in Milliseconds
        elapsedTime += deltaTime;
        printf("\rUpdated %u Particles in %fms\t(%03.2fs elapsed)", num_particles, deltaTime, elapsedTime);
        
        update_start = std::chrono::steady_clock::now();
        });

    //Kick the update job on n threads. 
    for (size_t i = 0; i < (num_threads - 1); i++) {
        threads[i] = std::thread([&] {
            while (true) {
            cvCuda.wait(lkCuda, [&](){
                    return !g_UseCuda;
                    });
                UpdateParticles(particles, block_size * i, block_size, deltaTime);

                particle_sync.arrive_and_wait();    //Wait for the other threads to finish before continuing. 
            }
            });
    }
    
    //Perform the same work on the main thread
    while (true) {
        //If we're using CUDA, then use the main thread to dispatch processing. 
        if(g_UseCuda){
            CUDAUpdate(&particles, num_particles, deltaTime);        
        }
        else{
            UpdateParticles(particles, block_size * (num_threads - 1), (block_size + block_remainder), deltaTime);

            particle_sync.arrive_and_wait();    //Wait for the other threads to finish before continuing. 
            DrawParticles(particles, num_particles);
        }

    }

    //When we're done, make sure to join all threads before exiting.
    for (auto& t : threads) {
        t.join();
    }

    CloseWindow(); 
}
