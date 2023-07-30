/*
    Particle Simulation
    Ewan Burnett - 2023
*/

#include <cstdio>
#include <Particle.h>
#include <cstdint>
#include <chrono>
#include <random>



constexpr uint32_t DEFAULT_NUM_PARTICLES = 1u << 25u;

//Initialisation Defaults
const float INITIAL_SPEED_MIN = 0.0f;
const float INITIAL_SPEED_MAX = 100.0f;
const float INITIAL_SIZE_MIN = FLT_EPSILON;
const float INITIAL_SIZE_MAX = 100.0f;
const Vector3 INITIAL_POS_MIN = { -100.0f, -100.0f, -100.0f };
const Vector3 INITIAL_POS_MAX = { 100.0f, 100.0f, 100.0f };

//Generates a random number between min and max. 
float RandomRange(float min, float max) {
    static std::mt19937 generator(std::random_device{}());  //Generate random numbers using a Mersenne Twister
    std::uniform_real_distribution<> dist(min, max);

    return static_cast<float>(dist(generator));
}

//Initialize the Particle data. 
void InitParticles(Particles& particles, const size_t num_particles) {
    //Iterate over each particle, and assign a random Position, Velocity, Speed and Mass. 
    for (size_t i = 0; i < num_particles; i++) {
        
        particles.speeds[i] = RandomRange(INITIAL_SPEED_MIN, INITIAL_SPEED_MAX);
        particles.masses[i] = RandomRange(INITIAL_SIZE_MIN, INITIAL_SIZE_MAX);

        particles.positions[i].x = RandomRange(INITIAL_POS_MIN.x, INITIAL_POS_MAX.x);
        particles.positions[i].y = RandomRange(INITIAL_POS_MIN.y, INITIAL_POS_MAX.y);
        particles.positions[i].z = RandomRange(INITIAL_POS_MIN.z, INITIAL_POS_MAX.z);

        //Velocity is clamped between -1 and 1, as it represents a direction vector. 
        particles.velocities[i].x = RandomRange(-1.0f, 1.0f); 
        particles.velocities[i].y = RandomRange(-1.0f, 1.0f);
        particles.velocities[i].z = RandomRange(-1.0f, 1.0f);
    }
}


//Update all particle states
void UpdateParticles(Particles& particles, const size_t num_particles, const float deltaTime) {
    //Iterate over each particle, and update their positions. 
    for (size_t i = 0; i < num_particles; i++) {
        (particles.masses[i] < 0.0f) ? particles.masses[i] = FLT_EPSILON : NULL;   //Ensure the size is greater than 0. 
        const float k = (1.0f / particles.masses[i]);      //Particles with more mass move slower than those with less, even at the same speed. 
        
        particles.positions[i].x += (particles.velocities[i].x * particles.speeds[i] * k * deltaTime);
        particles.positions[i].y += (particles.velocities[i].y * particles.speeds[i] * k * deltaTime);
        particles.positions[i].z += (particles.velocities[i].z * particles.speeds[i] * k * deltaTime);
    }
}


//Launch the application
//  -p / - particles : How many particles to simulate
int main(int argc, const char** argv){
    
    uint32_t num_particles = DEFAULT_NUM_PARTICLES;
    //Check if any CMD arguments were passed through
    if (argc > 1) { //argv[0] is the name of the program. 
        for (int i = 1; i < argc; i++) {
            //Parse the Argument
            if(std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "-particles") == 0) {
                num_particles = std::atoi(argv[i + 1]); //The caller specified how many particles to use, so retrieve them.
            }
        }
    }

    printf("Simulating %u Particles.\nCTRL + C to exit.\n", num_particles);

    //Create the number of particles specified
    Particles particles(num_particles);
    {
        const auto init_start = std::chrono::steady_clock::now();
        InitParticles(particles, num_particles);
        const auto init_end = std::chrono::steady_clock::now();
        const float initTime = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count() / 1000.0f;  //Compute how long particle initialization took in ms. 

        printf("Particle Initialization Complete in %fms.\n", initTime);
    }

    //Simulate Forever
    float deltaTime = 0.0f;
    float elapsedTime = 0.0f; 

    while (true) {
        const auto update_start = std::chrono::steady_clock::now();
        UpdateParticles(particles, num_particles, deltaTime);
        const auto update_end = std::chrono::steady_clock::now();

        deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_start).count() / 1000.0f; //Delta Time is in Milliseconds
        elapsedTime += deltaTime;

        printf("\rUpdated %u Particles in %fms\t(%03.2fs elapsed)", num_particles, deltaTime, elapsedTime);
    }

}
