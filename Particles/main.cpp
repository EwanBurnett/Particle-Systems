/*
    Particle Simulation
    Ewan Burnett - 2023
*/

#include <cstdio>
#include <Particle.h>
#include <cstdint>
#include <vector>
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
void InitParticles(std::vector<Particle>& particles) {
    //Iterate over each particle, and assign a random Position, Velocity, Speed and Mass. 
    for (auto& p : particles) {
        p.speed = RandomRange(INITIAL_SPEED_MIN, INITIAL_SPEED_MAX);
        p.mass = RandomRange(INITIAL_SIZE_MIN, INITIAL_SIZE_MAX);

        p.position.x = RandomRange(INITIAL_POS_MIN.x, INITIAL_POS_MAX.x);
        p.position.y = RandomRange(INITIAL_POS_MIN.y, INITIAL_POS_MAX.y);
        p.position.z = RandomRange(INITIAL_POS_MIN.z, INITIAL_POS_MAX.z);

        //Velocity is clamped between -1 and 1, as it represents a direction vector. 
        p.velocity.x = RandomRange(-1.0f, 1.0f); 
        p.velocity.y = RandomRange(-1.0f, 1.0f);
        p.velocity.z = RandomRange(-1.0f, 1.0f);
    }
}


//Update all particle states
void UpdateParticles(std::vector<Particle>& particles, const float deltaTime) {
    //Iterate over each particle, and update their positions. 
    for (auto& p : particles) {
        (p.mass < 0.0f) ? p.mass = FLT_EPSILON : NULL;   //Ensure the size is greater than 0. 
        float k = (1.0f / p.mass);      //Particles with more mass move slower than those with less, even at the same speed. 
        
        p.position.x += (p.velocity.x * p.speed * k * deltaTime);
        p.position.y += (p.velocity.y * p.speed * k * deltaTime);
        p.position.z += (p.velocity.z * p.speed * k * deltaTime);
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
    std::vector<Particle> particles(num_particles);
    {
        const auto init_start = std::chrono::steady_clock::now();
        InitParticles(particles);
        const auto init_end = std::chrono::steady_clock::now();
        const float initTime = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count() / 1000.0f;  //Compute how long particle initialization took in ms. 

        printf("Particle Initialization Complete in %fms.\n", initTime);
    }

    //Simulate Forever
    float deltaTime = 0.0f;
    float elapsedTime = 0.0f; 

    while (true) {
        const auto update_start = std::chrono::steady_clock::now();
        UpdateParticles(particles, deltaTime);
        const auto update_end = std::chrono::steady_clock::now();

        deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_start).count() / 1000.0f; //Delta Time is in Milliseconds
        elapsedTime += deltaTime;

        printf("\rUpdated %u Particles in %fms\t(%03.2fs elapsed)", num_particles, deltaTime, elapsedTime);
    }

}
