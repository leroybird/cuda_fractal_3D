#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <SFML/Graphics.hpp>

#include <SFML/Graphics.hpp>
#include <algorithm>
#include <stdint.h>
#include "math.h"
#include "simple_math.h"
#include <stdio.h>

const int WIDTH = 1024;
const int HEIGHT = 1024;
const int IMG_CH = 4;

// Maximum iterations for calculating the mandelbulb iterations
const int MAX_ITER = 200;
// Stop marching when we get too close
const float MIN_DIST = 1e-5;

// How many ray marches we before stopping
const int MAX_RAY_ITER = 128;


void mandelbrotGPU(sf::Uint8 *, float);

#define cudaAssertSuccess(ans)                     \
  {                                                \
    _cudaAssertSuccess((ans), __FILE__, __LINE__); \
  }
inline void _cudaAssertSuccess(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "_cudaAssertSuccess: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

__host__ __device__ float boxFold(float component)
{
  if (component > 1)
    component = 2 - component;
  else if (component < -1)
    component = -2 - component;
  return component;
}

// Define the distance estimation function so we can dynamically change the distance function at runtime
typedef float (*distanceFunction)(float3, float);

__host__ __device__ float distEstMandelBox(float3 pos, float time)
{
  // Animation Code
  // For MandelBox, we aviod the range [1, -1] and instead display from [-3, -1] and [1, 3]
  float scale = fmodf(time / 4, 4) - 2;
  scale =  scale < 0 ? scale - 1 : scale + 1;

  // Returns the distance (magnitude) from a point to the mandelbulb fractal.
  float3 offset = pos;
  float dr = 1.0;

  for (int n = 0; n < MAX_ITER; n++)
  {
    // bail out
    if (length(pos) > 16)
      break;

    // box fold
    pos = make_float3(boxFold(pos.x), boxFold(pos.y), boxFold(pos.z));

    // Sphere fold
    float mag = length(pos);
    if (mag < 0.5)
    {
      pos = pos * 4;
    }
    else if (mag < 1)
    {
      pos = pos * (1.0 / (mag * mag));
    }

    pos = pos * scale + offset;
    dr = dr * abs(scale) + 1.0;
  }

  return length(pos) / abs(dr);
}

__host__ __device__ float distEstMandelBulb(float3 pos, float time)
{
  // Returns the distance (magnitude) from a point to the mandelbulb fractal. Based off
  // http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
  float3 z = pos;

  // Animation code
  float power = fmodf(time / 3, 3) + 1;
  power =  power * power;
  
  float dr = 1.0;
  float r = 0.0;
  for (int i = 0; i < MAX_ITER; i++)
  {
    //bail out.
    r = length(z);
    if (r > 16)
      break;

    // Convert to polar coords
    float theta = acosf(z.z / r);
    float phi = atan2f(z.y, z.x);
    dr = powf(r, power - 1.0) * power * dr + 1.0;

    float zr = powf(r, power);
    theta = theta * power;
    phi = phi * power;

    // Back to euclidean
    z = make_float3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta)) * zr;
    z = z + pos;
  }

  return 0.5 * log(r) * r / dr;
}

__host__ __device__ float march(float3 origin, float3 direction, float time, distanceFunction func)
{
  // We can slowly march foward in the current direction upto the maximum number of ray iterations.
  // The distance esimation fractal is passed as an argument.
  int steps = 0;
  float total_dist = 0;
  for (; steps < MAX_RAY_ITER; steps++)
  {
    float3 p = origin + direction * total_dist;
    float distance = func(p, time);
    total_dist += distance;
    if (distance < MIN_DIST)
      break;
  }

  return 1.0 - (float)steps / (float)MAX_RAY_ITER;
}


__global__ void calculateBuffer(uint8_t *image_buffer, float power, float3 rayOrigin,  distanceFunction func)
{
  // TODO: dynamically change the distance estimation function
  //distanceFunction func = distEstMandelBulb;

  int row = blockIdx.y * blockDim.y + threadIdx.y; // WIDTH
  int col = blockIdx.x * blockDim.x + threadIdx.x; // HEIGHT
  int idx = IMG_CH * (row * WIDTH + col);
  if (col >= WIDTH || row >= HEIGHT)
    return;


  // Calculate the ray origin from the centre of the pixel
  float x0 = ((float)col / WIDTH) * 2.0f - 1.0f;
  float y0 = ((float)row / HEIGHT) * 2.0f - 1.0f;
  float3 center_dir = norm(make_float3(0, 0, 0) - rayOrigin);
  float3 xDir = norm(cross(center_dir, make_float3(0, 1, 0)));
  float3 yDir = norm(cross(center_dir, xDir));

  float3 pixelOrigin = rayOrigin + xDir * x0 + yDir * y0 + center_dir;
  float3 direction = pixelOrigin - rayOrigin;

  float colour = march(pixelOrigin, direction, power, func);

  // Convert the distance into a colour, for now just go with purple as it looks good.
  image_buffer[idx] = (uint8_t)255 * (0.5 * colour);
  image_buffer[idx + 1] = (uint8_t)0;
  image_buffer[idx + 2] = (uint8_t)255 * colour;
  image_buffer[idx + 3] = 255;
}

void runKernel(sf::Uint8 *image_buffer, float power, float3 rayOrigin, distanceFunction func)
{
  // Runs the CUDA kernel and copies the result back to memory.
  uint8_t *d_image_buffer;
  cudaAssertSuccess(cudaMalloc(&d_image_buffer, WIDTH * HEIGHT * IMG_CH));
  dim3 block_size(16, 16);
  dim3 grid_size(WIDTH / block_size.x, HEIGHT / block_size.y);
  calculateBuffer<<<grid_size, block_size>>>(d_image_buffer, power, rayOrigin, func);

  cudaAssertSuccess(cudaPeekAtLastError());
  cudaAssertSuccess(cudaDeviceSynchronize());
  cudaAssertSuccess(cudaMemcpy(image_buffer, d_image_buffer, IMG_CH * HEIGHT * WIDTH, cudaMemcpyDeviceToHost));
  cudaAssertSuccess(cudaFree(d_image_buffer));
}

__device__ distanceFunction p_bulbDev = distEstMandelBulb;
__device__ distanceFunction p_boxDev = distEstMandelBox;


int renderLoop()
{
  sf::RenderWindow window(sf::VideoMode(HEIGHT, WIDTH), "3D fractal viewer");

  sf::Texture text;
  text.create(HEIGHT, WIDTH);
  sf::Sprite imgSprite(text);

  int buf_size = WIDTH * HEIGHT * IMG_CH;
  sf::Uint8 *pixels = new sf::Uint8[buf_size];

  sf::Clock clock;
  clock.restart();

  bool changePower = true;
  bool isMandel = true;
  float3 rayOrigin = make_float3(0.f, 0.f, -2.f);

  float draw_time = 0;

  double hAngle = 0;
  double vAngle = 0;

  double viewR = 1.0;

  float renderTime = 0.0;

  // Setup distanace estimation pointers
  distanceFunction boxDistance;
  distanceFunction bulbDistance;
  distanceFunction * currDistFunc =  &boxDistance;
	cudaMemcpyFromSymbol(&boxDistance, p_boxDev, sizeof(distanceFunction));
	cudaMemcpyFromSymbol(&bulbDistance, p_bulbDev, sizeof(distanceFunction));

  //
  while (window.isOpen())
  {
    sf::Event event;
    while (window.pollEvent(event))
    {
      if (event.type == sf::Event::Closed)
        window.close();

      if (event.type == sf::Event::MouseWheelScrolled)
      {
        if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel)
        {
          float delta = event.mouseWheelScroll.delta * 0.1;
          viewR -= delta;
          if (viewR < 0)
          {
            viewR = 0;
          }
        }
      }
      else if (event.type == sf::Event::KeyPressed)
      {
        if (event.key.code == sf::Keyboard::P)
        {
          changePower = !changePower;
        }
        else if (event.key.code == sf::Keyboard::M)
        {
          isMandel = !isMandel;
        }
      }
    }

    // Use the draw time to control the amount of seconds to spend while
    draw_time = clock.restart().asSeconds();


    // Control the rotation
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
    {
      hAngle += 1.0 * draw_time;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
    {
      hAngle -= 1.0 * draw_time;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
    {
      vAngle += 1.0 * draw_time;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
    {
      vAngle -= 1.0 * draw_time;
    }


    // TODO: refactor this into a list as more fractal types are added.
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num1))
    {
      currDistFunc = &boxDistance;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num2))
    {
      currDistFunc = &bulbDistance;
    }

    rayOrigin.x = sin(hAngle) * cos(vAngle);
    rayOrigin.y = sin(vAngle) * sin(hAngle);
    rayOrigin.z = cos(hAngle);
    rayOrigin = rayOrigin * viewR;

    if (changePower)
    {
      // Slowly change the power
      renderTime += draw_time;
    }

    // Currently transfering from the GPU back to the CPU,
    window.clear();
    runKernel(pixels, renderTime, rayOrigin, *currDistFunc);
    text.update(pixels);
    window.draw(imgSprite);
    window.display();

    std::cout << vAngle << " " << hAngle << " " << std::endl;
    std::cout << rayOrigin.x << " " << rayOrigin.y << " " << rayOrigin.z << std::endl;
    float fps = 1.f / draw_time;
    std::cout << "fps: " << fps << '\n';
  }

  return 0;
}

int main(int argc, char **argv)
{
  return renderLoop();
}