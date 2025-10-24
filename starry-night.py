import pygame
import numpy as np
import time
from dataclasses import dataclass


# --- Config ---

WIDTH = 150
HEIGHT = 150
SCALE = 3

SCREEN_WIDTH = WIDTH * SCALE
SCREEN_HEIGHT = HEIGHT * SCALE

STAR_VARIANTS = [0, 1, 2, 3, 4]  # None, Small, Medium, Large
PROBABILITIES = [0.9885, 0.004, 0.004, 0.003, 0.0005]

BRIGHTNESS_MODIFIER = 0.35
MAX_BRIGHTNESS = 1.0
MIN_BRIGHTNESS = 0.3
SPREAD = 12  # Higher values result in less spread
SPEED = 800 # milliseconds

# --- Star Patterns ---

DIM_STAR = [128]

SMALL_STAR = [255]

MEDIUM_STAR = [
    [0,128,0],
    [128,255,128],
    [0,128,0]
]

LARGE_STAR = [
    [0,0,0,64,0,0,0],
    [0,0,0,128,0,0,0],
    [0,0,64,255,64,0,0],
    [64,128,255,255,255,128,64],
    [0,0,64,255,64,0,0],
    [0,0,0,128,0,0,0],
    [0,0,0,64,0,0,0]
]

STAR_PROPERTIES = {
    1: {'size': 1, 'pattern': DIM_STAR},
    2: {'size': 1, 'pattern': SMALL_STAR},
    3: {'size': 3, 'pattern': MEDIUM_STAR},
    4: {'size': 7, 'pattern': LARGE_STAR}
}


class Star:
   def __init__(self, variant):
    self.variant = variant
    self.brightness = np.random.uniform(MIN_BRIGHTNESS, MAX_BRIGHTNESS) if variant != 0 else 0.0
    

def main():
    screen = init_pygame()
    star_array = generate_star_array()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # sky = update_sky(sky)
        sky = generate_sky(star_array)
        draw_sky(screen, sky)
        update_brightness(star_array)
        pygame.time.delay(SPEED)


def init_pygame():
    # Initialize Pygame
    pygame.init()

    # Set up display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    return screen


def generate_sky(star_array):
    sky_array = generate_sky_array(star_array)
    sky_surface = convert_to_surface(sky_array)
    return sky_surface


def generate_sky_array(star_array):
    # Create an empty sky array
    sky_array = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    # Insert stars into the sky array
    for y in range(HEIGHT):
        for x in range(WIDTH):
            star = star_array[y, x]
            if star.variant != 0:
                insert_star(sky_array, star, (y, x))

    return sky_array


def convert_to_surface(sky_array):    # Convert to a 3D array for Pygame (R, G, B)
    greyscale_surface = np.stack((sky_array,) * 3, axis=-1)

    # Create a Pygame surface from the NumPy array
    surface = pygame.surfarray.make_surface(greyscale_surface)

    # Scale the surface
    scaled_surface = pygame.transform.scale(surface, (SCREEN_WIDTH, SCREEN_HEIGHT))

    return scaled_surface


def generate_star_array():
    # Define the pixel values and their respective probabilities
    variants = np.array(STAR_VARIANTS, dtype=np.uint8)
    probabilities = np.array(PROBABILITIES)

    # Create an array of stars with random variants based on the specified probabilities
    star_array = np.empty((HEIGHT, WIDTH), dtype=object)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            variant = np.random.choice(variants, p=probabilities)
            star_array[y, x] = Star(variant)
    
    return star_array



def insert_star(sky, star, position):
    variant = star.variant
    star_pixel_array = STAR_PROPERTIES[variant]['pattern']
    star_height = STAR_PROPERTIES[variant]['size']
    star_width = STAR_PROPERTIES[variant]['size']
    y, x = position

    # Set brightness
    star_pixel_array = np.array(star_pixel_array, dtype=np.float32) * star.brightness
    # star_pixel_array = np.clip(star_pixel_array, 0, 255).astype(np.uint8)

    # Ensure the small array fits within the large array bounds
    if (y + star_height <= sky.shape[0]) and (x + star_width <= sky.shape[1]):
        sky[y:y + star_height, x:x + star_width] = star_pixel_array


def update_brightness(star_array):
    for y in range(HEIGHT):
        for x in range(WIDTH):
            star = star_array[y, x]
            if star.variant != 0:
                # change = np.random.uniform(-BRIGHTNESS_MODIFIER, BRIGHTNESS_MODIFIER)
                std_dev =(BRIGHTNESS_MODIFIER - (-BRIGHTNESS_MODIFIER)) / SPREAD  # This will adjust the spread
                change = np.random.normal(0, std_dev)
                star.brightness = np.clip(star.brightness + change, MIN_BRIGHTNESS, MAX_BRIGHTNESS)


def draw_sky(screen, sky):
    #surface = pygame.surfarray.make_surface(sky)
    screen.blit(sky, (0, 0))
    pygame.display.flip()

if __name__=='__main__':
    main()

# Quit Pygame
pygame.quit()
