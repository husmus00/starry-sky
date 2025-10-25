from dataclasses import dataclass
from datetime import datetime
from PIL import Image
import pygame
import pygame_gui
import numpy as np
import argparse


# --- Display Config ---

WIDTH = 400
HEIGHT = 400
SCALE = 1.5

CONTROL_PANEL_WIDTH = 300
SKY_WIDTH = WIDTH * SCALE
SKY_HEIGHT = HEIGHT * SCALE

SCREEN_WIDTH = CONTROL_PANEL_WIDTH + SKY_WIDTH
SCREEN_HEIGHT = SKY_HEIGHT

# --- Simulation Config ---

@dataclass
class Config:
    brightness_modifier: float = 0.35
    max_brightness: float = 1.0
    min_brightness: float = 0.3
    spread: int = 15  # Higher values result in less spread
    delay: int = 75  # milliseconds
    gif_duration: int = 3000  # milliseconds, resulting gif will be double this duration due to ping-pong effect
    seed: int = 130388


# --- Star Patterns ---

# Sum should be 1.0
PROBABILITIES = [0.99201, 0.003, 0.003, 0.001, 0.0006, 0.00036, 0.00001, 0.00002]
STAR_VARIANTS = [0, 1, 2, 3, 4, 5, 6, 7]

DIM_STAR = [128]

SMALL_STAR = [255]

MEDIUM_STAR = [
    [0,128,0],
    [128,255,128],
    [0,128,0]
]

MEDIUM_STAR_2 = [
    [0,0,0,128,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,255,0,0,0],
    [128,0,255,255,255,0,128],
    [0,0,0,255,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,128,0,0,0]
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

LARGE_STAR_2 = [
    [0,0,0,0,64,0,0,0,0],
    [0,0,0,0,128,0,0,0,0],
    [0,0,0,192,192,192,0,0,0],
    [0,0,192,192,255,192,192,0,0],
    [64,128,192,255,255,255,192,128,64],
    [0,0,192,192,255,192,192,0,0],
    [0,0,0,192,192,192,0,0,0],
    [0,0,0,0,128,0,0,0,0],
    [0,0,0,0,64,0,0,0,0]
]

EXTRA_LARGE_STAR = [
    [0,0,0,0,0,32,0,0,0,0,0],
    [0,0,0,0,0,64,0,0,0,0,0],
    [0,0,0,0,64,128,64,0,0,0,0],
    [0,0,0,64,128,192,128,64,0,0,0],
    [0,0,64,128,192,255,192,128,64,0,0],
    [32,64,128,192,255,255,255,192,128,64,32],
    [0,0,64,128,192,255,192,128,64,0,0],
    [0,0,0,64,128,192,128,64,0,0,0],
    [0,0,0,0,64,128,64,0,0,0,0],
    [0,0,0,0,0,64,0,0,0,0,0],
    [0,0,0,0,0,32,0,0,0,0,0]
]


STAR_PROPERTIES = {
    1: {'size': 1, 'pattern': DIM_STAR},
    2: {'size': 1, 'pattern': SMALL_STAR},
    3: {'size': 3, 'pattern': MEDIUM_STAR},
    4: {'size': 7, 'pattern': MEDIUM_STAR_2},
    5: {'size': 7, 'pattern': LARGE_STAR},
    6: {'size': 9, 'pattern': LARGE_STAR_2},
    7: {'size': 11, 'pattern': EXTRA_LARGE_STAR},
}


# Global instances - will be initialised in main()
rng = None
config = None


class Star:
   def __init__(self, variant, rng):
    self.variant = variant
    self.brightness = rng.uniform(config.min_brightness, config.max_brightness) if variant != 0 else 0.0


def main():
    global rng
    global config

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate starry night gifs and interactive visualisations')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Run interactive window (without generating gif)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for star generation (0-999999)')
    args = parser.parse_args()

    # Initialise config
    config = Config()

    # Set seed from command line or use configured default
    if args.seed is not None:
        # Validate seed bounds
        if not 0 <= args.seed <= 999999:
            print(f"Error: Seed must be between 0 and 999999")
            return
        config.seed = args.seed
    elif config.seed is None:
        # Generate random seed
        temp_rng = np.random.default_rng()
        config.seed = temp_rng.integers(0, 1000000)

    print(f"Using random seed: {config.seed}")
    rng = np.random.default_rng(config.seed)

    # Generate star array
    star_array = generate_star_array()

    if args.interactive:
        # Run interactive mode only
        run_interactive(star_array)
    else:
        # Generate gif (default behaviour)
        generate_gif(star_array)


def generate_gif(star_array):
    """
    Generate a gif of the starry sky and save it with a timestamp
    """
    print(f"Generating {config.gif_duration * 2} ms gif...")

    # Calculate number of frames for desired duration
    duration_ms = config.gif_duration
    num_frames = duration_ms // config.delay

    # Make a deep copy of the star array so gif and display don't interfere
    # star_array = copy.deepcopy(original_star_array)

    # Capture frames
    frames = []
    for i in range(num_frames):
        # Generate sky array (not the pygame surface)
        sky_array = generate_sky_array(star_array)

        # Transpose to match pygame's coordinate system
        # pygame uses (width, height) and transposes, PIL uses (height, width)
        sky_array_transposed = sky_array.T

        # Convert to RGB format for PIL
        rgb_array = np.stack((sky_array_transposed,) * 3, axis=-1)

        # Create PIL Image
        img = Image.fromarray(rgb_array.astype(np.uint8))
        frames.append(img)

        # Update brightness for next frame using the gif's RNG
        update_brightness(star_array)

    # Create ping-pong effect: forward then backward
    # Append frames in reverse (excluding first and last to avoid duplicates)
    reversed_frames = frames[-2:0:-1]
    all_frames = frames + reversed_frames

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"starry-sky-{timestamp}-seed-{config.seed}.gif"

    # Save as animated gif
    all_frames[0].save(
        filename,
        save_all=True,
        append_images=all_frames[1:],
        duration=config.delay,
        loop=0
    )

    print(f"GIF saved as: {filename}")


def run_interactive(star_array):
    """
    Run the interactive pygame window with sliders
    """
    screen, ui_manager, sliders, clock = init_pygame()

    running = True
    last_update_time = 0

    while running:
        time_delta = clock.tick(60)/1000.0
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                handle_slider_event(event, sliders)

            ui_manager.process_events(event)

        ui_manager.update(time_delta)

        # Update brightness at configured speed (independent of UI framerate)
        if current_time - last_update_time >= config.delay:
            update_brightness(star_array)
            last_update_time = current_time

        sky = generate_sky(star_array)
        draw_sky(screen, sky, ui_manager)

    pygame.quit()

def init_pygame():
    pygame.init()

    # Set up display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Starry Night')

    ui_manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    sliders = create_sliders(ui_manager)

    return screen, ui_manager, sliders, clock


def create_sliders(ui_manager):
    # Left control panel layout
    x_margin = 15
    y_start = 20
    y_spacing = 70
    slider_width = 220
    label_width = 270
    value_label_width = 60

    sliders = {}
    y_offset = y_start

    # Title
    pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin, y_offset), (label_width, 30)),
        text='Controls',
        manager=ui_manager
    )
    y_offset += 50

    # Brightness Modifier slider
    pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin, y_offset), (label_width, 20)),
        text='Brightness Modifier:',
        manager=ui_manager
    )
    sliders['brightness_modifier'] = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((x_margin, y_offset + 25), (slider_width, 20)),
        start_value=config.brightness_modifier,
        value_range=(0.0, 1.0),
        manager=ui_manager
    )
    sliders['brightness_modifier_label'] = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin + slider_width + 10, y_offset + 25), (value_label_width, 20)),
        text=f'{config.brightness_modifier:.2f}',
        manager=ui_manager
    )

    # Max Brightness slider
    y_offset += y_spacing
    pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin, y_offset), (label_width, 20)),
        text='Max Brightness:',
        manager=ui_manager
    )
    sliders['max_brightness'] = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((x_margin, y_offset + 25), (slider_width, 20)),
        start_value=config.max_brightness,
        value_range=(0.1, 1.0),
        manager=ui_manager
    )
    sliders['max_brightness_label'] = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin + slider_width + 10, y_offset + 25), (value_label_width, 20)),
        text=f'{config.max_brightness:.2f}',
        manager=ui_manager
    )

    # Min Brightness slider
    y_offset += y_spacing
    pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin, y_offset), (label_width, 20)),
        text='Min Brightness:',
        manager=ui_manager
    )
    sliders['min_brightness'] = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((x_margin, y_offset + 25), (slider_width, 20)),
        start_value=config.min_brightness,
        value_range=(0.0, 1.0),
        manager=ui_manager
    )
    sliders['min_brightness_label'] = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin + slider_width + 10, y_offset + 25), (value_label_width, 20)),
        text=f'{config.min_brightness:.2f}',
        manager=ui_manager
    )

    # Spread slider
    y_offset += y_spacing
    pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin, y_offset), (label_width, 20)),
        text='Spread:',
        manager=ui_manager
    )
    sliders['spread'] = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((x_margin, y_offset + 25), (slider_width, 20)),
        start_value=config.spread,
        value_range=(1, 50),
        manager=ui_manager
    )
    sliders['spread_label'] = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin + slider_width + 10, y_offset + 25), (value_label_width, 20)),
        text=f'{config.spread}',
        manager=ui_manager
    )

    # Speed slider
    y_offset += y_spacing
    pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin, y_offset), (label_width, 20)),
        text='Speed (ms):',
        manager=ui_manager
    )
    sliders['speed'] = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((x_margin, y_offset + 25), (slider_width, 20)),
        start_value=config.delay,
        value_range=(50, 2000),
        manager=ui_manager
    )
    sliders['speed_label'] = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((x_margin + slider_width + 10, y_offset + 25), (value_label_width, 20)),
        text=f'{config.delay}',
        manager=ui_manager
    )

    return sliders


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
            if star:
                insert_star(sky_array, star, (y, x))

    return sky_array


def convert_to_surface(sky_array):
    # Convert to a 3D array for Pygame (R, G, B)
    greyscale_surface = np.stack((sky_array,) * 3, axis=-1)

    surface = pygame.surfarray.make_surface(greyscale_surface)

    scaled_surface = pygame.transform.scale(surface, (SKY_WIDTH, SKY_HEIGHT))

    return scaled_surface


def generate_star_array():
    # Define the pixel values and their respective probabilities
    variants = np.array(STAR_VARIANTS, dtype=np.uint8)
    probabilities = np.array(PROBABILITIES)

    # Create an array of stars with random variants based on the specified probabilities
    star_array = np.empty((HEIGHT, WIDTH), dtype=object)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            variant = rng.choice(variants, p=probabilities)
            star_array[y, x] = Star(variant, rng) if variant != 0 else None

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
        # sky[y:y + star_height, x:x + star_width] = star_pixel_array
        # Insert only non-zero values from the small array
        sky[y:y + star_height, x:x + star_width] = np.where(star_pixel_array != 0, star_pixel_array, sky[y:y + star_height, x:x + star_width])


def handle_slider_event(event, sliders):
    """Update config values based on slider movement"""
    if event.ui_element == sliders['brightness_modifier']:
        config.brightness_modifier = event.value
        sliders['brightness_modifier_label'].set_text(f'{event.value:.2f}')
    elif event.ui_element == sliders['max_brightness']:
        config.max_brightness = event.value
        sliders['max_brightness_label'].set_text(f'{event.value:.2f}')
    elif event.ui_element == sliders['min_brightness']:
        config.min_brightness = event.value
        sliders['min_brightness_label'].set_text(f'{event.value:.2f}')
    elif event.ui_element == sliders['spread']:
        config.spread = int(event.value)
        sliders['spread_label'].set_text(f'{int(event.value)}')
    elif event.ui_element == sliders['speed']:
        config.delay = int(event.value)
        sliders['speed_label'].set_text(f'{int(event.value)}')


def update_brightness(star_array):
    # Calculate std_dev once
    std_dev = (2 * config.brightness_modifier) / config.spread

    # Generate ALL random changes at once (vectorized!)
    changes = rng.normal(0, std_dev, size=(HEIGHT, WIDTH))

    # Update all brightnesses using vectorized operations
    for y in range(HEIGHT):
        for x in range(WIDTH):
            star = star_array[y, x]
            if star:
                star.brightness = np.clip(
                    star.brightness + changes[y, x],
                    config.min_brightness,
                    config.max_brightness
                )


def draw_sky(screen, sky, ui_manager):
    # Fill background
    screen.fill((0, 0, 0))

    # Draw a divider line between control panel and sky
    pygame.draw.line(screen, (80, 80, 80), (CONTROL_PANEL_WIDTH, 0), (CONTROL_PANEL_WIDTH, SCREEN_HEIGHT), 2)

    # Draw the starry sky in the right pane
    screen.blit(sky, (CONTROL_PANEL_WIDTH, 0))

    # Draw UI elements (controls in left pane)
    ui_manager.draw_ui(screen)

    pygame.display.flip()
    

if __name__=='__main__':
    main()
