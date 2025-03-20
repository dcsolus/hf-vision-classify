import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path

def load_image(file_path: Path) -> np.array:
    """
    Load an image from a given file path and convert it to a NumPy array.
    
    Args:
        file_path (Path): Path to the image file.
    
    Returns:
        np.array: The image as a NumPy array.
    """
    pil_image = Image.open(file_path)
    return np.array(pil_image)

def add_irregular_patch(noisy_image: np.array, patch_pixels: int, noise_value: int) -> np.array:
    """
    Adds an irregular, polygonal noise patch to the image.
    
    Args:
        noisy_image (np.array): The image array to modify.
        patch_pixels (int): Approximate area of the noise patch in pixels.
        noise_value (int): Noise value (255 for salt, 0 for pepper).
    
    Returns:
        np.array: The modified image with the noise patch added.
    """
    img_pil = Image.fromarray(noisy_image)
    draw = ImageDraw.Draw(img_pil)
    img_width, img_height = img_pil.size

    # Estimate an average radius from the patch area (area ≈ πr²)
    r = int(np.sqrt(patch_pixels / np.pi))
    r = max(r, 5)  # Ensure a minimum patch size

    # Choose a random center where the patch can fit
    center_x = np.random.randint(r, img_width - r)
    center_y = np.random.randint(r, img_height - r)

    # Random number of vertices for the irregular polygon
    num_points = np.random.randint(5, 10)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    angles += np.random.uniform(0, 2 * np.pi / num_points, size=num_points)
    # Generate random radii for each vertex around the average radius
    radii = np.random.uniform(0.5 * r, 1.5 * r, size=num_points)
    
    # Create the polygon points
    points = [
        (int(center_x + radius * np.cos(angle)), int(center_y + radius * np.sin(angle)))
        for angle, radius in zip(angles, radii)
    ]
    
    # Set the fill color based on image mode
    fill_color = (noise_value, noise_value, noise_value) if img_pil.mode == 'RGB' else noise_value

    draw.polygon(points, fill=fill_color)
    return np.array(img_pil)

def random_patch_areas(total: int, num_patches: int) -> np.array:
    """
    Split the total noise area into a random distribution of patch areas that sum to total.
    
    Args:
        total (int): Total number of noisy pixels for this noise type.
        num_patches (int): Number of patches.
    
    Returns:
        np.array: Array of patch areas.
    """
    areas = np.random.rand(num_patches)
    areas = areas / areas.sum()  # Normalize so areas sum to 1
    areas = (total * areas).astype(int)
    # Adjust in case of rounding issues
    diff = total - areas.sum()
    areas[0] += diff
    return areas

def apply_random_irregular_noise(image: np.array, amount: float, salt_vs_pepper: float) -> np.array:
    """
    Applies random irregular noise patches (both salt and pepper) to an image.
    
    Args:
        image (np.array): The original image.
        amount (float): Fraction of total image pixels to be noised.
        salt_vs_pepper (float): Fraction of noise that is salt (white).
    
    Returns:
        np.array: The noisy image.
    """
    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_pixels = int(amount * total_pixels)
    
    # Calculate noise areas for salt and pepper
    num_salt = int(salt_vs_pepper * num_pixels)
    num_pepper = num_pixels - num_salt

    # Randomly determine number of patches (e.g., 1 to 3 patches)
    num_salt_patches = np.random.randint(1, 10)
    num_pepper_patches = np.random.randint(1, 10)

    # Split the noise area into patch areas
    salt_patch_areas = random_patch_areas(num_salt, num_salt_patches)
    pepper_patch_areas = random_patch_areas(num_pepper, num_pepper_patches)

    # Add salt noise patches
    for area in salt_patch_areas:
        noisy = add_irregular_patch(noisy, area, 255)

    # Add pepper noise patches
    for area in pepper_patch_areas:
        noisy = add_irregular_patch(noisy, area, 0)
    
    return noisy

def plot_images(original: np.array, noisy: np.array) -> None:
    """
    Plots the original and noisy images side by side.
    
    Args:
        original (np.array): The original image.
        noisy (np.array): The noisy image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(noisy)
    axes[1].set_title("Image with Random Irregular Noise Patches")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

def main(file_path:Path, amount:float, salt_vs_pepper:float):

    # Load the image
    image = load_image(file_path)
    
    # Apply the noise
    noisy_image = apply_random_irregular_noise(image, amount, salt_vs_pepper)
    pil_image = Image.fromarray(noisy_image)
    
    # save noisy image
    output_file_path = Path('noisy_frames') / f"{file_path.stem}_noisy{file_path.suffix}"
    output_file_path.parent.mkdir(exist_ok=True)
    pil_image.save(output_file_path, format='JPEG')
    
    # Display the original and noisy images
    plot_images(image, noisy_image)

if __name__ == "__main__":
    file_path = Path('frames') / 'frame_0.jpg'
    amount = 0.01         # Fraction of image pixels to be noised
    salt_vs_pepper = 0.6  # Fraction of noise that is salt (white)

    main(file_path, amount, salt_vs_pepper)
