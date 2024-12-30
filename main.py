import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans2
from PIL import Image

custom_palette = [
    (46, 46, 67),
    (74, 75, 91),
    (112, 123, 137),
    (169, 188, 191),
    (230, 238, 237),
    (252, 251, 243),
    (252, 235, 168),
    (245, 196, 124),
    (227, 151, 100),
    (192, 104, 82),
    (157, 67, 67),
    (129, 54, 69),
    (84, 34, 64),
    (42, 21, 45),
    (79, 45, 77),
    (91, 58, 86),
    (121, 78, 109),
    (62, 76, 126),
    (73, 95, 148),
    (90, 120, 178),
    (115, 150, 213),
    (127, 187, 220),
    (170, 238, 234),
    (213, 248, 147),
    (150, 220, 127),
    (110, 192, 119),
    (78, 147, 99),
    (60, 108, 84),
    (44, 80, 73),
    (52, 64, 79),
    (64, 89, 103),
    (92, 137, 149),
]


def palettize_image_scipy(image, palette):
    pixels = image.reshape((-1, 3)).astype(np.float64)
    palette = np.array(palette, dtype=np.float64)

    centroids, labels = kmeans2(pixels, palette, iter=20, minit='points')

    quantized_colors = centroids[labels]
    quantized_image = quantized_colors.reshape(image.shape).astype(np.uint8)

    return quantized_image


def dither_image(image):
    pil_image = Image.fromarray(image.astype("uint8"))
    dithered_image = pil_image.convert(
        "P", dither=Image.FLOYDSTEINBERG, palette=Image.ADAPTIVE, colors=len(custom_palette)
    )
    return np.array(dithered_image.convert("RGB"))


def generate_pixel_art(image_path, pixelation_factor=5, edge_strength=0.3, target_size=1200):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if max(img.shape) > target_size:
        scale_factor = target_size / max(img.shape)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    small_img = cv2.resize(
        img,
        (img.shape[1] // pixelation_factor, img.shape[0] // pixelation_factor),
        interpolation=cv2.INTER_NEAREST,
    )
    pixelated_img = cv2.resize(
        small_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    palettized_img = palettize_image_scipy(pixelated_img, custom_palette)

    if edge_strength > 0:
        gray = cv2.cvtColor(palettized_img, cv2.COLOR_RGB2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
        abs_grad_x = cv2.convertScaleAbs(sobel_x)
        abs_grad_y = cv2.convertScaleAbs(sobel_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        _, edges = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)

        edge_image = np.zeros_like(palettized_img)

        for y in range(1, palettized_img.shape[0] - 1):
            for x in range(1, palettized_img.shape[1] - 1):
                if edges[y, x] > 0:
                    neighbors = palettized_img[y - 1: y + 2, x - 1: x + 2]
                    avg_color = np.mean(neighbors, axis=(0, 1))
                    outline_color = [max(0, c - 30) for c in avg_color]
                    edge_image[y, x] = outline_color

        palettized_img[edge_image > 0] = edge_image[edge_image > 0]

    kernel = np.ones((3, 3), np.uint8)
    palettized_img = cv2.morphologyEx(palettized_img, cv2.MORPH_CLOSE, kernel)

    return palettized_img


input_image_path = 'image.jpg'
output_image = generate_pixel_art(input_image_path, pixelation_factor=5, edge_strength=0.3, target_size=1200)

cv2.imshow('Pixel Art', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output.png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))