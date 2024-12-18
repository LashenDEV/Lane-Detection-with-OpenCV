import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# reading in an image
image = mpimg.imread('lane_1.jpg')

height, width, _ = image.shape

print(height, width)

region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


plt.imshow(image)
plt.show()

cropped_image = region_of_interest(
    image,
    np.array([region_of_interest_vertices], np.int32),
)

plt.figure()


print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(cropped_image)
plt.show()


