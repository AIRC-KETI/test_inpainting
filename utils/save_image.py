from torchvision.utils import save_image

def save_the_first_images(img, text):
    first = img[0]
    save_image(first, text)
