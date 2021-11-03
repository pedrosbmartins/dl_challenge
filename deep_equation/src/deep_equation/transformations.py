import torchvision.transforms as T
import PIL.ImageOps

def to_negative(img):
  img = PIL.ImageOps.invert(img)
  return img

class ToDarkBackground(object):
  def __init__(self):
    pass
  
  def __call__(self, img):
    if T.ToTensor()(img).mean() >= 0.5:
      return to_negative(img)
    else:
      return img

def image_transform():
  return T.Compose([
    T.Grayscale(),
    T.Resize(size=(28,28)),
    ToDarkBackground(),
    T.ToTensor()])