from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import torch
import io
import os
import subprocess
import pickle

# Convolutional neural network
class LeNet5(torch.nn.Module):
  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = torch.nn.Conv2d(
       in_channels = 1, out_channels=6, kernel_size=5, padding=2)
    self.act1 = torch.nn.ReLU()
    self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2 = torch.nn.Conv2d(
       in_channels = 6, out_channels=16, kernel_size=5, padding=0)
    self.act2 = torch.nn.ReLU()
    self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = torch.nn.Linear(5*5*16, 120)
    self.act3 = torch.nn.Sigmoid()

    self.fc2 = torch.nn.Linear(120, 84)
    self.act4 = torch.nn.Sigmoid()

    self.fc3 = torch.nn.Linear(84, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.act1(x)
    x = self.pool1(x)

    x = self.conv2(x)
    x = self.act2(x)
    x = self.pool2(x)

    x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

    x = self.fc1(x)
    x = self.act3(x)
    x = self.fc2(x)
    x = self.act4(x)
    x = self.fc3(x)

    return x


# Draw picture
def print(event):
  color = 'yellow'
  x1, y1 = (event.x-7), (event.y-7)
  x2, y2 = (event.x+7), (event.y+7)
  c.create_oval(x1, y1, x2, y2, fill=color, outline=color)

  c.create_oval(x1, y1+1, x2, y2+1, fill=color, outline=color)
  c.create_oval(x1+1, y1, x2+1, y2, fill=color, outline=color)
  c.create_oval(x1-1, y1, x2-1, y2, fill=color, outline=color)
  c.create_oval(x1, y1-1, x2, y2-1, fill=color, outline=color)

  draw.ellipse((x1, y1, x2, y2), fill='yellow')

  draw.ellipse((x1+1, y1, x2+1, y2), fill='yellow')
  draw.ellipse((x1, y1+1, x2, y2+1), fill='yellow')
  draw.ellipse((x1-1, y1, x2-1, y2), fill='yellow')
  draw.ellipse((x1, y1-1, x2, y2-1), fill='yellow')
  draw.ellipse((x1 + 1, y1, x2 + 1, y2), fill='yellow')
  draw.ellipse((x1, y1 + 1, x2, y2 + 1), fill='yellow')
  draw.ellipse((x1 - 1, y1, x2 - 1, y2), fill='yellow')
  draw.ellipse((x1, y1 - 1, x2, y2 - 1), fill='yellow')
  draw.ellipse((x1 + 1, y1, x2 + 1, y2), fill='yellow')
  draw.ellipse((x1, y1 + 1, x2, y2 + 1), fill='yellow')
  draw.ellipse((x1 - 1, y1, x2 - 1, y2), fill='yellow')
  draw.ellipse((x1, y1 - 1, x2, y2 - 1), fill='yellow')

# Filter
def filtr(img):
  """Предполагалось, что этот фильтр будет избаляться от пикселей, который имели малые значения. """
  draw = ImageDraw.Draw(img)
  pix = img.load()
  # for i in range(28):
  #   for j in range(28):
  #     a = pix[i, j][0]
  #     b = pix[i, j][1]
  #     c = pix[i, j][2]
  #     S = (a + b + c) // 3
  #
  #     if (S > 30):
  #       S = 150
  #       draw.point((i, j), (S, S, S))
  #     else:
  #       draw.point((i, j), (0, 0, 0))
  img = img.convert('L')
  f = np.asarray(img)
  return f

# Get number
def click_button():
  global  image1, draw
  filename = "my_drawing.jpg"
  image1 = image1.resize((28, 28), Image.ANTIALIAS)
  image1.save(filename)

  with open('model.pickle', 'rb') as f:
    neural_net = pickle.load(f)

  pic = Image.open('my_drawing.jpg')
  img = filtr(pic)
  data = torch.tensor(img)
  data = data.unsqueeze(0).float()
  data = data.unsqueeze(0).float()
  pred = neural_net.forward(data)
  text.insert(1.0, pred.argmax(dim=1).numpy()[0])

# clear canvas and picture
def click_update():
  global image1, draw
  c.delete('all')
  del image1
  image1 = Image.new("RGB", (canvas_width, canvas_height))
  draw = ImageDraw.Draw(image1)
  text.delete(1.0, END)






lenet5_net = LeNet5()

# Create Gui
master = Tk()
btn1 = Button(master , text = 'Получить число', state = ACTIVE,command = click_button)
btn1.pack()
btn2 = Button(master , text = 'Заново', state = ACTIVE,command = click_update)
btn2.pack()
text = Text(width=1, height=1)
text.pack()

master.title('painting in python')
message=Label(master, text='Press and Drag to draw')
message.pack(side=BOTTOM)


# Create canvas
canvas_width = 200
canvas_height = 200
c = Canvas(master, width=canvas_width, height=canvas_height, bg='black')
c.pack(expand=YES, fill=BOTH)
c.bind('<B1-Motion>', print)

# Create picture
image1 = Image.new("RGB", (canvas_width, canvas_height))
draw = ImageDraw.Draw(image1)

# Start Application
master.mainloop()


