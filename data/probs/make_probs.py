import os

def frange(start, stop, step):
  i = start
  while i < stop:
    yield i
    i += step

for prob in frange(0.01, 1.0, 0.01):
    os.system("convert -fill black -background white -bordercolor white -border 4 -font futura-normal -pointsize 18 label:\"%s\" \"%s.png\""%(prob, prob))
