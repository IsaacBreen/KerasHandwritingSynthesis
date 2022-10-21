import tensorflow as tf
import os
import pickle
import random
import xml.etree.ElementTree as ET
import html

import numpy as np
import svgwrite
from IPython.display import SVG, display

from collections import defaultdict

# ----------- BEGIN ADDED STUFF -----------
alphabet = [
'\n', '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
'%', '&', '+', '/',
'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
'?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
'y', 'z'
]

alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))

def encode_ascii(ascii_string):
  # encodes ascii string to array of ints
  return np.array(list(map(lambda x: alpha_to_num[x], ascii_string)) + [0])

# ------------ END ADDED STUFF ------------

def get_bounds(data, factor):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

# old version, where each path is entire stroke (smaller svg size, but
# have to keep same color)


def draw_strokes(data, factor=10, svg_filename='sample.svg'):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)

  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

  lift_pen = 1

  abs_x = 25 - min_x
  abs_y = 25 - min_y
  p = f"M{abs_x},{abs_y} "

  command = "m"

  for i in range(len(data)):
      if (lift_pen == 1):
          command = "m"
      elif (command != "l"):
          command = "l"
      else:
          command = ""
      x = float(data[i, 0]) / factor
      y = float(data[i, 1]) / factor
      lift_pen = data[i, 2]
      p += command + str(x) + "," + str(y) + " "

  the_color = "black"
  stroke_width = 1

  dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))

  dwg.save()
  display(SVG(dwg.tostring()))


def draw_strokes_eos_weighted(
        stroke,
        param,
        factor=10,
        svg_filename='sample_eos.svg'):
    c_data_eos = np.zeros((len(stroke), 3))
    for i in range(len(param)):
        # make color gray scale, darker = more likely to eos
        c_data_eos[i, :] = (1 - param[i][6][0]) * 225
    draw_strokes_custom_color(
        stroke,
        factor=factor,
        svg_filename=svg_filename,
        color_data=c_data_eos,
        stroke_width=3)


def draw_strokes_random_color(
        stroke,
        factor=10,
        svg_filename='sample_random_color.svg',
        per_stroke_mode=True):
  c_data = np.array(np.random.rand(len(stroke), 3) * 240, dtype=np.uint8)
  if per_stroke_mode:
    switch_color = False
    for i in range(len(stroke)):
      if switch_color == False and i > 0:
          c_data[i] = c_data[i - 1]
      switch_color = stroke[i, 2] >= 1
  draw_strokes_custom_color(
      stroke,
      factor=factor,
      svg_filename=svg_filename,
      color_data=c_data,
      stroke_width=2)


def draw_strokes_custom_color(
        data,
        factor=10,
        svg_filename='test.svg',
        color_data=None,
        stroke_width=1):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)

  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

  lift_pen = 1
  abs_x = 25 - min_x
  abs_y = 25 - min_y

  for i in range(len(data)):

    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor

    prev_x = abs_x
    prev_y = abs_y

    abs_x += x
    abs_y += y

    if (lift_pen == 1):
      p = f"M {str(abs_x)},{str(abs_y)} "
    else:
      p = f"M +{str(prev_x)},{str(prev_y)} L {str(abs_x)},{str(abs_y)} "

    lift_pen = data[i, 2]

    the_color = "black"

    if (color_data is not None):
      the_color = f"rgb({int(color_data[i, 0])},{int(color_data[i, 1])},{int(color_data[i, 2])})"

    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill(the_color))
  dwg.save()
  display(SVG(dwg.tostring()))


def draw_strokes_pdf(data, param, factor=10, svg_filename='sample_pdf.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    abs_x = 25 - min_x
    abs_y = 25 - min_y

    num_mixture = len(param[0][0])

    for i in range(len(data)):

        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor

        for k in range(num_mixture):
            pi = param[i][0][k]
            if pi > 0.01:  # optimisation, ignore pi's less than 1% chance
                mu1 = param[i][1][k]
                mu2 = param[i][2][k]
                s1 = param[i][3][k]
                s2 = param[i][4][k]
                sigma = np.sqrt(s1 * s2)
                dwg.add(dwg.circle(center=(abs_x + mu1 * factor,
                                           abs_y + mu2 * factor),
                                   r=int(sigma * factor)).fill('red',
                                                               opacity=pi / (sigma * sigma * factor)))

        prev_x = abs_x
        prev_y = abs_y

        abs_x += x
        abs_y += y

    dwg.save()
    display(SVG(dwg.tostring()))


class DataLoader(tf.keras.utils.Sequence):
    def __init__(
            self,
            data_dir="./data",
            batch_size=50,
            seq_length=300,
            maximise_seq_len=False,
            scale_factor=10,
            limit=500):
      self.data_dir = data_dir
      self.batch_size = batch_size
      self.seq_length = seq_length
      self.maximise_seq_len = maximise_seq_len
      self.scale_factor = scale_factor  # divide data by this factor
      self.limit = limit  # removes large noisy gaps in the data

      data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
      transcriptions_file = os.path.join(self.data_dir, "transcriptions.cpkl")
      raw_data_dir = f"{self.data_dir}/lineStrokes"

      if not (os.path.exists(data_file)):
          print("creating training data pkl file from raw source")
          self.preprocess(raw_data_dir, data_file, transcriptions_file)

      self.load_preprocessed(data_file, transcriptions_file)
      self.reset_batch_pointer()
      try:
        self.max_transcription_length = max(len(transcription) for transcription in self.transcriptions)
      except:
        pass

    def preprocess(self, data_dir, data_file, transcriptions_file):
      # create data file from raw xml files from iam handwriting source.

      # build the list of xml files
      filelist = []
      # Set the directory you want to start from
      rootDir = data_dir
      for dirName, subdirList, fileList in os.walk(rootDir):
            #print('Found directory: %s' % dirName)
        for fname in fileList:
                #print('\t%s' % fname)
          filelist.append(f"{dirName}/{fname}")

      # function to read each individual xml file
      def getStrokes(filename):
          tree = ET.parse(filename)
          root = tree.getroot()

          result = []

          x_offset = 1e20
          y_offset = 1e20
          y_height = 0
          for i in range(1, 4):
              x_offset = min(x_offset, float(root[2][i].attrib['x']))
              y_offset = min(y_offset, float(root[2][i].attrib['y']))
              y_height = max(y_height, float(root[2][i].attrib['y']))
          y_height -= y_offset
          x_offset -= 100
          y_offset -= 100

          for stroke in root[3].findall('Stroke'):
              points = []
              for point in stroke.findall('Point'):
                  points.append(
                      [float(point.attrib['x']) - x_offset, float(point.attrib['y']) - y_offset])
              result.append(points)

          return result

      def getTranscriptionLines(filename):
          tree = ET.parse(filename)
          root = tree.getroot()

          string  = html.unescape(list(root[1][0].itertext())[0])
          string = string.replace(" '", "'")

          return string


      # converts a list of arrays into a 2d numpy int16 array
      def convert_stroke_to_array(stroke):

          n_point = 0
          for i in range(len(stroke)):
              n_point += len(stroke[i])
          stroke_data = np.zeros((n_point, 3), dtype=np.int16)

          prev_x = 0
          prev_y = 0
          counter = 0

          for j in range(len(stroke)):
              for k in range(len(stroke[j])):
                  stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                  stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                  prev_x = int(stroke[j][k][0])
                  prev_y = int(stroke[j][k][1])
                  stroke_data[counter, 2] = 0
                  if (k == (len(stroke[j]) - 1)):  # end of stroke
                      stroke_data[counter, 2] = 1
                  counter += 1
          return stroke_data

      # build stroke database of every xml file inside iam database
      strokes = []
      transcriptions = []
      for i in range(len(filelist)):
          try:
            if (filelist[i][-3:] == 'xml'):
              # print('processing ' + filelist[i])
              strokes.append(
                  convert_stroke_to_array(
                      getStrokes(
                          filelist[i])))
              transcriptions.append(
                getTranscriptionLines(filelist[i]))
          except Exception as e:
          	continue
              # print("Failed")
              # print(e)

      with open(data_file, "wb") as f:
        pickle.dump(strokes, f, protocol=2)
      with open(transcriptions_file, "wb") as f2:
        pickle.dump(transcriptions, f2, protocol=2)


    def load_preprocessed(self, data_file, transcriptions_file):
      with open(data_file, "rb") as f:
        self.raw_data = pickle.load(f)
      with open(transcriptions_file, "rb") as f:
        self.transcriptions = pickle.load(f)
      # goes thru the list, and only keeps the text entries that have more
      # than seq_length points
      self.data = []
      self.train_transcriptions = []
      self.valid_data = []
      self.valid_transcriptions = []
      counter = 0

      # every 1 in 20 (5%) will be used for validation data
      cur_data_counter = 0
      for data, transcription in zip(self.raw_data, self.transcriptions):
          if len(data) > (self.seq_length + 2):
              # removes large gaps from the data
              data = np.minimum(data, self.limit)
              data = np.maximum(data, -self.limit)
              data = np.array(data, dtype=np.float32)
              data[:, 0:2] /= self.scale_factor
              cur_data_counter = cur_data_counter + 1

              if cur_data_counter % 20 == 0:
                  self.valid_data.append(data)
                  self.valid_transcriptions.append(transcription)
              else:
                  self.data.append(data)
                  self.train_transcriptions.append(transcription)
                  # number of equiv batches this datapoint is worth
                  counter += int(len(data) / ((self.seq_length + 2)))

      # minus 1, since we want the ydata to be a shifted version of x data
      self.num_batches = int(counter / self.batch_size)

    def validation_data(self):
        # returns validation data
        x_batch = []
        y_batch = []
        transcriptions = []
        for i in range(self.batch_size):
            data = self.valid_data[i % len(self.valid_data)]
            transcription = self.valid_transcriptions[i % len(self.valid_transcriptions)]
            transcription = encode_ascii(transcription)
            transcription = tf.one_hot(transcription, len(alphabet))

            idx = 0
            x_batch.append(np.copy(data[idx:idx + self.seq_length]))
            y_batch.append(np.copy(data[idx + 1:idx + self.seq_length + 1]))
            transcriptions.append(transcription)
            # transcriptions.append(self.valid_transcriptions[i % len(self.valid_data)])
        return x_batch, y_batch, transcriptions

    def __getitem__(self, index):
      # returns a randomised, seq_length sized portion of the training data
      x_batch = []
      y_batch = []
      transcriptions = []
      j=0
      # idx = random.randint(0, len(data) - self.seq_length - 2)  # select random indice 
      idx = 0

      for i in range(index*self.batch_size, (index+1)*self.batch_size):
        pointer = (i-j)%len(self.data)
        data = self.data[pointer]
        transcription = self.transcriptions[pointer]
        transcription = encode_ascii(transcription)
        transcription = tf.one_hot(transcription, len(alphabet))
        transcriptions.append(transcription)
        # number of equiv batches this datapoint is worth
        n_batch = int(len(data) / ((self.seq_length + 2)))
        x = data
        x_batch.append(x[:-1])
        y_batch.append(x[1:])

              # x = np.copy(data[idx:idx + self.seq_length])
              # x_batch.append(x)
              # y_batch.append(np.copy(data[idx + 1:idx + self.seq_length + 1]))
              # # adjust sampling probability.
              # if random.random() < (1.0 / float(n_batch)):
              #     # if this is a long datapoint, sample this data more with
              #     # higher probability
              #     j = j+1

        # print(transcriptions[0].shape)
      largest_transcription = max(t.shape[0] for t in transcriptions)
      transcription_ascii = tf.stack([tf.pad(t, tf.constant([[0,largest_transcription-t.shape[0]],[0,0]]))
          for t in transcriptions])

      if self.maximise_seq_len:
        seq_len = min(x.shape[0] for x in x_batch)
        seq_len = min(self.seq_length, seq_len)
      else:
        seq_len = self.seq_length
      # print(seq_len)

      x_batch = [x[:seq_len] for x in x_batch]
      y_batch = [y[:seq_len] for y in y_batch]

      x_batch = tf.stack(x_batch)
      y_batch = tf.stack(y_batch)

      # print(transcription_ascii.shape)
      # print(x_batch.shape)
      # print(y_batch.shape)

      return [x_batch, transcription_ascii], y_batch

    def get_sample(self, i):
        # returns a randomised, seq_length sized portion of the training data
        pointer = i%len(self.data)
        data = self.data[pointer]
        transcription = self.transcriptions[pointer]
        transcription_ascii = encode_ascii(transcription[:self.seq_length])
        transcription_ascii = tf.one_hot(transcription_ascii, len(alphabet))

        return data, transcription, transcription_ascii

    def next_batch(self):
      # returns a randomised, seq_length sized portion of the training data
      x_batch = []
      y_batch = []
      transcriptions = []

      for _ in range(self.batch_size):
        data = self.data[self.pointer]
        transcription = self.transcriptions[self.pointer]
        transcription = encode_ascii(transcription)
        transcription = tf.one_hot(transcription, len(alphabet))
        transcriptions.append(transcription)
        # number of equiv batches this datapoint is worth
        n_batch = int(len(data) / ((self.seq_length + 2)))
        idx = random.randint(0, len(data) - self.seq_length - 2)
        x_batch.append(np.copy(data[idx:idx + self.seq_length]))
        y_batch.append(np.copy(data[idx + 1:idx + self.seq_length + 1]))
        self.tick_batch_pointer()
      largest_transcription = max(t.shape[0] for t in transcriptions)
      transcription_ascii = tf.stack([tf.pad(t, tf.constant([[0, int(largest_transcription-t.shape[0])],[0,0]]))
          for t in transcriptions])

      x_batch = tf.stack(x_batch)
      y_batch = tf.stack(y_batch)

      return x_batch, y_batch, transcription_ascii

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        self.pointer = 0

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.train_transcriptions) / self.batch_size))
      
      