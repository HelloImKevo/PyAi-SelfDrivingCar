# Self Driving Car

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Fix for the window collapsing to a small view on Mac devices
# If the user re-sizes the window, this will crash at runtime
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '700')
Config.set('graphics', 'resizable', 0)

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5, 3, 0.9)
action2rotation = [0, 20, -20]
last_reward = 0
scores = []

# Initializing the map
first_update = True


def init():
    # The sand that causes the car to slow down or avoid
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((map_width, map_height))

    # Goal is to reach the upper left of the map (agent gets a bad score if it touches the edge)
    goal_x = 20
    goal_y = map_height - 20
    first_update = False


# Initializing the last distance
last_distance = 0


# Creating the game class
class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global map_width
        global map_height

        # Width of the map (horizontal edge)
        map_width = self.width
        # Height of the map (vertical edge)
        map_height = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

        # Update our AI agent (the brain for the Car)
        # Action is represented by a torch.Tensor
        action = brain.update(last_reward, last_signal)
        # print("last_reward: %s, last_signal: %s, action: %s" % (last_reward, last_signal, action))
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            # We are on the sand
            last_reward = -0.95
        else:  # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            # We are further from the goal
            last_reward = -0.30
            # We are closer to the goal
            if distance < last_distance:
                last_reward = 0.12

        # If we are too close to the edge of the map, use a bad reward
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y
        last_distance = distance


# Creating the car class
class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

        self.signal1 = int(np.sum(sand[
                                  int(self.sensor1_x) - 10:int(self.sensor1_x) + 10,
                                  int(self.sensor1_y) - 10:int(self.sensor1_y) + 10])) / 400.
        self.signal2 = int(np.sum(sand[
                                  int(self.sensor2_x) - 10:int(self.sensor2_x) + 10,
                                  int(self.sensor2_y) - 10:int(self.sensor2_y) + 10])) / 400.
        self.signal3 = int(np.sum(sand[
                                  int(self.sensor3_x) - 10:int(self.sensor3_x) + 10,
                                  int(self.sensor3_y) - 10:int(self.sensor3_y) + 10])) / 400.

        if (self.sensor1_x > map_width - 10
                or self.sensor1_x < 10
                or self.sensor1_y > map_height - 10
                or self.sensor1_y < 10):
            self.signal1 = 1.

        if (self.sensor2_x > map_width - 10
                or self.sensor2_x < 10
                or self.sensor2_y > map_height - 10
                or self.sensor2_y < 10):
            self.signal2 = 1.

        if (self.sensor3_x > map_width - 10
                or self.sensor3_x < 10
                or self.sensor3_y > map_height - 10
                or self.sensor3_y < 10):
            self.signal3 = 1.


# Define the sensors (see kivy tutorials: https://kivy.org/docs/tutorials/pong.html)
class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


# Adding the painting tools
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            try:
                sand[int(touch.x), int(touch.y)] = 1
            except IndexError:
                pass

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.
            density = n_points / length
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10: int(touch.x) + 10,
            int(touch.y) - 10: int(touch.y) + 10] = 1
            last_x = x
            last_y = y


# Adding the API Buttons (clear, save and load)
class CarApp(App):
    painter = None
    btn_clear = None
    btn_save = None
    btn_load = None

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)

        self.painter = MyPaintWidget()
        btn_clear: Button = Button(text='clear')
        btn_save: Button = Button(text='save', pos=(parent.width, 0))
        btn_load: Button = Button(text='load', pos=(2 * parent.width, 0))

        btn_clear.bind(on_release=self.clear_canvas)
        btn_save.bind(on_release=self.save)
        btn_load.bind(on_release=self.load)

        parent.add_widget(self.painter)
        parent.add_widget(btn_clear)
        parent.add_widget(btn_save)
        parent.add_widget(btn_load)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((map_width, map_height))

    # noinspection PyMethodMayBeStatic
    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    # noinspection PyMethodMayBeStatic
    def load(self, obj):
        print("loading last saved brain...")
        brain.load()
