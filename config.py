import pygame as py
py.font.init()

FPS = 60
WIN_WIDTH = 1600
WIN_HEIGHT = 900
STARTING_POS = (WIN_WIDTH/1.5, WIN_HEIGHT-100)
BAD_GENOME_TRESHOLD = 200

INPUT_NEURONS = 9
OUTPUT_NEURONS = 4

CAR_DBG = False
FRICTION  = -0.1
MAX_VEL = 10
MAX_VEL_REDUCTION = 1              
ACC_STRENGHT = 0.2
BRAKE_STREGHT = 1
TURN_VEL = 2
SENSOR_DISTANCE = 200
ACTIVATION_TRESHOLD = 0.5

ROAD_DBG = False
MAX_ANGLE = 1
MAX_DEVIATION = 300
SPACING = 200
NUM_POINTS  = 15                
SAFE_SPACE = SPACING + 50       
ROAD_WIDTH = 200

NODE_RADIUS = 20
NODE_SPACING = 5
LAYER_SPACING = 100
CONNECTION_WIDTH = 1

WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
RED = (255,0,0)
OUTPUT_NODES_COLOR = (74, 66, 189)
OUTPUT_NODES_COLOR_2 = (26, 16, 173)
OUTPUT_NODES_COLOR_3 = (55, 52, 107)
OUTPUT_NODES_COLOR_4 = (129, 124, 214)
INPUT_NODES_COLOR = (109, 195, 232)
INPUT_NODES_COLOR_2 = (41, 171, 227)
INPUT_NODES_COLOR_3 = (25, 88, 115)
INPUT_NODES_COLOR_4 = (39, 69, 82)
MIDDLE_NODES_COLOR = (242, 250, 132)
MIDDLE_NODES_COLOR_2 = (215, 227, 50)
MIDDLE_NODES_COLOR_3 = (71, 57, 17)
MIDDLE_NODES_COLOR_4 = (156, 141, 98)

NODE_FONT = py.font.SysFont("orange juice", 20)

GEN = 0
ACC = 0
BRAKE = 1
TURN_LEFT = 2
TURN_RIGHT = 3
INPUT = 0
MIDDLE = 1
OUTPUT = 2
