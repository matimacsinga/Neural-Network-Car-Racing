import pygame as py
from config import *
from car import decodeCommand
from vector import Vector
from node import *

py.font.init()

class NN:

    def __init__(self, config, genome, pos):
        self.input_nodes = []
        self.output_nodes = []
        self.nodes = []
        self.genome = genome
        self.pos = (int(pos[0]+NODE_RADIUS), int(pos[1]))
        input_names = ["Sensor T", "Sensor TR", "Sensor R", "Sensor BR", "Sensor B", "Sensor BL", "Sensor L", "Sensor TL", "Speed"]
        output_names = ["Accelerate", "Brake", "Turn Left", "Turn Right"]
        middle_nodes = [n for n in genome.nodes.keys()]
        nodeIdList = []

        h = (INPUT_NEURONS-1)*(NODE_RADIUS*2 + NODE_SPACING)
        for i, input in enumerate(config.genome_config.input_keys):
            n = Node(input, pos[0]+100, pos[1]+int(-h/2 + i*(NODE_RADIUS*2 + NODE_SPACING))+250, INPUT, [INPUT_NODES_COLOR, INPUT_NODES_COLOR_2, INPUT_NODES_COLOR_3, INPUT_NODES_COLOR_4], input_names[i], i)
            self.nodes.append(n)
            nodeIdList.append(input)

        h = (OUTPUT_NEURONS-1)*(NODE_RADIUS*2 + NODE_SPACING)
        for i,out in enumerate(config.genome_config.output_keys):
            n = Node(out+INPUT_NEURONS, pos[0] + 2*(LAYER_SPACING+2*NODE_RADIUS) + 100, pos[1]+int(-h/2 + i*(NODE_RADIUS*2 + NODE_SPACING)) + 250, OUTPUT, [OUTPUT_NODES_COLOR, OUTPUT_NODES_COLOR_2, OUTPUT_NODES_COLOR_3, OUTPUT_NODES_COLOR_4], output_names[i], i)
            self.nodes.append(n)
            middle_nodes.remove(out)
            nodeIdList.append(out)

        h = (len(middle_nodes)-1)*(NODE_RADIUS*2 + NODE_SPACING)
        for i, m in enumerate(middle_nodes):
            n = Node(m, self.pos[0] + (LAYER_SPACING+2*NODE_RADIUS) + 100, self.pos[1]+int(-h/2 + i*(NODE_RADIUS*2 + NODE_SPACING)) + 250, MIDDLE, [MIDDLE_NODES_COLOR, MIDDLE_NODES_COLOR_2, MIDDLE_NODES_COLOR_3, MIDDLE_NODES_COLOR_4])
            self.nodes.append(n)
            nodeIdList.append(m)

        self.connections = []
        for c in genome.connections.values():
            if c.enabled:
                input, output = c.key
                self.connections.append(Connection(self.nodes[nodeIdList.index(input)],self.nodes[nodeIdList.index(output)], c.weight))

    def draw(self, world):
        for c in self.connections:
            c.drawConnection(world)
        for node in self.nodes:
            node.draw_node(world)