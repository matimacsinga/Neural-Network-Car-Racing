import pygame as py
import neat
import os
from network_visualisation import NN
from car import Car
from world import World
from track import Track
from config import *
py.font.init()


background = py.Surface((WIN_WIDTH, WIN_HEIGHT))
background.fill(WHITE)

def draw_win(cars, road, world):
    road.draw(world)
    for car in cars:
        car.draw(world)
    world.bestNN.draw(world)
    py.display.update()
    world.win.blit(background, (0,0))

def main(genomes = [], config = []):

    networks = []
    genome = []
    cars = []
    time = 0

    world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)
    world.win.blit(background, (0,0))
    neural_networks = []

    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        networks.append(net)
        cars.append(Car(0, 0, 0))
        g.fitness = 0
        genome.append(g)
        neural_networks.append(NN(config, g, (90, 210)))

    road = Track(world)
    clock = py.time.Clock()

    run = True
    while run:
        time += 1
        clock.tick(FPS)
        world.updateScore(0)

        for event in py.event.get():
            if event.type == py.QUIT:
                run = False
                py.quit()
                quit()

        (xb, yb) = (0,0)
        i = 0
        while(i < len(cars)):
            car = cars[i]

            input = car.getInputs(world, road)
            input.append(car.vel/MAX_VEL)
            car.commands = networks[i].activate(tuple(input))

            y_old = car.y
            (x, y) = car.move(road,time)

            if time>10 and (car.detectCollision(road) or y > world.getBestCarPos()[1] + BAD_GENOME_TRESHOLD or y>y_old or car.vel < 0.1):
                genome[i].fitness -= 1
                cars.pop(i)
                networks.pop(i)
                genome.pop(i)
                neural_networks.pop(i)
            else:
                genome[i].fitness += -(y - y_old)/100
                if(genome[i].fitness > world.getScore()):
                    world.updateScore(genome[i].fitness)
                    world.bestNN = neural_networks[i]
                    world.bestInputs = input
                    world.bestCommands = car.commands
                i += 1

            if y < yb:
                (xb, yb) = (x, y)

        if len(cars) == 0:
            run = False
            break

        world.updateBestCarPos((xb, yb))
        road.update(world)
        draw_win(cars, road, world)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats =neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 10000)     

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_file.txt")
    run(config_path)
