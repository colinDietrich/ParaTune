from random import choices, randint, randrange, random, shuffle, sample
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from typing import List, Tuple, Callable, Optional
from IPython.display import clear_output
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from ParaTune.media.SPDCCrystal import SPDCCrystal
from ParaTune.interactions.InteractionSPDC import InteractionSPDC
from ParaTune.light.Pulse import Pulse

class GA:
    def __init__(self, size: int, option: str, Ps: float, Pu1: float, Pu2: float,
                 weight_Pm: float,
                 max_length: float, min_length: float,
                number_grid_points_z: int, number_grid_points_freq:int, freq_span: float,
                pump_amplitude_freq: List[float], freq_banwidth: float, medium: str, wavelength_central: float,
                wavelength_array: List[float],
                pump: str, signal: str, idler: str, target: List[float], domain_width: Optional[float] = None):
        self.size = size
        self.option = option
        self.Ps = Ps
        self.Pu1 = Pu1
        self.Pu2 = Pu2
        self.weight_Pm = weight_Pm
        self.domain_width = domain_width
        self.max_length = max_length
        self.min_length = min_length
        self.number_grid_points_z = number_grid_points_z
        self.number_grid_points_freq = number_grid_points_freq
        self.pump_amplitude_freq = pump_amplitude_freq
        self.freq_banwidth = freq_banwidth
        self.freq_span = freq_span
        self.medium = medium
        self.wavelength_central = wavelength_central
        self.wavelength_array = wavelength_array
        self.pump = pump
        self.signal = signal
        self.idler = idler
        self.target = target


    def generate_genome(self, option: str) -> 'SPDCCrystal':
        length = None
        if(self.max_length == self.min_length): length = self.max_length
        return SPDCCrystal(configuration=option,
                        medium=self.medium,
                        number_grid_points_z=self.number_grid_points_z,
                        wavelength_central=self.wavelength_central,
                        pump=self.pump,
                        signal=self.signal,
                        idler=self.idler,
                        length=length,
                        maximum_length=self.max_length,
                        minimum_length=self.min_length,
                        domain_width=self.domain_width)

    def generate_population(self, size: int, option: str) -> List['SPDCCrystal']:
        return [self.generate_genome(option) for _ in range(size)]

    def fitness_function(self, genome: 'SPDCCrystal') -> float:
        simulation_spdc = InteractionSPDC(
            self.wavelength_central, 
            self.freq_span*2*np.pi, 
            genome.length,
            self.number_grid_points_freq, 
            genome.number_grid_points_z, 
            genome.n_p, genome.n_s,genome.n_i,
            genome.k_p, genome.k_s,genome.k_i, 
            genome.wavevector_mismatch,
            genome.deff, 
            genome.domain_bounds, 
            self.freq_banwidth,
            dimensions=1)
        
        parameters = np.array(genome.poling_function(np.array(genome.domain_values)), dtype=np.float32)[:len(genome.z_grid)]
        _, As_out, Ai_out, _, _, _, _, _, _, _ = simulation_spdc.run(self.pump_amplitude_freq, parameters)
        genome.signal_spectrum = As_out
        genome.idler_spectrum = Ai_out

        fit = - self.mean_squared_error(np.abs(self.target), np.abs(As_out)) - self.mean_squared_error(np.abs(self.target), np.abs(Ai_out))
        return fit
    
    def mean_squared_error(self, array1, array2) -> float:
        array1 = array1 / np.linalg.norm(np.abs(array1))
        array2 = array2 / np.linalg.norm(np.abs(array2))
        return np.square(np.subtract(array1, array2)).mean()

    def single_point_crossover(self, a: 'SPDCCrystal', b: 'SPDCCrystal') -> Tuple['SPDCCrystal', 'SPDCCrystal']:
        """
        Performs single-point crossover on two genomes.
        
        Args:
            a (SPDCCrystal): The first parent genome.
            b (SPDCCrystal): The second parent genome.
        
        Returns:
            Tuple[SPDCCrystal, SPDCCrystal]: The resulting pair of genomes after crossover.
        """
        length = min(len(a.domain_values), len(b.domain_values))
        if length < 2:
            return a, b
        
        p = randint(1, length - 1)
        saveA = a.domain_values
        saveB = b.domain_values
        
        a.domain_values = saveA[:p] + saveB[p:]
        b.domain_values = saveB[:p] + saveA[p:]
        
        if a.length > self.max_length or a.length < self.min_length:
            a.domain_values = saveA
        if b.length > self.max_length or b.length < self.min_length:
            b.domain_values = saveB
        
        return a, b

    def uniform_crossover(self, a: 'SPDCCrystal', b: 'SPDCCrystal') -> Tuple['SPDCCrystal', 'SPDCCrystal']:
        """
        Performs uniform crossover on two genomes.
        
        Args:
            a (SPDCCrystal): The first parent genome.
            b (SPDCCrystal): The second parent genome.
        
        Returns:
            Tuple[SPDCCrystal, SPDCCrystal]: The resulting pair of genomes after crossover.
        """
        la = len(a.domain_values)
        lb = len(b.domain_values)
        minL = min(la, lb)
        offset_a = la - lb if la > lb else 0
        offset_b = lb - la if lb > la else 0
        
        i_new = 0
        i_old = 0
        save_a = a.domain_values[:offset_a]
        save_b = b.domain_values[:offset_b]
        
        while i_old < minL:
            up = minL - i_old
            i_new += randint(1, up)
            if random() < self.Pu2:
                save_a += b.domain_values[offset_b + i_old: offset_b + i_new]
                save_b += a.domain_values[offset_a + i_old: offset_a + i_new]
            else:
                save_a += a.domain_values[offset_a + i_old: offset_a + i_new]
                save_b += b.domain_values[offset_b + i_old: offset_b + i_new]
            i_old = i_new
        
        a.domain_values = save_a + a.domain_values[offset_a + i_old:]
        b.domain_values = save_b + b.domain_values[offset_b + i_old:]
        
        return a, b

    def mutation(self, genome: 'SPDCCrystal', num: int = 1) -> 'SPDCCrystal':
        """
        Mutates a genome with a probability inversely proportional to its length.
        
        Args:
            genome (SPDCCrystal): The genome to mutate.
            num (int): The number of mutations to perform (default is 1).
        
        Returns:
            SPDCCrystal: The mutated genome.
        """
        Pm = 1 / len(genome.domain_values) * self.weight_Pm
        for _ in range(num):
            index = randrange(len(genome.domain_values))
            if random() <= Pm:
                genome.domain_values[index] *= -1
        return genome

    def population_fitness(self, population: List['SPDCCrystal'], fitness_func: Callable[['SPDCCrystal'], float]) -> Tuple[float, List[float]]:
        """
        Calculates the total fitness of a population.
        
        Args:
            population (List[Crystal]): The population to evaluate.
            fitness_func (Callable[[Crystal], float]): The fitness function.
        
        Returns:
            Tuple[float, List[float]]: The total fitness and a list of individual fitness values.
        """
        total_fitness = 0
        fitness_values = []
        for genome in population:
            fit = fitness_func(genome)
            total_fitness += fit
            fitness_values.append(fit)
        return total_fitness, fitness_values

    def selection_pair(self, population: List['SPDCCrystal'], fitness_func: Callable[['SPDCCrystal'], float]) -> Tuple['SPDCCrystal', 'SPDCCrystal']:
        """
        Selects a pair of genomes from the population based on their fitness.
        
        Args:
            population (List[Crystal]): The population to select from.
            fitness_func (Callable[[Crystal], float]): The fitness function.
        
        Returns:
            Tuple[Crystal, Crystal]: A pair of selected genomes.
        """
        return tuple(choices(population=population,
                             weights=[fitness_func(gene) for gene in population],
                             k=2))

    def sort_population(self, population: List['SPDCCrystal'], fitness_values: List[float]) -> List['SPDCCrystal']:
        """
        Sorts the population based on their fitness in descending order.
        
        Args:
            population (List[Crystal]): The population to sort.
            fitness_values (List[float]): The fitness values associated with the population.
        
        Returns:
            List[Crystal]: The sorted population.
        """
        return [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]

    def print_stats(self, population: List['SPDCCrystal'], generation_id: int, metrics: dict) -> None:
        """
        Prints statistics about the current generation.
        
        Args:
            population (List[Crystal]): The current population.
            generation_id (int): The ID of the current generation.
            metrics (dict): Dictionary containing metrics for the population.
        """
        clear_output(wait=True)
        
        print(f"GENERATION {generation_id:02d}")
        print(f"\t - Population length: {len(population)}")
        print(f"\t - Avg. Fitness: {metrics['average_fitness'][generation_id]:.6f}")
        print(f"\t - Avg. Length: {metrics['average_length'][generation_id]:.6f}")
        print(f"\t - Best Crystal")
        print(f"\t\t -> length: {metrics['best_length'][generation_id]}")
        print(f"\t\t -> fitness: ({metrics['best_fitness'][generation_id]:.6f})")
        print(f"\t - Worst Crystal")
        print(f"\t\t -> length: {metrics['worst_length'][generation_id]}")
        print(f"\t\t -> fitness: ({metrics['worst_fitness'][generation_id]:.6f})")

        best_genome = population[0]

        fig, axs = plt.subplots(2, 2, figsize=(12, 6))

        # Plot the target and simulated PMF profiles for the signal
        axs[0, 0].plot(self.wavelength_array, np.abs(self.target)/np.linalg.norm(np.abs(self.target)), '--', label='Target Signal', color='black')
        axs[0, 0].plot(self.wavelength_array, np.abs(best_genome.signal_spectrum)/np.linalg.norm(np.abs(best_genome.signal_spectrum)), label='Simulated Signal', color='crimson')
        axs[0, 0].legend()
        axs[0, 0].set_title('Signal PMF Profiles')
        axs[0, 0].set_xlabel('Wavelength (nm)')
        axs[0, 0].set_ylabel('PMF Amplitude')

        # Plot the target and simulated PMF profiles for the idler
        axs[0, 1].plot(self.wavelength_array, np.abs(self.target)/np.linalg.norm(np.abs(self.target)), '--', label='Target Signal', color='royalblue')
        axs[0, 1].plot(self.wavelength_array, np.abs(best_genome.idler_spectrum)/np.linalg.norm(np.abs(best_genome.idler_spectrum)), label='Simulated Signal', color='forestgreen')
        axs[0, 1].legend()
        axs[0, 1].set_title('Idler PMF Profiles')
        axs[0, 1].set_xlabel('Wavelength (nm)')
        axs[0, 1].set_ylabel('PMF Amplitude')

        axs[1, 0].plot(metrics['average_fitness'], label='Average Fitness', color='b')
        axs[1, 0].legend()
        axs[1, 0].set_xlabel('Generation')
        axs[1, 0].set_ylabel('Average Fitness')
        axs[1, 0].set_title('Average Fitness Over Generations')

        # Plot best fitness over generations
        axs[1, 1].plot(metrics['best_fitness'], label='Best Fitness', color='forestgreen')
        axs[1, 1].legend()
        axs[1, 1].set_title('Best Fitness Over Generations')
        axs[1, 1].set_xlabel('Generation')
        axs[1, 1].set_ylabel('Best Fitness')

        plt.tight_layout()
        plt.show()

    def run_evolution(self, nb_generation: int, printer: bool = False, fitness_limit: float = 0, nb_level: int = 1, 
                      restart: Optional[int] = None, restart_depth: int = 4, population: Optional[List['SPDCCrystal']] = None) -> Tuple[List['SPDCCrystal'], int]:
        """
        Runs the genetic algorithm for a given number of generations.
        
        Args:
            nb_generation (int): Number of generations to run.
            printer (bool): Whether to print statistics during evolution.
            fitness_limit (float): Fitness limit to stop the evolution early.
            nb_level (int): Level for domain width adjustment.
            restart (Optional[int]): Generation interval for restarting the population.
            restart_depth (int): Number of new genomes to inject during restart.
            population (Optional[List[Crystal]]): Initial population. If None, a new population will be generated.
        
        Returns:
            Tuple[List[Crystal], int]: The final population and the number of generations completed.
        """
        if population is None:
            population = self.generate_population(self.size, self.option)
        else:
            total_fitness, fitness_values = self.population_fitness(population, self.fitness_function)
            population = self.sort_population(population, fitness_values)
        
        self.metrics = {
            'best_fitness': [],
            'best_length': [],
            'average_fitness': [],
            'average_length': [],
            'worst_fitness': [],
            'worst_length': [],
            'std_dev_fitness': [],
            'generations': 0
        }
        level = 1

        for i in tqdm(range(nb_generation), desc="Evolution Progress"):
            start_time = time.time()
            total_fitness, fitness_values = self.population_fitness(population, self.fitness_function)
            self.metrics['best_fitness'].append(max(fitness_values))
            self.metrics['average_fitness'].append(np.mean(fitness_values))
            self.metrics['worst_fitness'].append(min(fitness_values))
            self.metrics['best_length'].append(population[0].length)
            self.metrics['average_length'].append(np.mean([genome.length for genome in population]))
            self.metrics['worst_length'].append(population[-1].length)
            self.metrics['std_dev_fitness'].append(np.std(fitness_values))
            self.metrics['generations'] += 1

            if printer:
                self.print_stats(population, i, self.metrics)

            next_generation = []

            if i in nb_level:
                level += 1
                self.domain_width /= 2

            # Crossover and mutation
            best_genomes = copy.deepcopy(population[:self.size // 2])
            shuffle(population)

            # Crossover and mutation
            best_genomes = copy.deepcopy(population[0:self.size//10])
            shuffle(population)
            for j in range(int(len(population) / 2)):
                parents = [population[2 * j], population[2 * j + 1]]
                if random() < self.Pu1:
                    parents[0], parents[1] = self.uniform_crossover(parents[0], parents[1])
                if random() < self.Ps:
                    parents[0], parents[1] = self.single_point_crossover(parents[0], parents[1])
                parents[0] = self.mutation(parents[0])
                parents[1] = self.mutation(parents[1])
                parents[0].update(level=level)
                parents[1].update(level=level)
                population[j].update(level=level)
                next_generation += [parents[0], parents[1]]

            new_population_injection = self.generate_population(self.size // 2, 'random')
            N = self.size - len(best_genomes) - len(new_population_injection)
            population = best_genomes + new_population_injection + sample(next_generation, N)
            total_fitness, fitness_values = self.population_fitness(population, self.fitness_function)
            population = self.sort_population(population, fitness_values)

            if restart is not None:
                if i % restart == 0 and i != 0:
                    new_population_injection = self.generate_population(restart_depth, 'random')
                    population = new_population_injection + population[:self.size - restart_depth]
                    total_fitness, fitness_values = self.population_fitness(population, self.fitness_function)
                    population = self.sort_population(population, fitness_values)

            end_time = time.time()

        return population, i