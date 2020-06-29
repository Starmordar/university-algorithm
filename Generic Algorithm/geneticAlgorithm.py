import numpy as np

def getEquation(u, w, x, y, z): 
    return u*w*z*x + 2*u*w*z + 2*x*y + w + x

def cal_pop_fitness(equation_inputs, pop):
    fitness = numpy.sum(pop * equation_inputs, axis = 1)
    return fitness

def weighted_random_choice(chromosomes):
    max = sum(chromosome.fitness for chromosome in chromosomes)
    pick = random.uniform(0, max)
    current = 0
    for chromosome in chromosomes:
        current += chromosome.fitness
        if current > pick:
            return chromosome

def select_mating_pool(pop, fitness, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)

    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            random_value = numpy.random.uniform(-200, 200, 1)

            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

n = 6
chromosome = np.random.randint(-200, 200 , (n, 4))
print("chromosomes :",chromosome)
epoch = 0

while epoch <  200 :
    firstChromosome = chromosome[:,0]
    secondChromosome = chromosome[:,1]
    thirdChromosome = chromosome[:,2]
    fourthChromosome = chromosome[:,3]

    objective = abs(
        40 - 
        firstChromosome * secondChromosome *fourthChromosome  -
        2*secondChromosome * thirdChromosome * fourthChromosome - 
        3*thirdChromosome * firstChromosome -
        fourthChromosome  - thirdChromosome
    )
    print("Fitness object :", objective)
    
    fitness =  1/(1 + objective)
    print("Fitness :",fitness)
    
    total = fitness.sum()
    print("Total :",total)
    
    prob = fitness/total
    print("Probability :",prob)
    
    cum_sum = np.cumsum(prob)
    print("Cumulative Sum :", cum_sum)
    
    Ran_nums = np.random.random((chromosome.shape[0]))
    print("Random Numbers :",Ran_nums)
    
    chromosome_2 = np.zeros((chromosome.shape[0],4))
    
    for i in range(Ran_nums.shape[0]):
        for j in range(chromosome.shape[0]):
            if Ran_nums[i]  < cum_sum[j]:
                chromosome_2[i,:] = chromosome[j,:]
                break
            
    chromosome = chromosome_2
    print("Chromosomes after updation :",chromosome)
        
    R = [np.random.random() for i in range(n)]
    print("Random Values :",R)
    
    pc = 0.25
    flag = Ran_nums < pc
    print("Flagged Values :",flag)
    
    cross_chromosome = chromosome[[(i == True) for i in flag]]
    print("Cross chromosome :",cross_chromosome)
    len_cross_chrom = len(cross_chromosome)
    
    cross_values = np.random.randint(1,3,len_cross_chrom)
    print("Cross Values :",cross_values)
    
    cpy_chromosome = np.zeros(cross_chromosome.shape)
    
    for i in range(cross_chromosome.shape[0]):
        cpy_chromosome[i , :] = cross_chromosome[i , :]
        
    if len_cross_chrom == 1:
        cross_chromosome = cross_chromosome
    else :
        for i in range(len_cross_chrom):
            c_val = cross_values[i]
            if i == len_cross_chrom - 1 :
                cross_chromosome[i , c_val:] = cpy_chromosome[0 , c_val:]
            else :
                cross_chromosome[i , c_val:] = cpy_chromosome[i+1 , c_val:]
        
    print("Crossovered Chromosome :",cross_chromosome)
    
    index_chromosome = 0
    index_newchromosome = 0
    for i in flag :
        if i == True :
            chromosome[index_chromosome, :] = cross_chromosome[index_newchromosome, :]
            index_newchromosome = index_newchromosome + 1
        index_chromosome = index_chromosome + 1 
    
    print("New Chromosomes:", chromosome)
    
    a ,b = chromosome.shape[0] ,chromosome.shape[1]
    total_gen = a*b
    print("Total Generations :",total_gen)
    
    pm = 0.1
    no_of_mutations = int(np.round(pm * total_gen))
    print("No. of Mutations :" ,no_of_mutations)
    
    gen_num = np.random.randint(0,total_gen - 1, no_of_mutations)
    print(" Generated Random Numbers : " , gen_num)
    
    Replacing_num = np.random.randint(-200, 200, no_of_mutations)
    print(" Numbers to be replaced : " , Replacing_num)
    
    for i in range(no_of_mutations):
        a = gen_num[i]
        row = a//4
        col = a%4
        chromosome[row , col] = Replacing_num[i]
    
    print(" Chromosomes After Mutation : " , chromosome)
  
    epoch = epoch + 1