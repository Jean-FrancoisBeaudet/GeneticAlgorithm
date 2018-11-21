package GenericGeneticAlgorithm;

import java.util.*;
import java.util.concurrent.TimeUnit;

public class GeneticAlgorithm<T extends Individual<T>> {

    private final static int DEFAULT_POPULATION_SIZE = 250;
    private final static double DEFAULT_RANK_SPACE_RATIO = 0.25;
    private final static double DEFAULT_MUTATION_RATIO = 0.01;
    private final static double DEFAULT_SELECTION_RATIO = 0.80;
    private final static double DEFAULT_UNIFORMITY_RATIO = 0.02;

    private int generationCount;
    private final Random random = new Random();
    private Population<T> currentPopulation;
    private Population<T> initialPopulation;
    private IgenerateRandomIndividual<T> generateRandomIndividualFunction;
    private int populationSize;
    private double rankSpaceRatio;
    private double mutationRatio;
    private double selectionRatio;
    private double uniformityRatio;

    public GeneticAlgorithm(int populationSize,
                            double rankSpaceRatio,
                            double mutationRatio,
                            double selectionRatio,
                            double uniformityRatio,
                            IgenerateRandomIndividual<T> randomIndividualFunction) {

        this.assertPopulationSize(populationSize);
        this.assertRankSpaceRation(rankSpaceRatio);
        this.assertMutationRation(mutationRatio);
        this.assertSelectionRation(selectionRatio);
        this.assertUniformityRation(uniformityRatio);

        this.populationSize = populationSize;
        this.rankSpaceRatio = rankSpaceRatio;
        this.mutationRatio = mutationRatio;
        this.selectionRatio = selectionRatio;
        this.uniformityRatio = uniformityRatio;
        this.generateRandomIndividualFunction = randomIndividualFunction;
        this.reset(populationSize);
    }

    public GeneticAlgorithm(IgenerateRandomIndividual<T> randomIndividualFunction) {
        this.populationSize = DEFAULT_POPULATION_SIZE;
        this.rankSpaceRatio = DEFAULT_RANK_SPACE_RATIO;
        this.mutationRatio = DEFAULT_MUTATION_RATIO;
        this.selectionRatio = DEFAULT_SELECTION_RATIO;
        this.uniformityRatio = DEFAULT_UNIFORMITY_RATIO;
        this.generateRandomIndividualFunction = randomIndividualFunction;
        this.reset(populationSize);
    }

    public void setDefaultParameters(){
        this.populationSize = DEFAULT_POPULATION_SIZE;
        this.rankSpaceRatio = DEFAULT_RANK_SPACE_RATIO;
        this.mutationRatio = DEFAULT_MUTATION_RATIO;
        this.selectionRatio = DEFAULT_SELECTION_RATIO;
        this.uniformityRatio = DEFAULT_UNIFORMITY_RATIO;
    }

    public int getPopulationSize() { return this.populationSize; }

    public double getRankSpaceRatio() {
        return this.rankSpaceRatio;
    }

    public double getMutationRatio() {
        return this.mutationRatio;
    }

    public double getSelectionRatio() {
        return this.selectionRatio;
    }

    public double getUniformityRatio() {
        return this.uniformityRatio;
    }

    public void setPopulationSize(int populationSize) {
        this.assertPopulationSize(populationSize);
        this.populationSize = populationSize;
    }

    public void setRankSpaceRatio(double rankSpaceRatio) {
        this.assertRankSpaceRation(rankSpaceRatio);
        this.rankSpaceRatio = rankSpaceRatio;
    }

    public void setMutationRatio(double mutationRatio) {
        this.assertMutationRation(mutationRatio);
        this.mutationRatio = mutationRatio;
    }

    public void setSelectionRatio(double selectionRatio) {
        this.assertSelectionRation(selectionRatio);
        this.selectionRatio = selectionRatio;
    }

    public void setUniformityRatio(double uniformityRatio) {
        this.assertUniformityRation(uniformityRatio);
        this.uniformityRatio = uniformityRatio;
    }

    public int getGenerationCount() {
        return generationCount;
    }

    public Population<T> getCurrentPopulation() {
        return this.currentPopulation;
    }

    public Population<T> getInitialPopulation() {
        return this.initialPopulation;
    }

    public double getBestFitnessScore() {
        return this.getBestIndividual().getFitnessScore();
    }

    public double getWorstFitnessScore() {
        return this.getWorstIndividual().getFitnessScore();
    }

    public T getBestIndividual() {
        return this.currentPopulation.getIndividualAtIndex(0);
    }

    public T getWorstIndividual() {
        return this.currentPopulation.getIndividualAtIndex(this.populationSize - 1);
    }

    public void startForTimeInMilliSeconds(long timeInMilliSeconds) {
        long stop = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeInMilliSeconds) ;
        while(stop > System.nanoTime()) {
            this.evolveToNextGeneration();
        }
    }

    public void startUntilSpecificFitnessScore(double fitnessScore) {
        while (this.getBestFitnessScore() < fitnessScore) {
            this.evolveToNextGeneration();
        }
    }

    public void startForGenerationCount(int generationCount) {
        for (int i = 0; i < generationCount; i++) {
            this.evolveToNextGeneration();
        }
    }

    public void reset(int populationSize){
        this.generateInitialPopulation(populationSize);
        this.currentPopulation = this.initialPopulation;
        this.calculateFitness();
        this.sortPopulationByFitness();
        this.generationCount = 1;
    }

    public void addToCurrentPopulation(T individual){
        this.currentPopulation.removeIndividualAt(this.populationSize - 1);
        individual.calculateFitness();
        this.currentPopulation.addIndividual(individual);
        this.sortPopulationByFitness();
    }

    private void generateInitialPopulation(int populationSize) {
        this.initialPopulation = new Population<>(populationSize);
        for (int i = 0; i < this.populationSize; i++) {
            this.initialPopulation.addIndividual(this.generateRandomIndividualFunction.generateRandomIndividual());
        }
    }

    private void evolveToNextGeneration() {

        List<T> primeIndividuals = this.selectPrimeIndividuals();
        Population<T> newPopulation = new Population<>(this.populationSize);
        int nbOfIndividualThatJumpToNextGeneration = (int) (this.populationSize * this.uniformityRatio);

        int index = 0;
        while (newPopulation.size() < this.populationSize) {
            if (index < nbOfIndividualThatJumpToNextGeneration) {
                newPopulation.addIndividual(this.currentPopulation.getIndividualAtIndex(index));
            } else {
                List<T> children = this.getChildrenFromPrimeIndividuals(primeIndividuals);
                this.processMutation(children);
                    children.get(0).calculateFitness();
                    children.get(1).calculateFitness();
                    newPopulation.addGroupOfIndividual(children);
            }
            index++;
        }

        this.currentPopulation = newPopulation;
        this.sortPopulationByFitness();
        this.generationCount++;
    }

    private List<T> getChildrenFromPrimeIndividuals(List<T> primeIndividuals) {
        int firstParentIndex = this.random.nextInt(primeIndividuals.size());
        int secondParentIndex = this.random.nextInt(primeIndividuals.size());

        while (primeIndividuals.get(firstParentIndex) == primeIndividuals.get(secondParentIndex)) {
            secondParentIndex = this.random.nextInt(primeIndividuals.size());
        }

        return primeIndividuals.get(firstParentIndex).crossover(primeIndividuals.get(secondParentIndex));
    }

    private void processMutation(List<T> children) {
        for (int i = 0; i < children.size(); i++) {
            if (this.random.nextDouble() < this.mutationRatio) {
                children.get(i).mutate();
            }
        }
    }

    private List<T> selectPrimeIndividuals() {
        int selectionCapacity = (int) (this.populationSize * this.selectionRatio);
        List<T> primeIndividuals = new ArrayList<>();

        while (primeIndividuals.size() < selectionCapacity) {

            double chance = this.random.nextDouble();

            if (chance < this.rankSpaceRatio) {
                primeIndividuals.add(this.currentPopulation.getIndividualAtIndex(0));
            } else {
                double limit = this.rankSpaceRatio;
                for (int i = 1; i < this.populationSize; i++) {

                    double probability = Math.pow((1 - this.rankSpaceRatio), i) * this.rankSpaceRatio;
                    if (i == this.currentPopulation.size() - 1) {
                        probability = Math.pow((1 - this.rankSpaceRatio), i);
                    }

                    if (chance > limit && chance < limit + probability) {
                        primeIndividuals.add(this.currentPopulation.getIndividualAtIndex(i));
                    }
                    limit = limit + probability;
                }
            }
        }
        return primeIndividuals;
    }

    private void calculateFitness() {
        for (int i = 0; i < this.populationSize; i++) {
            this.currentPopulation.getIndividualAtIndex(i).calculateFitness();
        }
    }

    private void sortPopulationByFitness() {
        this.currentPopulation.sortPopulationByFitness();
    }

    private void assertPopulationSize(int populationSize)throws IllegalArgumentException {
        if(populationSize < 2 || populationSize > 10_000){
            throw new IllegalArgumentException("The population size must be superior to 2 individuals and inferior to 10 000");
        }
    }

    private void assertRankSpaceRation(double rankSpaceRatio)throws IllegalArgumentException {
        if(rankSpaceRatio < 0.01 && rankSpaceRatio > 0.99){
            throw new IllegalArgumentException("The rank space ratio must be between 0.01 and 0.99");
        }
    }

    private void assertMutationRation(double rankSpaceRatio)throws IllegalArgumentException {
        if(rankSpaceRatio < 0.01 && rankSpaceRatio > 0.99){
            throw new IllegalArgumentException("The mutation ratio must be between 0.01 and 0.99");
        }
    }

    private void assertSelectionRation(double rankSpaceRatio)throws IllegalArgumentException {
        if(rankSpaceRatio < 0.01 && rankSpaceRatio > 0.99){
            throw new IllegalArgumentException("The selection ratio must be between 0.01 and 0.99");
        }
    }

    private void assertUniformityRation(double rankSpaceRatio)throws IllegalArgumentException {
        if(rankSpaceRatio < 0.01 && rankSpaceRatio > 0.99){
            throw new IllegalArgumentException("The uniformity ratio must be between 0.01 and 0.99");
        }
    }
}
