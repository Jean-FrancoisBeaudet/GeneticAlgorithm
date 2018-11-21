package GenericGeneticAlgorithm;

import java.util.*;

public class Population<Type extends Individual<Type>> implements Iterable<Type> {

    private List<Type> population;
    private int populationCapacity;
    private final Random random = new Random();

    public Population(int populationSize){
        this.populationCapacity = populationSize;
        this.population = new ArrayList<>(populationSize);
    }

    public void addIndividual(Type individual){
        if(this.population.size() < this.populationCapacity) {
            this.population.add(individual);
        }
    }

    public void addGroupOfIndividual(List<Type> individuals) {
        for (int i = 0; i < individuals.size(); i++) {
            if(this.population.size() < this.populationCapacity) {
                this.population.add(individuals.get(i));
            }
            else {
                return;
            }
        }
    }

    public void removeIndividual(Type individual){
        this.population.remove(individual);
    }

    public void removeIndividualAt(int index){
        this.population.remove(index);
    }

    public int size() {
        return this.population.size();
    }

    public Type getRandomIndividual() {
        return this.population.get(this.random.nextInt(this.population.size()));
    }

    public Type getIndividualAtIndex(int index) {
        return this.population.get(index);
    }

    public int getPopulationCapacity() { return this.populationCapacity; }

    public boolean contains(Type other){
        for(Type individual: this){
            if(individual.equals(other)){
                return true;
            }
        }
        return false;
    }

    public void sortPopulationByFitness() {
        Collections.sort(this.population, new Comparator<Type>() {
            @Override
            public int compare(Type firstIndividual, Type secondIndividual) {
                return Double.compare(secondIndividual.getFitnessScore(), firstIndividual.getFitnessScore());
            }
        });
    }

    @Override
    public Iterator<Type> iterator() {
        return this.population.iterator();
    }
}
