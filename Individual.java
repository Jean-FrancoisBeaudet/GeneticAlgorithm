package GenericGeneticAlgorithm;

import java.util.List;

public abstract class Individual<Type extends Individual<Type>> {

    private double fitnessScore;

    protected abstract List<Type> crossover(Type otherParent);

    protected abstract void mutate();

    protected abstract void calculateFitness();

    public double getFitnessScore(){
        return this.fitnessScore;
    }

    protected void setFitnessScore(double score){
        this.fitnessScore = score;
    }
}
