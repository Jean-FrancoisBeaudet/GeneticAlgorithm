package GenericGeneticAlgorithm;

public interface IgenerateRandomIndividual<T extends Individual<T>> {
    T generateRandomIndividual();
}
