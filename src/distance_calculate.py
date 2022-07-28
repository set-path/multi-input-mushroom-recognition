from sklearn.metrics.pairwise import cosine_distances,euclidean_distances, manhattan_distances, pairwise_distances,cosine_similarity

def distance_calculate(pred_genetic_dis, true_genetic_dis, option):
    """
    Calculate the distance between predicted and true genetic distance
    :param pred_genetic_dis: predicted genetic distance
    :param true_genetic_dis: true genetic distance
    :return: distance between predicted and true genetic distance
    """
    if option == 'cosine':
        return cosine_similarity(pred_genetic_dis, true_genetic_dis)
    elif option == 'euclidean':
        return euclidean_distances(pred_genetic_dis, true_genetic_dis)