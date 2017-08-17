#ifndef PAM_H
#define PAM_H

#include <random>
#include <cstring>

namespace parser
{
/*
 * pam.cpp - Partitioning Around Medoids
 *
 * Clusters a data set using the partitioning around medoids algorithm. 
 *
 * The calling syntax is:
 *
 *		center_indices = pam(distance_matrix, K)
 * where 
 *      distance_matrix is an NxN matrix if pairwise distances between all 
 *      data points
 *      K is the number of clusters. 
 *
 * Depending on your compiler, you can compile the function using
 * one of the following calls:
 * $ mex CXXFLAGS='$CXXFLAGS -std=c++0x' COPTIMFLAGS='-O3 -DNDEBUG'  -largeArrayDims pam.cpp
 * or
 * $ mex CXXFLAGS='$CXXFLAGS -std=c++11' COPTIMFLAGS='-O3 -DNDEBUG'  -largeArrayDims pam.cpp
 *
 * Author: Tobias Pohlen <tobias.pohlen@rwth-aachen.de>
 *
 * References:
 */

    std::random_device rd;
    std::mt19937 g(rd());

    inline double get_distance(size_t n, size_t m, size_t N, const double* distances)
    {
        return distances[n*N+m];
    }


    void sample_without_replacement(size_t K, size_t N, size_t* result)
    {
        // The algorithm works as follows:
        // We create a sequence of random cycles and then compute the image of
        // the identity permutation when applying the cycles. 
        size_t* cycles = new size_t[K];
        std::uniform_int_distribution<size_t> index_dist(0, N-1);

        // Set up the cycles
        for (size_t k = 0; k < K; k++)
        {
            cycles[k] = index_dist(g);
        }

        // Compute the result array
        for (size_t k = 0; k < K; k++)
        {
            // Compute the image of k under the previously computed permutation
            size_t image = k;
            for (size_t i = 0; i < K; i++)
            {
                // The cycles are of the form (i cycles[i])
                if (i == image)
                {
                    image = cycles[i];
                }
                else if (image == cycles[i])
                {
                    image = i;
                }

                result[k] = image;
            }
        }

        delete[] cycles;
    }

    size_t get_closest_cluster_center(
        size_t n, 
        size_t K, 
        size_t N, 
        const double* distances, 
        const size_t* centers)
    {
        size_t closest = centers[0];
        double min_dist = get_distance(n, closest, N, distances);
        for (size_t k = 1; k < K; k++)
        {
            const double dist = get_distance(n, centers[k], N, distances);
            if (dist < min_dist)
            {
                closest = centers[k];
                min_dist = dist;
            }
        }
        return closest;
    }

    double compute_cluster_assignments(
        size_t K, 
        size_t N, 
        const double* distances, 
        const size_t* centers, 
        size_t* closest)
    {
        double result = 0.0;

        // For each data point, we have to compute the closest cluster center
        for (size_t n = 0; n < N; n++)
        {
            // Get the closest cluster center
            closest[n] = get_closest_cluster_center(n, K, N, distances, centers);

            // Compute the total objective
            result += distances[n*N+closest[n]];
        }

        return result;
    }

    double compute_objective_swap(
        size_t K, 
        size_t N, 
        const double* distances, 
        size_t* centers, 
        size_t* cluster_assignments, 
        size_t old_cluster_center_index, 
        size_t new_cluster_center)
    {
        double objective = 0.0;
        const size_t old_cluster_center = centers[old_cluster_center_index];
        centers[old_cluster_center_index] = new_cluster_center;

        // Go through all data points to compute the objective and update 
        // the assignments
        for (size_t n = 0; n < N; n++)
        {
            // We know that right now cluster_assignments[n] is the closest 
            // cluster center for point n
            // We only swap once cluster center. So, the assignment only has
            // to be updated if either the old cluster center was the closest
            // or the new cluster center is closer than the old one
            if (cluster_assignments[n] == old_cluster_center)
            {
                // We have to go through all cluster centers to determine the 
                // closest one
                cluster_assignments[n] = get_closest_cluster_center(n, K, N, distances, centers);
            }
            else if (get_distance(n, cluster_assignments[n], N, distances) > get_distance(n, new_cluster_center, N, distances))
            {
                // This means that the new cluster center is now the closest one
                cluster_assignments[n] = new_cluster_center;
            }

            objective += distances[n*N+cluster_assignments[n]];
        }

        return objective;
    }

    void pam(
        size_t K, 
        size_t N, 
        const double* distances, 
        size_t* centers, 
        size_t* cluster_assignments, 
        double* objective)
    {
        // Pick the initial cluster centers by sampling K out of N data points
        // without replacement
        sample_without_replacement(K, N, centers);

        // Set up an array of cluster assignments for a speedy computation
        *objective = compute_cluster_assignments(K, N, distances, centers, cluster_assignments);

        // We use a working copy of the cluster assignments in order
        // to test whether a new assignment is better than an old one
        size_t* working_cluster_assignments = new size_t[N];
        std::memcpy(working_cluster_assignments, cluster_assignments, sizeof(size_t)*N);

        // While there is a way to reduce the objective, do it
        bool objective_updated;

        do {
            objective_updated = false;

            // Go through all clusters and check for each data point if it would
            // be a better fit for the cluster
            for (size_t k = 0; k < K; k++)
            {
                size_t old_cluster_center = centers[k];

                for (size_t n = 0; n < N; n++)
                {
                    // If this is already a cluster center, skip the assignment
                    bool is_cluster_center = false;
                    for (size_t p = 0; p < K; p++)
                    {
                        is_cluster_center = centers[p] == n;
                        if (is_cluster_center)
                        {
                            break;
                        }
                    }

                    if (is_cluster_center)
                    {
                        continue;
                    }

                    // Compute the swapped assignment
                    double new_objective = compute_objective_swap(K, N, distances, centers, working_cluster_assignments, k, n);

                    // If this is an improvement, then we can stick with the updated centers
                    // and assignments. Otherwise, we have to revert the effect
                    if (new_objective < *objective)
                    {
                        // It's better
                        *objective = new_objective;
                        std::memcpy(cluster_assignments, working_cluster_assignments, sizeof(size_t)*N);
                        objective_updated = true;
                        old_cluster_center = centers[k];
                    }
                    else
                    {
                        // It's not better
                        // Revert the effect
                        centers[k] = old_cluster_center;
                        std::memcpy(working_cluster_assignments, cluster_assignments, sizeof(size_t)*N);
                    }
                }
            }
        } while (objective_updated);

        // Get rid of the temporary arrays
        delete[] working_cluster_assignments;
    }
}
#endif