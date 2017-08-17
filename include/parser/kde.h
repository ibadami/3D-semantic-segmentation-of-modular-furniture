#ifndef PARSER_KDE_H
#define PARSER_KDE_H

#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

/**
 * This file contains a simple implementation of a kernel denstiy
 * estimator for Gaussian kernels.
 */

namespace parser {
    
    
    /**
     * Abstract class for all parameter models
     */
    class AbstractParameterModel {
    public:
        /**
         * Saves the model using the open CV cv::FileStorage object. 
         * The model is stored with the given prefix. 
         * 
         * @param _fs The file storage object
         * @param _prefix the prefix under which the model shall be saved in the fs
         */
        virtual void save(cv::FileStorage & _fs, const std::string & _prefix = "") = 0;
        
        /**
         * Loads the model from an open cv file storage object from a specific 
         * prefix. 
         * 
         * @param _fs The file storage object
         * @param _prefix the prefix under which the model shall be saved in the fs
         */
        virtual void load(const cv::FileStorage & _fs, const std::string & _prefix = "") = 0;
        
        /**
         * Saves the model to a file
         * 
         * @param _filename The filename
         */
        virtual void saveToFile(const std::string & _filename)
        {
            // Open the file storage
            cv::FileStorage fs(_filename, cv::FileStorage::WRITE);

            // Save everything
            save(fs);
            
            // Free the storage
            fs.release();
        }
        
        /**
         * Loads the model from a file. 
         * 
         * @param _filename
         */
        virtual void loadFromFile(const std::string & _filename)
        {
            // Open the file storage
            cv::FileStorage fs(_filename, cv::FileStorage::READ);
            
            // Load everything
            load(fs);
            
            // Free the storage
            fs.release();
        }
    };

    /**
     * This is the parent class for all distributions
     */
    template <typename T=float, typename O=float, int D=2>
    class AbstractDistribution {
    public:
        /**
         * All data has to be given in this vector format
         */
        typedef cv::Vec<T,D> Vec;
        /**
         * Pointer type for these objects
         */
        typedef std::shared_ptr<AbstractDistribution> ptr;
        
        /**
         * The model class for this distribution
         */
        class Model : public AbstractParameterModel {
        public:
            /**
             * The function estimates the parameters based on a selection of 
             * samples. 
             */
            virtual void estimate(const std::vector<Vec> & _samples) = 0;
        };
        
        /**
         * Returns the probability of a point under this distribution
         * 
         * @param _x the query point
         * @return The probability
         */
        virtual O eval(const Vec & _x) const = 0;
        
        /**
         * Evaluates the pdf at _x
         * 
         * @param _x the query point
         * @return The value of the PDE
         */
        virtual O eval1D(T x) const
        {
            cv::Vec<T, D> v;
            v[0] = x;
            return eval(v);
        }
    };
    
    /**
     * This class represents a distribution by performing kernel density 
     * estimation. The Gaussian kernel is used and cannot be replaced.
     */
    template <typename T=float, typename O=float, int D=2>
    class KernelDistribution : public AbstractDistribution<T,O,D> {
    public:
        typedef typename AbstractDistribution<T, O, D>::Vec Vec;
        typedef std::shared_ptr<KernelDistribution> ptr;
        
        /**
         * The parameter model for this class
         */
        class Model : public  AbstractDistribution<T,O,D>::Model {
        public:
            /**
             * Saves the model using the open CV cv::FileStorage object. 
             * The model is stored with the given prefix. 
             * 
             * @param _fs The file storage object
             * @param _prefix the prefix under which the model shall be saved in the fs
             */
            virtual void save(cv::FileStorage & _fs, const std::string & _prefix = "") 
            {
                _fs << _prefix + "h" << h;
                _fs << _prefix + "points" << cv::Mat(points);
            }

            /**
             * Loads the model from an open cv file storage object from a specific 
             * prefix. 
             * 
             * @param _fs The file storage object
             * @param _prefix the prefix under which the model shall be saved in the fs
             */
            virtual void load(const cv::FileStorage & _fs, const std::string & _prefix = "")
            {
                cv::Mat _points;
                _fs[_prefix + "h"] >> h;
                _fs[_prefix + "points"] >> _points;
                for (int i = 0; i < _points.rows; i++)
                {
                    points.push_back(_points.at<Vec>(i,0));
                }
            }
            

            /**
             * The function estimates the parameters based on a selection of 
             * samples. 
             */
            virtual void estimate(const std::vector<Vec> & _samples)
            {
                // Keep all the points
                points = _samples;
                
                // Perform simple bandwidth selection after silverman (Rule of
                // thumb)
                
                h = cv::Mat::zeros(D,1, CV_32FC1);
                float N = _samples.size();
                
                if (_samples.size() > 0)
                {
                    // Calculate the standard deviation
                    Vec mean;
                    Vec variance;
                    for (int i = 0; i < D; i++)
                    {
                        mean[i] = 0.;
                        variance[i] = 0.;
                    }
                    for (size_t i = 0; i < _samples.size(); i++)
                    {
                        mean += _samples[i];
                    }
                    for (int i = 0; i < D; i++)
                    {
                        mean[i] /= N;
                    }
                    
                    for (size_t i = 0; i < _samples.size(); i++)
                    {
                        for (int k = 0; k < D; k++)
                        {
                            const float temp = mean[k] - _samples[i][k];
                            variance[k] += temp*temp;
                        }
                    }
                    if (N > 1)
                    {
                        for (int i = 0; i < D; i++)
                        {
                            variance[i] /= (N - 1);
                        }
                    }
                    for (int i = 0; i < D; i++)
                    {
                        variance[i] = std::sqrt(variance[i]);
                        variance[i] = std::max(0.1f, variance[i]);
                    }
                    
                    for (int i = 0; i < D; i++)
                    {
                        h.at<float>(i,0) = variance[i] * std::pow(N, -1./(D+4.));
                    }
                }
                
                // Enforce some regularization
                for (int i = 0; i < D; i++)
                {
                    h.at<float>(i,0) = std::max(5e-2f, h.at<float>(i,0));
                }
            }
            
            /**
             * The bandwidth
             */
            cv::Mat h;
            /**
             * The list of points
             */
            std::vector<Vec> points;
        };
        
        KernelDistribution() {}
        
        /**
         * Default constructor
         * 
         * @param _points The list of points from which the distribution shall
         *                be estimated
         * @param _h The variance
         */
        explicit KernelDistribution(const Model & _model) : model(_model)
        {
            if (_model.points.size() > 0)
            {
                // Enforce some regularization
                for (int i = 0; i < D; i++)
                {
                    model.h.template at<float>(i,0) = std::max(1e-1f, model.h.template at<float>(i,0));
                }
                // Precompute the normalization constants
                normalizationConstant = 1./(_model.points.size());
                for (int i = 0; i < D; i++)
                {
                    normalizationConstant /= _model.h.template at<float>(i,0);
                }
                normalizationConstant /= (std::pow(2.*M_PI, D/2.));
                //normalize();
            }
            else
            {
                normalizationConstant = 0;
                expNormalize = 0;
            }
        }
        
        /**
         * Evaluates the pdf at _x
         * 
         * @param _x the query point
         * @return The value of the PDE
         */
        virtual O eval(const Vec & _x) const
        {
            O res = 0;
            // Evaluate the kernel at all points
            for (size_t i = 0; i < model.points.size(); i++)
            {
                res += evalKernel(_x, model.points[i]);
            }
            return res*normalizationConstant;
        }
        
        /**
         * Evaluates the kernel at two given points
         * 
         * @param _x
         * @param _y
         * @return Value of the kernel
         */
        O evalKernel(const Vec & _x, const Vec & _y) const 
        {
            // Calculate the squared distance
            O dist = 0;
            for (int i = 0; i < D; i++)
            {
                O temp = _x[i] - _y[i];
                temp /= model.h.template at<float>(i,0);
                dist += temp*temp;
            }
            return std::exp(dist * (-0.5));
        }
        
        /**
         * Calculates an empirical normalization constant such that the maximum
         * of the distribution is 1
         */
        void normalize()
        {
            // Calculate the maximum
            float maximum = 0;
            for (size_t i = 0; i < model.points.size(); i++)
            {
                const float val = eval(model.points[i]);
                if (val > maximum)
                {
                    maximum = val;
                }
            }
            if (maximum > 0)
            {
                normalizationConstant = normalizationConstant/maximum;
            }
        }
        
        /**
         * Normalizes the distribution to integrate to one
         */
        void normalizeMCMC()
        {
            normalizationConstant = 1;
            float integral = 0;
            int samples = 10000;
            std::default_random_engine rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<float> p(0,1);
            
            for (int i = 0; i < samples; i++)
            {
                Vec v;
                for (int d = 0; d < D; d++)
                {
                    v[d] = p(g);
                }
                
                integral += eval(v);
            }
            if (integral > 0)
            {
                normalizationConstant = samples/integral;
            }
            else
            {
                normalizationConstant = 1;
            }
        }
        
        /**
         * Visualizes the density estimator for 2d
         */
        void visualize() const
        {
            // Get the extrem-values
            float minX = 1e30, minY = 1e30, maxX = -1e30, maxY = -1e30;
            for (size_t d = 0; d < model.points.size(); d++)
            {
                minX = std::min(minX, model.points[d][0]);
                minY = std::min(minY, model.points[d][1]);
                maxX = std::max(maxX, model.points[d][0]);
                maxY = std::max(maxY, model.points[d][1]);
            }
            minX = 0;
            minY = 0;
            maxX = 1;
            maxY = 1;
            
            std::cout << "[X,Y] = meshgrid(" << minX << ":0.01:" << maxX << "," << minY << ":0.01:" << maxY <<");\n";
            std::cout  << "Z = [";
            for (float y = minY; y <= maxY; y += 0.01f)
            {
                for (float x = minX; x <= maxX; x += 0.01f)
                {
                    cv::Vec2d v;
                    v[0] = x; v[1] = y;
                    const float _z = eval(v);
                    std::cout << _z << ' ';
                }
                std::cout << '\n';
            }
            std::cout << "];\n";
            std::cout << "figure('name', 'KDE');\n";
            std::cout << "surf(X,Y,Z);\n";
        }
        
        
    public:
        /**
         * The global normalization constant
         */
        O normalizationConstant;
        /**
         * The exponent normalization constant.
         */
        O expNormalize;
        /**
         * The parameters
         */
        Model model;
    };
    
    /**
     * This class can be used in order to handle arbitrary conditional 
     * distributions. The distributions can  be conditioned on integer values
     */
    template <class TDist, class TModel, typename T=double, typename O=double, int D=2>
    class ConditionalDistribution {
    public:
        typedef typename AbstractDistribution<T, O, D>::Vec Vec;
        
        /**
         * The model for this class
         */
        class Model : public AbstractParameterModel {
        public:
            /**
             * Saves the model using the open CV cv::FileStorage object. 
             * The model is stored with the given prefix. 
             * 
             * @param _fs The file storage object
             * @param _prefix the prefix under which the model shall be saved in the fs
             */
            virtual void save(cv::FileStorage & _fs, const std::string & _prefix = "") 
            {
                std::string prefix(_prefix);
                if (prefix == "")
                {
                    prefix = "m";
                }
                std::vector<std::string> keys;
                getKeyMap(keys);
                
                _fs << prefix + "NumKeys" << static_cast<int>(keys.size());
                for (size_t i = 0; i < keys.size(); i++)
                {
                    std::stringstream ss;
                    ss << prefix << "Key" << i;
                    _fs << ss.str() << keys[i];
                    
                }
                // Save all models
                
                for (typename std::map<std::string, TModel>::iterator itr = models.begin(); itr != models.end(); ++itr)
                {
                    itr->second.save(_fs, prefix + itr->first);
                }
            }

            /**
             * Loads the model from an open cv file storage object from a specific 
             * prefix. 
             * 
             * @param _fs The file storage object
             * @param _prefix the prefix under which the model shall be saved in the fs
             */
            virtual void load(const cv::FileStorage & _fs, const std::string & _prefix = "")
            {
                std::string prefix(_prefix);
                if (prefix == "")
                {
                    prefix = "m";
                }
                // Load the key map
                std::vector<std::string> keys;
                int numKeys;
                _fs[prefix + "NumKeys"] >> numKeys;
                
                for (int i = 0; i < numKeys; i++)
                {
                    std::string key;
                    std::stringstream ss;
                    ss << prefix << "Key" << i;
                    _fs[ss.str()] >> key;
                    keys.push_back(key);
                }
                
                // Attempt to load all model
                for (size_t i = 0; i < keys.size(); i++)
                {
                    TModel model;
                    model.load(_fs, prefix + keys[i]);
                    models[ keys[i] ] = model;
                }
            }
            
            /**
             * Estimates a conditional distribution
             */
            void estimate(const std::vector<Vec> & _samples, const std::vector<int> & _conditions)
            {
                // Estimate the model and store it under the hash
                TModel model;
                model.estimate(_samples);
                
                std::string hash;
                createHash(hash, _conditions);
                
                models[hash] = model;
            }

            /**
             * Returns a key map 
             */
            void getKeyMap(std::vector<std::string> & _keyMap)
            {
                for (typename std::map<std::string, TModel>::iterator itr = models.begin(); itr != models.end(); ++itr)
                {
                    _keyMap.push_back(itr->first);
                }
            }
            
            /**
             * The models for the different distributions
             */
            std::map<std::string, TModel> models;
        };
        
        /**
         * Default constructor
         */
        ConditionalDistribution(const Model & _model) : model(_model)
        {
            // Create the distributions
            for (typename std::map<std::string, TModel>::iterator itr = model.models.begin(); itr != model.models.end(); ++itr)
            {
                TDist d(itr->second);
                distributions[itr->first] = d;
            }
        }
        
        /**
         * Creates the hash from the given conditions
         */
        static void createHash(std::string & _result, const std::vector<int> & _conditions)
        {
            std::stringstream ss;

            for (size_t i = 0; i < _conditions.size(); i++)
            {
                ss << _conditions[i] << "_";
            }

            _result = ss.str();
        }
        
        /**
         * Returns the value of the query point under the given condition
         * 
         * @param _x the query point
         * @return The value of the PDE
         */
        virtual O eval(const Vec & _x, const std::vector<int> & _condition) const
        {
            // Get the hash
            std::string hash;
            createHash(hash, _condition);
            
            // Get the distribution and evaluate it
            const double res = distributions.at(hash).eval(_x);
            return res;
        }
        
        /**
         * Returns a conditional distribution
         */
        const TDist & getDistribution(const std::vector<int> & _condition) const
        {
            std::string hash;
            createHash(hash, _condition);
            return distributions.at(hash);
        }
        
        TDist & getDistribution(const std::vector<int> & _condition)
        {
            std::string hash;
            createHash(hash, _condition);
            return distributions.at(hash);
        }
        
    private:
        /**
         * The distributions
         */
        std::map<std::string, TDist> distributions;
        /**
         * The model for this distribution
         */
        Model model;
    };

    /**
     * A simple discrete probability table for a random variable in the range
     * [0, n]
     */
    template <typename T=unsigned long, typename O=double>
    class DiscreteTableDistribution : public AbstractDistribution<T, O, 1> {
    public:
        typedef typename AbstractDistribution<T, O, 1>::Vec Vec;
        typedef std::shared_ptr<DiscreteTableDistribution> ptr;
        
        /**
         * The parameter model class for this class
         */
        class Model : public AbstractDistribution<T,O,1>::Model {
        public:
            /**
             * Saves the model using the open CV cv::FileStorage object. 
             * The model is stored with the given prefix. 
             * 
             * @param _fs The file storage object
             * @param _prefix the prefix under which the model shall be saved in the fs
             */
            virtual void save(cv::FileStorage & _fs, const std::string & _prefix = "") 
            {
                _fs << _prefix + "table" << table;
            }

            /**
             * Loads the model from an open cv file storage object from a specific 
             * prefix. 
             * 
             * @param _fs The file storage object
             * @param _prefix the prefix under which the model shall be saved in the fs
             */
            virtual void load(const cv::FileStorage & _fs, const std::string & _prefix = "")
            {
                _fs[_prefix + "table"] >> table;
            }

            /**
             * The function estimates the parameters based on a selection of 
             * samples. 
             */
            virtual void estimate(const std::vector<Vec> & _samples)
            {
                // Get the maximum number in the samples
                T max = 0;
                for (size_t i = 0; i < _samples.size(); i++)
                {
                    max = std::max(max, _samples[i][0]);
                }
                
                // Create the table
                table = cv::Mat::zeros(1, max+1, CV_64FC1);
                
                for (size_t i = 0; i < _samples.size(); i++)
                {
                    table.at<double>(0,_samples[i][0]) += 1;
                }
                
                for (T i = 0; i <= max; i++)
                {
                    table.at<double>(0,i) /= _samples.size();
                }
            }
            
            /**
             * The probability table
             */
            cv::Mat table;
        };
        
        /**
         * Default constructor
         */
        explicit DiscreteTableDistribution (const Model & _model) : model(_model) {}
        DiscreteTableDistribution() {}
        
        /**
         * Evaluates the pdf at _x
         * 
         * @param _x the query point
         * @return The value of the PDE
         */
        virtual O eval(const Vec & _x) const
        {
            const int bin = _x[0];
            cv::Mat table(model.table);
            const double res = table.at<double>(0, bin);
            return res;
        }
        
    private:
        Model model;
    };
    
}

#endif

