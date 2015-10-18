#include "model.h"

#include <algorithm>
#include <iostream>
#include <cmath>

const int count_class = 2;

double kernel(const std::vector<double> &f1, const std::vector<double> &f2)
{
    const double sigma = 0.6;

    double sum = 0;

    for(size_t i = 0; i < f1.size(); ++i)
    {
        sum += std::pow(f1[i] - f2[i], 2);
    }

    return std::exp(-sum * sigma);
}

Model::Model()
{

}

void Model::train(const Data &data, double c)
{
    c_ = c;
    sv_ = data;
    w0_ = 0;
    lambda_.resize(data.size(), 0);
    error_cache_.resize(data.size(), 0);

    while(true)
    {
        bool changed = false;

        for(size_t i = 0; i < sv_.size(); ++i)
        {
            changed |= processing_(i);
        }

        if (changed == false)
        {
            break;
        }

        changed = false;

        for(size_t i = 0; i < sv_.size(); ++i)
        {
            if (lambda_[i] > 0 && lambda_[i] < c_)
            {
                changed |= processing_(i);
            }
        }

        if (changed == false)
        {
            break;
        }
    }
}

int Model::predict(const Object &object) const
{
    return (u_(object.features()) < 0) ? -1 : 1;
}

std::vector<std::pair<int, int> > Model::test(const Data &data) const
{
    std::vector<std::pair<int, int>> ans(data.size());

    for(size_t i = 0; i < data.size(); ++i)
    {
        ans[i] = (std::make_pair(data[i].label(), predict(data[i])));
    }

    return ans;
}

double Model::u_(const std::vector<double> &features) const
{
    double ans = -w0_;

    for(size_t i = 0; i < sv_.size(); ++i)
    {
        ans += lambda_[i] * sv_[i].label() * kernel(sv_[i].features(), features);
    }

    return ans;
}

bool Model::processing_(size_t j)
{
    double e = (lambda_[j] > 0 && lambda_[j] < c_) ? error_cache_[j] : (u_(sv_[j].features()) - sv_[j].label());

    double r = e * sv_[j].label();
    double tau = 1e-3;

    if ((r < -tau && lambda_[j] < c_) || (r > tau && lambda_[j] > 0))
    {
        return processing2_(j, e);
    }

    return false;
}

bool Model::processing2_(size_t j, double e_j)
{
    int best_i = -1;

    for(size_t i = 0; i < sv_.size(); ++i)
    {
        if (lambda_[i] > 0 && lambda_[i] < c_)
        {
            if (best_i == -1)
            {
                best_i = i;
            }

            if (std::fabs(e_j - error_cache_[i]) > std::fabs(e_j - error_cache_[best_i]))
            {
                best_i = i;
            }
        }
    }

    if (best_i != -1)
    {
        if(optimization_(best_i, j))
        {
            return true;
        }

        shuffle_();

        for(size_t i = 0; i < sv_.size(); ++i)
        {
            if (optimization_(i, j))
            {
                return true;
            }
        }
    }

    shuffle_();

    for(size_t i = 0; i < sv_.size(); ++i)
    {
        if (optimization_(i, j))
        {
            return true;
        }
    }

    return false;
}

bool Model::optimization_(size_t i, size_t j)
{
    if (i == j)
    {
        return false;
    }

    double l;
    double h;

    if (sv_[j].label() == sv_[i].label())
    {
        double gamma = lambda_[i] + lambda_[j];

        if (gamma > c_)
        {
            l = gamma - c_;
            h = c_;
        }
        else
        {
            l = 0;
            h = gamma;
        }
    }
    else
    {
        double gamma = lambda_[i] - lambda_[j];

        if (gamma > 0)
        {
            l = 0;
            h = c_ - gamma;
        }
        else
        {
            l = -gamma;
            h = c_;
        }
    }

    if (l == h)
    {
        return false;
    }

    double e_j = (lambda_[j] > 0 && lambda_[j] < c_) ? error_cache_[j] : u_(sv_[j].features()) - sv_[j].label();
    double e_i = (lambda_[i] > 0 && lambda_[i] < c_) ? error_cache_[i] : u_(sv_[i].features()) - sv_[i].label();

    double eta = 2 * kernel(sv_[i].features(), sv_[j].features()) - kernel(sv_[i].features(), sv_[i].features()) - kernel(sv_[j].features(), sv_[j].features());

    double lambda1;
    double lambda2;

    double eps = 1e-3;

    if (eta < 0)
    {
        lambda1 = lambda_[j] - sv_[j].label() * (e_i - e_j) / eta;
        if (lambda1 >= h)
        {
            lambda2 = h;
        }
        else
        {
            lambda2 = (lambda1 <= l) ? l : lambda1;
        }
    }
    else
    {
        double c1 = eta / 2;
        double c2 = sv_[j].label() * (e_i - e_j) - eta * lambda_[j];

        double l1 = c1 * l * l + c2 * l;
        double h1 = c1 * h * h + c2 * h;

        if (l1 > h1 + eps)
        {
            lambda2 = l;
        }
        else
        {
            lambda2 = (l1 < h1 - eps) ? h : lambda_[j];
        }
    }

    if (lambda2 < 1e-8)
    {
        lambda2 = 0;
    }
    else
    {
        if (lambda2 > c_ - 1e-8)
        {
            lambda2 = c_;
        }
    }

    if (std::fabs(lambda2 - lambda_[j]) < eps * (lambda2 + lambda_[j] + eps))
    {
        return false;
    }

    double s = sv_[i].label() * sv_[j].label();

    lambda1 = lambda_[i] + s * (lambda_[j] - lambda2);

    double lambda_i2;
    double lambda_j2;

    if (lambda1 < 0)
    {
        lambda_j2 = s * lambda1 + lambda2;
        lambda_i2 = 0;
    }
    else
    {
        if (lambda1 > c_)
        {
            lambda_j2 = lambda2 + s * (lambda1 - c_);
            lambda_i2 = c_;
        }
        else
        {
            lambda_i2 = lambda1;
            lambda_j2 = lambda2;
        }
    }

    double b1 = w0_ + e_i + sv_[i].label() * (lambda_i2 - lambda_[i]) * kernel(sv_[i].features(), sv_[i].features())
            + sv_[j].label() * (lambda_j2 - lambda_[j]) * kernel(sv_[i].features(), sv_[j].features());

    double b2 = w0_ + e_j + sv_[i].label() * (lambda_i2 - lambda_[i]) * kernel(sv_[i].features(), sv_[j].features())
            + sv_[j].label() * (lambda_j2 - lambda_[j]) * kernel(sv_[j].features(), sv_[j].features());

    double b3 = (b1 + b2) / 2;

    double w0_2;

    if (lambda_i2 > 0 && lambda_i2 < c_)
    {
        w0_2 = b1;
    }
    else
    {
        if (lambda_j2 > 0 && lambda_j2 < c_)
        {
            w0_2 = b2;
        }
        else
        {
            w0_2 = b3;
        }
    }

    double delta_w0 = w0_2 - w0_;

    // update_error_cache
    {
        double t_j = sv_[j].label() * (lambda_j2 - lambda_[j]);
        double t_i = sv_[i].label() * (lambda_i2 - lambda_[i]);

        for(size_t k = 0; k < sv_.size(); ++k)
        {
            if (lambda_[k] > 0 && lambda_[k] < c_)
            {
                error_cache_[k] += t_j * kernel(sv_[j].features(), sv_[k].features()) + t_i * kernel(sv_[i].features(), sv_[k].features()) - delta_w0;
            }
        }

        error_cache_[i] = 0;
        error_cache_[j] = 0;
    }

    w0_ = w0_2;
    lambda_[j] = lambda_j2;
    lambda_[i] = lambda_i2;

    return true;
}

void Model::shuffle_()
{
    std::vector<int> a(sv_.size());

    for(size_t i = 0; i < a.size(); ++i)
    {
        a[i] = i;
    }

    std::random_shuffle(a.begin(), a.end());

    Data sv;
    std::vector<double> lambda;
    std::vector<double> error_cache;

    for(auto x : a)
    {
        sv.add(sv_[x]);
        lambda.push_back(lambda_[x]);
        error_cache.push_back(error_cache_[x]);
    }

    sv_ = sv;
    lambda_ = lambda;
    error_cache_ = error_cache;
}

double cross_validation(const Data &data, int folder, int t)
{
    std::cout << "Cross validation..." << std::endl;

    double ans = 0;
    for(int i = 0; i < t; ++i)
    {
        std::vector<Data> data_split;
        data.split(folder, data_split);

        std::vector<std::pair<int, int>> result;

        for(size_t i = 0; i < data_split.size(); ++i)
        {
            Data data_train;

            for(size_t j = 0; j < data_split.size(); ++j)
            {
                if (j != i)
                {
                    data_train.add(data_split[j]);
                }
            }

            Model model;
            model.train(data_train, 10);

            auto curr_result = model.test(data_split[i]);
            result.insert(result.end(), curr_result.begin(), curr_result.end());
        }

        ans += print_result(result);
    }

   return ans / t;
}

double print_result(const std::vector<std::pair<int, int> > &result)
{
    std::vector<std::vector<int>> conf_matrix(count_class, std::vector<int>(count_class, 0));

    for(auto pair : result)
    {
        conf_matrix[std::max(0, pair.first)][std::max(0, pair.second)]++;
    }

    for(size_t i = 0; i < conf_matrix.size(); ++i)
    {
        double presision = 0;
        double recall = 0;

        for(size_t j = 0; j < conf_matrix[i].size(); ++j)
        {
            presision += conf_matrix[i][j];
        }

        for(size_t j = 0; j < conf_matrix.size(); ++j)
        {
            recall += conf_matrix[j][i];
        }

        presision = conf_matrix[i][i] / presision;
        recall = conf_matrix[i][i] / recall;

        printf("Fscore %d class: %lf\n", (int)i, 2 * presision * recall / (presision + recall));
    }

    int correct = 0;
    for(size_t i = 0; i < conf_matrix.size(); ++i)
    {
        correct += conf_matrix[i][i];
    }


    printf("Accuracy: %lf%%(%d/%d)\n", 100.0 * correct / result.size(), correct, (int)result.size());

    return 100.0 * correct / result.size();
}
