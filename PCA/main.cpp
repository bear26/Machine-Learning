#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage <filename to datasets1> <filename to datasets2> <filename to dataset3>" << std::endl;
        return -1;
    }

    for(int test = 1; test <= 3; ++test)
    {
        std::vector<double> data;

        std::fstream stream(argv[test]);
        std::string line;

        int count = 0;

        while(std::getline(stream, line))
        {
            std::vector<std::string> tokens;

            boost::trim(line);
            boost::split(tokens, line, boost::is_any_of(" "), boost::token_compress_on);

            std::vector<double> features;

            for(const auto &str : tokens)
            {
                if (!str.empty())
                {
                    features.push_back(std::stod(str));
                }
            }

            data.insert(data.end(), features.begin(), features.end());
            ++count;
        }

        cv::Mat_<double> data_cv(count, data.size() / count, data.data());

        cv::PCA pca(data_cv, cv::Mat(), CV_PCA_DATA_AS_ROW);

        double sum = cv::sum(pca.eigenvalues)[0];
        double curr_sum = 0;
        int count_components = 0;

        for(int i = 0; i < pca.eigenvalues.rows; ++i)
        {
            curr_sum += pca.eigenvalues.at<double>(i, 0);

            if (curr_sum >= sum * 0.99)
            {
                count_components = i + 1;
                break;
            }
        }

        count_components = std::max(count_components, 1);

        std::cout << count_components << std::endl;

        cv::Mat_<double> project = pca.project(data_cv).colRange(0, count_components);

        std::fstream stream_out(std::string(argv[test]) + ".pca", std::ios_base::out);

        for(int i = 0; i < project.rows; ++i)
        {
            for(int j = 0; j < project.cols; ++j)
            {
                stream_out << project(i, j) << " ";
            }

            stream_out << std::endl;
        }
    }

    return 0;
}

