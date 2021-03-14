#include "frontend/SuperPointExtractor.h"

#include <opencv2/opencv.hpp>

#include <chrono>

#include "Settings.h"

namespace ldso {
    SuperPointExtractor::SuperPointExtractor():model(new SuperPointNet()), device(torch::kCPU){
        torch::set_num_interop_threads(torch::get_num_threads());
        torch::load(model, setting_superPointModelPath);
        model->to(device);
    }

    void SuperPointExtractor::DetectAndDescribe(int nFeatures, cv::Mat img, shared_ptr<Frame> frame,
                                                std::vector<shared_ptr<SuperPoint>>& features){
        if (img.cols * img.rows != 0){
            cv::Mat imgCopy = img.clone();
            auto t_start = chrono::high_resolution_clock::now();
            torch::Tensor imgT = torch::from_blob(imgCopy.data, {1, 1, imgCopy.rows, imgCopy.cols}, torch::kByte);
            torch::Tensor x = imgT.clone();
            x = x.to(torch::kFloat) / 255; // convert image entries to [0, 1] range
            x = x.set_requires_grad(false);
            auto t_end = chrono::high_resolution_clock::now();
            cout << "SP Image Setup time = " << double(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count()) / 1e6 << " ms" << endl;

            t_start = chrono::high_resolution_clock::now();
            auto result = model->forward(x.to(device));
            t_end = chrono::high_resolution_clock::now();
            cout << "SP Forward Pass time = " << double(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count()) / 1e6 << " ms" << endl;

            torch::Tensor kpts = result.first;
            torch::Tensor desc = result.second;

            /*float maxScore = -1.f;
            for (auto xx = 0; xx < kpts.size(1); xx++){
                for (auto yy= 0; yy < kpts.size(0); yy++){
                    maxScore = max(maxScore, kpts[yy][xx].item<float>());
                }
            }

            LOG(INFO) << "MAXSCORE = " << maxScore;*/

            // interpolate descriptor map at keypoint locations
            auto conf_kpts = torch::nonzero(kpts > conf_threshold).to(torch::kInt); // [N, 2]
            t_start = chrono::high_resolution_clock::now();
            torch::Tensor grid = torch::zeros({1, 1, conf_kpts.size(0), 2});
            grid[0][0].slice(1, 0, 1) = conf_kpts.slice(1, 1, 2) / (kpts.size(1) / 2.0) - 1;
            grid[0][0].slice(1, 1, 2) = conf_kpts.slice(1, 0, 1) / (kpts.size(0) / 2.0) - 1;

            desc = torch::nn::functional::grid_sample(desc, grid.to(device),
                    torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear)
                                                                .padding_mode(torch::kZeros)
                                                                .align_corners(false)); // [1, 256, 1, N]
            desc = desc.squeeze(0).squeeze(1);

            auto desc_norm = torch::norm(desc, 2, 1);
            desc = desc.div(torch::unsqueeze(desc_norm, 1));

            desc = desc.transpose(0, 1).contiguous();  // [N, 256]

            t_end = chrono::high_resolution_clock::now();
            cout << "SP Grid Sampling time = " << double(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count()) / 1e6 << " ms" << endl;

            features.clear();
            vector<shared_ptr<SuperPoint>> new_features;
            new_features.reserve(conf_kpts.size(0));

            // conf_kpts = conf_kpts.to(torch::kCPU);
            // desc = desc.to(torch::kCPU);
            // kpts = kpts.to(torch::kCPU);

            int* conf_kpts_arr = conf_kpts.data_ptr<int>();
            float* desc_arr = desc.data_ptr<float>();
            float* kpts_arr = kpts.data_ptr<float>();
            for (auto i = 0; i < conf_kpts.size(0); i++){
                //int y1 = conf_kpts[i][0].item<int>();
                //int x1 = conf_kpts[i][1].item<int>();
                int yy = *(conf_kpts_arr++);
                int xx = *(conf_kpts_arr++);
                shared_ptr<SuperPoint> feat(new SuperPoint(xx, yy, frame));
                for (auto j = 0; j < 256; j++){
                    feat->descriptor[j] = (*desc_arr++); //desc[i][j].item<float>();
                }
                feat->isCorner = true;
                feat->score = *(kpts_arr + yy * kpts.size(1) + xx); //kpts[y][x].item<float>();
                feat->fType = Feature::FeatureType::SUPEPROINT;
                new_features.push_back(feat);
            }

            if (new_features.size() >= nFeatures){
                sort(new_features.begin(), new_features.end(), [](shared_ptr<SuperPoint> feat1, shared_ptr<SuperPoint> feat2) { return feat1->score > feat2->score; });
                for (auto i = 0 ; i < nFeatures; i++){
                    features.push_back(new_features[i]);
                }
            } else {
                features = new_features;
            }
        } else {
            LOG(WARNING) << "Empty image, skipping feature extraction!";
        }
    }

    void SuperPointExtractor::DrawFeatures(cv::Mat image, vector<shared_ptr<SuperPoint>> features){
        cv::Mat imgCopy = image.clone();
        cv::cvtColor(image, imgCopy, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < features.size(); i++){
            cv::circle(imgCopy, cv::Point(features[i]->uv[0], features[i]->uv[1]), 2, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("Detected features", imgCopy);
        cv::waitKey(3);
    }
}