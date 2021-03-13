#include "frontend/SuperPointExtractor.h"

#include <opencv2/opencv.hpp>

#include "Settings.h"

namespace ldso {
    SuperPointExtractor::SuperPointExtractor(){
        model = std::shared_ptr<SuperPointNet>(new SuperPointNet());
        torch::load(model, setting_superPointModelPath);
    }

    void SuperPointExtractor::DetectAndDescribe(int nFeatures, cv::Mat img, shared_ptr<Frame> frame,
                                                std::vector<shared_ptr<SuperPoint>>& features){
        if (img.cols * img.rows != 0){
            cv::Mat imgCopy = img.clone();
            torch::Tensor imgT = torch::from_blob(imgCopy.data, {1, 1, imgCopy.rows, imgCopy.cols}, torch::kByte);
            torch::Tensor x = imgT.clone();
            x = x.to(torch::kFloat) / 255; // convert image entries to [0, 1] range
            x = x.set_requires_grad(false);

            auto result = model->forward(x);

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
            auto conf_kpts = torch::nonzero(kpts > conf_threshold); // [N, 2]

            torch::Tensor grid = torch::zeros({1, 1, conf_kpts.size(0), 2});
            grid[0][0].slice(1, 0, 1) = conf_kpts.slice(1, 1, 2) / (kpts.size(1) / 2.0) - 1;
            grid[0][0].slice(1, 1, 2) = conf_kpts.slice(1, 0, 1) / (kpts.size(0) / 2.0) - 1;

            desc = torch::nn::functional::grid_sample(desc, grid,
                    torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear)
                                                                .padding_mode(torch::kZeros)
                                                                .align_corners(false)); // [1, 256, 1, N]
            desc = desc.squeeze(0).squeeze(1);

            auto desc_norm = torch::norm(desc, 2, 1);
            desc = desc.div(torch::unsqueeze(desc_norm, 1));

            desc = desc.transpose(0, 1).contiguous();  // [N, 256]

            features.clear();
            vector<shared_ptr<SuperPoint>> new_features;
            new_features.reserve(conf_kpts.size(0));

            for (auto i = 0; i < conf_kpts.size(0); i++){
                int y = conf_kpts[i][0].item<int>();
                int x = conf_kpts[i][1].item<int>();
                shared_ptr<SuperPoint> feat(new SuperPoint(x, y, frame));
                for (auto j = 0; j < 256; j++){
                    feat->descriptor[j] = desc[i][j].item<float>();
                }
                feat->isCorner = true;
                feat->score = kpts[y][x].item<float>();
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
        //cv::cvtColor(image, imgCopy, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < features.size(); i++){
            cv::circle(imgCopy, cv::Point(features[i]->uv[0], features[i]->uv[1]), 2, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("Detected features", imgCopy);
        cv::waitKey(3);
    }
}