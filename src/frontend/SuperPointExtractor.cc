#include "frontend/SuperPointExtractor.h"

#include <opencv2/opencv.hpp>

#include "Settings.h"

namespace ldso {
    SuperPointExtractor::SuperPointExtractor(){
        model = std::shared_ptr<SuperPointNet>(new SuperPointNet());
        torch::load(model, setting_superPointModelPath);
    }

    void SuperPointExtractor::detectAndDescribe(int nFeatures, cv::Mat img, shared_ptr<Frame> frame, std::vector<shared_ptr<SuperPoint>>& features){
        if (img.cols * img.rows != 0){
            torch::Tensor x = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);
            x = x.to(torch::kFloat) / 255; // convert image entries to [0, 1] range
            x = x.set_requires_grad(false);

            auto result = model->forward(x);

            torch::Tensor kpts = result.first;
            torch::Tensor desc = result.second;

            float maxScore = -1.f;
            for (auto x = 0; x < kpts.size(1); x++){
                for (auto y = 0; y < kpts.size(0); y++){
                    maxScore = max(maxScore, kpts[y][x].item<float>());
                }
            }

            LOG(INFO) << "MAX score = " << maxScore;

            // interpolate descriptor map at keypoint locations
            auto conf_kpts = torch::nonzero(kpts > conf_threshold); // [N, 2]
            LOG(INFO) << conf_kpts.size(0) << " " << conf_kpts.size(1);

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
            for (auto i = 0; i < conf_kpts.size(0); i++){
                int y = conf_kpts[i][0].item<int>();
                int x = conf_kpts[i][1].item<int>();
                shared_ptr<SuperPoint> feat(new SuperPoint(x, y, frame));
                for (auto j = 0; j < 256; j++){
                    feat->descriptor[j] = desc[i][j].item<float>();
                }
                feat->isCorner = true;
                feat->score = kpts[y][x].item<float>();

                features.push_back(feat);
            }
        } else {
            LOG(WARNING) << "Empty image, skipping feature extraction!";
        }
    }
}