#pragma once
#ifndef LDSO_SUPERPOINTEXTRACTOR_H_
#define LDSO_SUPERPOINTEXTRACTOR_H_

#include <utility>
#include "torch/torch.h"

#include "Feature.h"
#include "Frame.h"

namespace ldso {
    struct SuperPointNet : torch::nn::Module {
        SuperPointNet():
            relu(torch::nn::ReLUOptions().inplace(true)),
            pool(torch::nn::MaxPool2dOptions(2).stride(2)),
            conv1a(torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1)),
            conv1b(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            conv2a(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            conv2b(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            conv3a(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            conv3b(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            conv4a(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            conv4b(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            convPa(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            convPb(torch::nn::Conv2dOptions(256, 65, 1).stride(1).padding(0)),
            convDa(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            convDb(torch::nn::Conv2dOptions(256, 256, 1).stride(1).padding(0))
        {
            register_module("conv1a", conv1a);
            register_module("conv1b", conv1b);
            register_module("conv2a", conv2a);
            register_module("conv2b", conv2b);
            register_module("conv3a", conv3a);
            register_module("conv3b", conv3b);
            register_module("conv4a", conv4a);
            register_module("conv4b", conv4b);
            register_module("convPa", convPa);
            register_module("convPb", convPb);
            register_module("convDa", convDa);
            register_module("convDb", convDb);
        }

        std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
            // shared encoder
            x = relu(conv1a(x));
            x = relu(conv1b(x));
            x = pool(x);
            x = relu(conv2a(x));
            x = relu(conv2b(x));
            x = pool(x);
            x = relu(conv3a(x));
            x = relu(conv3b(x));
            x = pool(x);
            x = relu(conv4a(x));
            x = relu(conv4b(x));

            // detector
            torch::Tensor cPa = relu(convPa(x));
            torch::Tensor semi = convPb(cPa);

            semi = torch::softmax(semi, 1);
            semi = semi.slice(1, 0, 64); // remove "dustbin": [1, 64, H / 8, W / 8]
            semi = semi.permute({0, 2, 3, 1}); // [1, H / 8, W / 8, 64]

            // reshape to get full heatmap
            int hC = semi.size(1);
            int wC = semi.size(2);
            int cell = 8;
            semi = semi.contiguous().view({-1, hC, wC, cell, cell});
            semi = semi.permute({0, 1, 3, 2, 4});
            semi = semi.contiguous().view({-1, hC * cell, wC * cell});
            semi = semi.squeeze(0); // [H, W]

            // descriptor
            torch::Tensor cDa = relu(convDa(x));
            torch::Tensor desc = convDb(cDa);

            torch::Tensor dn = torch::norm(desc, 2, 1);
            desc = desc.div(torch::unsqueeze(dn, 1)); // [1, 256, H / 8, W / 8]

            return std::make_pair(semi, desc);
        }

        torch::nn::ReLU relu;
        torch::nn::MaxPool2d pool;

        torch::nn::Conv2d conv1a, conv1b;
        torch::nn::Conv2d conv2a, conv2b;
        torch::nn::Conv2d conv3a, conv3b;
        torch::nn::Conv2d conv4a, conv4b;

        torch::nn::Conv2d convPa, convPb;
        torch::nn::Conv2d convDa, convDb;
    };

    class SuperPointExtractor {
        public:
            SuperPointExtractor();
            void DetectAndDescribe(int nFeatures, cv::Mat img, shared_ptr<Frame> frame,
                                   std::vector<shared_ptr<SuperPoint>>& features);

            void DrawFeatures(cv::Mat image, vector<shared_ptr<SuperPoint>> features);
        private:
            shared_ptr<SuperPointNet> model;
            float conf_threshold = 0.01;
    };
}

#endif