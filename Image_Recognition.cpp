#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

int main()
{
    Mat img = imread("Image.png");
    if (img.empty()) {
        cout << "图像读取失败！" << endl;
        return -1;
    }
    // 获取图像的行数(高度)和列数(宽度),后续遍历像素
    int rows = img.rows;
    int cols = img.cols;

    // ===== 转 HSV =====H(色相)不受光照影响，适合航拍图颜色聚类
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // ===== 3. 构建特征（解决 Hue 环形问题）=====
    // 用 cos(h)*s, sin(h)*s 表示颜色
    Mat data(rows * cols, 2, CV_32F);//data矩阵，每行代表原图的一个像素，两列代表该像素的极坐标

    int idx = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            Vec3b pixel = hsv.at<Vec3b>(r, c);

            float h = pixel[0] * 2.0f * CV_PI / 180.0f;
            float s = pixel[1] / 255.0f;

            data.at<float>(idx, 0) = cos(h) * s;
            data.at<float>(idx, 1) = sin(h) * s;

            idx++;
        }
    }

    // ===== 4. K-means 聚类 =====
    int K = 6; // 可调（3~8）
    Mat labels, centers;

    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1);

    //K-means 输出: labels 矩阵(1*像素总数),每个值：0~5(K = 6),代表对应像素的"聚类 ID"
    kmeans(data, K, labels, criteria,
        3, KMEANS_PP_CENTERS, centers);

    // ===== 5. 构建 label 图 =====
    Mat labelImg = labels.reshape(1, rows); // 变成 rows x cols
    labelImg.convertTo(labelImg, CV_32S);

    int totalRegions = 0;
    vector<int> areas;// 存储每个有效区域的面积（像素数）

    // ===== 6. 对每个标签 c，找出所有标签为 c 的像素中，哪些是 “连续连通” 的，统计这些连通块的面积。 =====
    for (int c = 0; c < K; c++) {

        // 生成 mask
        Mat mask = (labelImg == c);//快速筛选同一颜色类的所有像素块

        // 将掩码mask的(0-1)缩放为(0-255)
        mask.convertTo(mask, CV_8U, 255);

        // 形态学优化（去噪）
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 连通域分析
        Mat ccLabels, stats, centroids;//这三个参数均为'connectedComponentsWithStats'函数的输出信息
        int num = connectedComponentsWithStats(mask, ccLabels, stats, centroids);//num是该函数分析的连通域总数

        for (int i = 1; i < num; i++) { 
            int area = stats.at<int>(i, CC_STAT_AREA);

            if (area > 50) { // 过滤小噪声
                totalRegions++;
                areas.push_back(area);//把面积存入 areas 向量，后续输出
            }
        }
    }

    // ===== 7. 输出结果 =====
    cout << "===== 区域统计结果 =====" << endl;
    cout << "区域总数: " << totalRegions << endl;

    for (size_t i = 0; i < areas.size(); i++) {
        cout << "区域 " << i + 1 << " 面积: " << areas[i] << endl;
    }

    // ===== 8. 可视化 =====
    Mat colorResult = Mat::zeros(img.size(), CV_8UC3);
    RNG rng(12345);

    for (int c = 0; c < K; c++) {
        Vec3b color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

        for (int r = 0; r < rows; r++) {
            for (int col = 0; col < cols; col++) {
                if (labelImg.at<int>(r, col) == c) {
                    colorResult.at<Vec3b>(r, col) = color;
                }
            }
        }
    }
    imshow("KMeans分割结果", colorResult);

    waitKey(0);
    destroyAllWindows();

    return 0;
}