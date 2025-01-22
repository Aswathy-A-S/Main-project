*Image Tampering Detection Using ELA and Metadata Analysis*

This project proposes a comprehensive and integrated framework combining Error Level Analysis (ELA)
and Metadata Analysis for robust image tampering detection. The proposed system addresses the limitations of existing single-modality approaches by combining both pixel-level and contextual analysis,
leveraging cutting-edge machine learning models and domain-specific datasets to significantly improve
tampering detection accuracy and scalability. ELA is employed to identify tampered regions by detecting
discrepancies introduced during the lossy compression process. When an image is saved in formats like
JPEG, the compression introduces artifacts that create pixel-level inconsistencies. Tampered regions disrupt these artifacts, making them detectable through ELA. The system amplifies these variations by calculating the absolute differences between the original and recompressed images, highlighting manipulated
areas. To implement ELA, the CASIA2.0 dataset is utilized, which contains a curated collection of both
real and tampered images. The dataset is preprocessed to generate ELA representations, with an optimal
compression quality level set to 90 percentage to achieve the best results. These ELA images are then
fed into a deep learning model, DenseNet121, which is fine-tuned to classify real and tampered images.
The model achieves high accuracy by leveraging the intricate features extracted from the ELA maps, enabling it to detect subtle and complex manipulations that are often undetectable by traditional methods.
The training pipeline incorporates advanced optimization techniques, data augmentation, and regularization methods to ensure the model is resilient to overfitting and adaptable to diverse image qualities and
resolutions. By integrating ELA, the system effectively identifies tampering at the pixel level, offering
a solid foundation for tampering detection. To complement the pixel-level detection achieved through
ELA, the system incorporates Metadata Analysis, which validates the contextual integrity of the image.
Digital images carry extensive metadata, including timestamp, geographical location, camera model, and
other parameters. These metadata elements are cross-verified to detect inconsistencies or anomalies that
may indicate tampering. A key feature of the proposed system is weather validation, where the depicted
weather conditions in an image are validated against historical weather data. A Weather-CNN model
is trained to classify weather conditions in outdoor images into categories such as sunny, cloudy, rainy,
and lightning. The training dataset consists of 1,804 images for training and 451 images for validation,
carefully collected from various sources to ensure diversity and coverage of different weather scenarios.
Using the metadata, the system extracts crucial details such as longitude, latitude, timestamp, and camera
settings. These details are then sent as input to a weather API, which retrieves historical weather data for
the specified time and location. This data is compared with the predictions of the Weather-CNN model.
Any discrepancies between the depicted and actual weather conditions provide strong evidence of image
manipulation.The proposed framework integrates the outputs of ELA and Metadata Analysis to create
a unified decision-making process. While ELA focuses on the detection of pixel-level discrepancies,
Metadata Analysis validates the broader contextual information, ensuring that tampering is detected both
visually and contextually. To ensure scalability and user-friendliness, the framework is implemented as
a Streamlit application, providing a seamless interface for users to upload images and receive tampering
analysis results.
