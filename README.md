# বার্তাবন্ধু - Bengali SMS Classifier

## Overview
বার্তাবন্ধু (BartaBondhu) is an intelligent Bengali SMS classification system that automatically categorizes Bengali text messages into three distinct categories: spam, promotional, or normal messages. This tool helps Bengali speakers organize their messages effectively and filter out unwanted content.

## Live Demo
Try the live application: [Bengali SMS Classifier App](https://bengalismsclassifier.streamlit.app/)

## Features
-  Accurate classification of Bengali SMS messages
-  Real-time message processing
-  Machine learning-based categorization
-  User-friendly interface through Streamlit
-  High precision and recall metrics

## How It Works
The classifier uses Natural Language Processing (NLP) techniques and machine learning algorithms to analyze message content and determine its category:
- **Normal Messages**: Personal communications and important notifications
- **Promotional Messages**: Marketing content, offers, and advertisements
- **Spam Messages**: Unsolicited or potentially harmful messages

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Bengali_sms_classifier.git
   cd Bengali_sms_classifier
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Dataset Information
The classifier is trained on multiple datasets:
- `bengali_sms_dataset.csv`: Combined dataset for training
- `bengali_normal_sms_dataset.csv`: Dataset containing normal messages
- `promo_sms_dataset.csv`: Dataset with promotional messages
- `sms_dataset.csv`: General SMS dataset for training and validation

## Model Information
The pre-trained model (`model1.pkl`) and vectorizer (`vectorizer1.pkl`) are included in the repository for immediate use. The model has been trained on thousands of Bengali SMS messages to ensure high accuracy.

## Technical Implementation
The project uses:
- **Jupyter Notebook** for model development and analysis
- **Python** with scikit-learn for machine learning implementation
- **NLTK** for natural language processing tasks
- **Streamlit** for creating the web interface

## Contributing
Contributions to improve বার্তাবন্ধু are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Implement your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Future Enhancements
- Implementation of deep learning models for improved accuracy
- Support for more Bengali dialects and regional variations
- Integration with SMS apps for automatic filtering
- Enhanced user interface with more detailed analysis

## License
This project is available for use under the MIT License.

## Acknowledgments
- Thanks to all contributors who have helped develop and improve this project
- Special recognition to the open-source NLP community for resources and tools
