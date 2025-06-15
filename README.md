# Universal Data Analyst Agent 🤖📊

*AI-Powered Analysis for ALL File Types - Your Complete Data Analysis Companion*

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

[![Together.ai](https://img.shields.io/badge/Powered%20by-Together.ai-green.svg)](https://together.ai/)

## 🌟 Overview

The Universal Data Analyst Agent is a comprehensive AI-powered tool that can analyze virtually any type of file or data format. Whether you're working with structured data (CSV, Excel), documents (PDF, Word, PowerPoint), text files, or even images, this agent provides intelligent insights, automated analysis, and interactive visualizations.

### ✨ Key Features

- **🗂️ Universal File Support**: Handles 10+ file formats including CSV, Excel, PDF, Word, PowerPoint, images, and text files
- **🤖 AI-Powered Analysis**: Leverages Together.ai's Llama models for intelligent insights and natural language querying
- **📊 Automated Visualizations**: Creates comprehensive dashboards and interactive plots
- **💬 Natural Language Queries**: Ask questions about your data in plain English
- **📈 Statistical Analysis**: Automatic correlation analysis, missing value detection, and data quality assessment
- **🖼️ OCR Text Extraction**: Extracts and analyzes text from images
- **📝 Comprehensive Reporting**: Generates detailed analysis reports in multiple formats
- **📚 Analysis History**: Tracks all your analyses with exportable history

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Together.ai API key ([Get one here](https://together.ai/))

### Installation

1. **Clone or download the script**:
   ```bash
   # Save the code as universal_analyst.py
   ```

2. **Install required packages**:
   ```bash
   pip install pandas numpy requests matplotlib seaborn openpyxl python-docx PyPDF2 pillow pdfplumber plotly wordcloud scikit-learn nltk python-pptx pytesseract
   ```

3. **Run the agent**:
   ```bash
   python universal_data_analyst.py
   ```

4. **Enter your Together.ai API key** when prompted

## 📋 Supported File Types

| Category | File Types | Capabilities |
|----------|------------|--------------|
| **Structured Data** | CSV, Excel (.xlsx, .xls), JSON arrays | Full statistical analysis, visualizations, correlations |
| **Documents** | PDF, Word (.docx), PowerPoint (.pptx) | Text extraction, content analysis, sentiment analysis |
| **Text Files** | TXT, Markdown | Natural language processing, keyword analysis, sentiment |
| **Images** | JPG, PNG, BMP, TIFF, GIF | OCR text extraction, metadata analysis, content insights |

## 🎯 Core Commands

### File Operations
```bash
load <filename>          # Load any supported file type
summary                  # Get comprehensive file overview
info                     # Display current file information
```

### Analysis Commands
```bash
analyze                  # Perform AI-powered comprehensive analysis
ask <question>           # Ask specific questions about your data
report                   # Generate detailed analysis report
```

### Visualizations (Structured Data)
```bash
visualize                # Create data analysis dashboard
visualize interactive    # Create interactive Plotly visualizations
```

### Export & History
```bash
export json              # Export analysis results as JSON
export txt               # Export analysis results as text
history                  # View all previous analyses
```

### Help
```bash
help                     # Display comprehensive help menu
quit                     # Exit the application
```

## 🔧 Detailed Usage Examples

### Example 1: Analyzing Sales Data (CSV)
```bash
🔍 Enter command: load sales_data.csv
✅ Successfully loaded structured data 'sales_data.csv'
   Shape: 1000 rows × 8 columns
   Columns: product, price, quantity, date, region
   Size: 0.15 MB

🔍 Enter command: analyze
🔍 Performing comprehensive analysis...
✅ **File Overview**
   • File: sales_data.csv
   • Type: Structured
   • Shape: 1000 rows × 8 columns
   
✅ **AI Analysis Insights**
   Key patterns identified:
   - Seasonal trends in Q4 showing 40% increase in sales
   - Top performing regions: North (35%), South (28%)
   - Premium products (>$100) show higher profit margins
   [Additional insights...]

🔍 Enter command: ask What are the top 5 selling products?
✅ **Question:** What are the top 5 selling products?
**Answer:**
Based on the sales data analysis:
1. Product A: 145 units sold ($14,500 revenue)
2. Product B: 132 units sold ($13,200 revenue)
[Detailed breakdown with supporting data...]

🔍 Enter command: visualize
✅ Visualization created successfully!
   Saved as: visualization_20241215_143022.png
   Type: Data Analysis Dashboard
```

### Example 2: Analyzing a Research Paper (PDF)
```bash
🔍 Enter command: load research_paper.pdf
✅ Successfully loaded text document 'research_paper.pdf'
   Words: 8,547
   Lines: 342
   Size: 2.3 MB

🔍 Enter command: ask What are the main findings of this research?
✅ **Answer:**
The research presents three main findings:
1. Machine learning models show 23% improvement over traditional methods
2. Data preprocessing techniques reduce error rates by 15%
3. Cross-validation results demonstrate model reliability
[Detailed analysis with supporting quotes...]

🔍 Enter command: report
📊 Comprehensive report generated!
   File: analysis_report_20241215_143555.md
   Sections: Executive Summary, Content Analysis, Key Insights, Recommendations
```

### Example 3: Analyzing an Image with Text
```bash
🔍 Enter command: load infographic.png
✅ Successfully loaded image 'infographic.png'
   Dimensions: 1920×1080 pixels
   📝 Contains text: Yes
   Size: 1.2 MB

🔍 Enter command: analyze
🔍 Performing comprehensive analysis...
✅ **Image Properties**
   • Format: PNG
   • Dimensions: 1920×1080 pixels
   • Contains extracted text about market trends
   
✅ **AI Analysis Insights**
   The infographic contains key business metrics:
   - Revenue growth: 15% YoY
   - Customer satisfaction: 87%
   [OCR extracted content analysis...]
```

## 🎨 Visualization Capabilities

The agent automatically creates comprehensive dashboards including:

- **📊 Data Distribution Histograms**: Shows distribution of numeric columns
- **🔥 Missing Values Heatmap**: Visualizes data completeness patterns
- **🔗 Correlation Matrix**: Displays relationships between variables
- **📈 Statistical Overviews**: Bar charts of key metrics

### Interactive Visualizations
When you use `visualize interactive`, the agent creates:
- Interactive scatter plots with hover details
- Dynamic box plots for outlier detection
- Zoomable and filterable charts using Plotly

## 🧠 AI Analysis Features

### For Structured Data
- **Data Quality Assessment**: Automatic detection of missing values, outliers, and inconsistencies
- **Statistical Insights**: Correlation analysis, descriptive statistics, trend identification
- **Business Intelligence**: Revenue patterns, customer behavior, performance metrics
- **Predictive Insights**: Trend forecasting and anomaly detection

### For Text Content
- **Sentiment Analysis**: Emotional tone assessment using VADER sentiment analyzer
- **Topic Modeling**: Key themes and concept extraction
- **Content Quality**: Readability analysis and structure assessment
- **Keyword Analysis**: Most frequent terms and phrases

### For Images
- **OCR Text Extraction**: Automatic text recognition from images
- **Metadata Analysis**: Image properties, dimensions, and technical details
- **Content Classification**: Visual content type identification

## 📊 Export and Reporting

### Analysis Reports
Generated reports include:
- **Executive Summary**: Key findings and recommendations
- **Technical Analysis**: Detailed methodology and statistics
- **Business Insights**: Actionable recommendations
- **Visual Summaries**: Chart descriptions and interpretations

### Export Formats
- **JSON**: Machine-readable format with complete analysis history
- **Text**: Human-readable format for documentation
- **Markdown**: Formatted reports with proper structure

## 🔧 Advanced Configuration

### API Settings
The agent uses Together.ai's Llama-4-Maverick model by default. You can modify the model in the code:
```python
self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
```

### Visualization Customization
Modify the `create_visualization()` method to customize:
- Chart types and styling
- Color schemes
- Plot dimensions
- Statistical measures

### Text Analysis Settings
Configure NLTK settings for:
- Custom stopwords
- Language-specific processing
- Sentiment analysis thresholds

## 🚨 Troubleshooting

### Common Issues

1. **NLTK Download Errors**:
   ```bash
   # Manual NLTK data download
   python -c "import nltk; nltk.download('all')"
   ```

2. **Missing Dependencies**:
   ```bash
   # Install specific packages
   pip install --upgrade pandas matplotlib seaborn
   ```

3. **OCR Issues (Tesseract)**:
   ```bash
   # Install Tesseract OCR
   # Windows: Download from GitHub
   # macOS: brew install tesseract
   # Linux: sudo apt-get install tesseract-ocr
   ```

4. **API Connection Issues**:
   - Verify your Together.ai API key
   - Check internet connection
   - Ensure API quota availability

### Performance Tips

- **Large Files**: Files over 100MB may take longer to process
- **Image Processing**: High-resolution images require more processing time
- **Memory Usage**: Large datasets may require increased system RAM

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**: Create detailed issue reports
2. **Feature Requests**: Suggest new file formats or analysis types
3. **Code Contributions**: Submit pull requests with improvements
4. **Documentation**: Help improve guides and examples

### Development Setup
```bash
# Clone the repository
git clone <repository-url>
cd universal-data-analyst

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```


## 🙏 Acknowledgments

- **Together.ai**: For providing the powerful Llama AI models
- **Open Source Libraries**: pandas, matplotlib, nltk, and many others
- **Community**: All contributors and users providing feedback

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join our community discussions
- **Documentation**: Check our comprehensive docs
- **Email**: Contact us at [your-email]

## 🔮 Roadmap

### Upcoming Features
- [ ] **Database Connectivity**: MySQL, PostgreSQL, MongoDB support
- [ ] **Cloud Storage**: AWS S3, Google Drive integration
- [ ] **Advanced ML**: Automated model training and prediction
- [ ] **Web Interface**: Browser-based GUI
- [ ] **API Endpoints**: RESTful API for integration
- [ ] **Collaborative Features**: Team sharing and collaboration
- [ ] **Scheduled Analysis**: Automated recurring reports

### Version History
- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added image analysis and OCR support
- **v1.2.0**: Enhanced AI insights and interactive visualizations

---

**Ready to analyze your data?** 🚀

```bash
python universal_data_analyst.py
```

*Transform your data into insights with the power of AI!*
