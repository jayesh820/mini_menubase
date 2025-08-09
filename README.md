Multi-Task System Administration & Machine Learning Dashboard

A comprehensive Python-based dashboard built with Streamlit that provides enterprise-level system administration, machine learning capabilities, and automation tools in a single unified interface.

🌟 Features
🖥️ Windows System Management
System Information & Monitoring: Real-time CPU, RAM, and process monitoring

File & Folder Operations: Complete file management with search, compression, and content analysis

Application Management: Launch system applications and manage Windows services

Network Connectivity: Network diagnostics, adapter management, and connectivity testing

Power Operations: System shutdown, restart, and sleep controls with safety confirmations

🐧 Linux Administration (SSH-based)
Remote System Management: Full Linux system administration via SSH

Process & Service Management: SystemD service control and process monitoring

Package Management: DNF/YUM package installation and system updates

User & Group Management: Complete user administration with sudo support

Network & Firewall: Network diagnostics and firewalld management

Log Analysis: System log monitoring and cron job management

🐳 Docker Management
Container Operations: Complete container lifecycle management

Image Management: Build, tag, push, and pull Docker images

Network & Volume Management: Docker networking and persistent storage

Docker Compose: Multi-container application deployment

Docker Swarm: Cluster management and orchestration

System Cleanup: Automated pruning and resource optimization

🤖 Machine Learning Pipeline
Data Preprocessing: Advanced data cleaning, imputation, and feature engineering

Interactive Visualization: Plotly-powered charts and data exploration

Smart ML Trainer: Automated model selection with 12+ algorithms

Model Persistence: Save and load trained models for production use

Unsupervised Learning: K-means clustering with PCA visualization

Cross-Validation: Robust model evaluation with k-fold validation

📱 Communication & Automation
Multi-Channel Messaging: WhatsApp, SMS, and email integration

Camera Operations: Photo capture and video recording

System Monitoring: Process tracking and resource utilization

🛠️ Technical Stack
Frontend: Streamlit with custom styling and responsive design

System Integration: Paramiko (SSH), psutil (system monitoring), subprocess (system commands)

Machine Learning: Scikit-learn, XGBoost, LightGBM with automated preprocessing

Data Visualization: Plotly, Matplotlib, Seaborn for interactive charts

Communication APIs: Twilio, PyWhatKit, SMTP for messaging capabilities

Container Management: Docker API integration and Docker Compose support

🚀 Getting Started
Clone the repository

bash
git clone [repository-url]
cd mini_menu_base
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
streamlit run mini_menu_base.py
📋 Prerequisites
Python 3.7+

Docker (for container management features)

SSH access to target Linux machines

Optional: Twilio account for messaging features

🎯 Use Cases
System Administrators: Centralized management of hybrid Windows/Linux environments

DevOps Engineers: Docker container orchestration and deployment automation

Data Scientists: Complete ML workflow from data preprocessing to model deployment

IT Teams: Streamlined monitoring and maintenance of enterprise infrastructure

🔧 Configuration
The application includes comprehensive configuration options for:

SSH connections and credentials management

ML model hyperparameter tuning

System operation safety confirmations

Communication API credentials

📈 Enterprise Ready
Built with enterprise requirements in mind:

Modular architecture for easy extension

Comprehensive error handling and logging

Security-first approach with credential protection

Scalable design supporting large-scale operations

🤝 Contributing
Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.
