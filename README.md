# TE Group 03

# Automated Security Auditing for LLM

## Problem Statement  

Large Language Models (LLMs) are increasingly deployed in production environments, but they remain vulnerable to adversarial attacks like prompt injection, jailbreaking, and manipulation techniques. Current security testing methods are manual, time-consuming, and lack adaptive capabilities to discover evolving attack vectors. Organizations need an automated, intelligent system that can:
- Systematically test LLMs against diverse attack strategies
- Adapt attack techniques based on model responses
- Generate comprehensive security reports
- Provide actionable insights for model hardening

This project addresses the critical need for automated red-teaming tools that can proactively identify vulnerabilities before malicious actors exploit them.

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">
</div>

### Tech Stack  

- **Frontend:** Streamlit *(for rapid prototyping and interactive dashboards)*  
- **Backend:** Python with FastAPI *(for async API handling and orchestration)*  
- **LLM Orchestration:** Custom agents with Groq, NVIDIA APIs, Ollama *(for free, production-grade inference)*  
- **Database:** MongoDB *(for storing attack results, conversation history, and audit logs)*  
- **Attack Strategies:** 15+ custom jailbreak techniques *(persona adoption, prefix injection, multi-turn attacks, etc.)*  
- **Analysis Engine:** Pattern-based + LLM-powered evaluation *(for verdict classification)*  
- **Deployment:** Docker containers *(for reproducible environments)* with local or cloud deployment option

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">
</div>

### Expected Outcome  

1. **Automated Security Testing**: A production-ready system capable of testing any LLM (local or cloud-based) against 15+ adversarial attack strategies
2. **Multi-Turn Adaptive Attacks**: Intelligent conversation-based attacks that learn from model responses and adapt strategies across 5-10 turns
3. **Comprehensive Reporting**: Detailed security audit reports with:
   - Attack Success Rate (ASR) metrics
   - Strategy effectiveness analysis
   - Vulnerability heatmaps
   - Exportable PDF/JSON reports
4. **Real-time Monitoring**: Live dashboard showing:
   - Active attack execution status
   - Response analysis (REFUSED/PARTIAL/JAILBROKEN)
   - Historical trends and patterns
5. **Zero-Cost Operation**: Entire pipeline runs on free APIs (Groq, NVIDIA) and local models (Ollama)
6. **Extensible Architecture**: Modular design allowing easy addition of new attack strategies and target 

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">
</div>

## Hypothesis  

**Primary Hypothesis:**  
*"An automated, multi-agent system employing adaptive attack strategies can achieve a 60-80% success rate in identifying LLM vulnerabilities compared to 10-20% with single-shot attacks, while reducing manual red-teaming effort by 90%."*

**Secondary Hypotheses:**
1. Multi-turn conversational attacks will be 3-5x more effective than single-turn attacks
2. Adaptive strategy selection based on model responses will outperform random strategy selection by 40%+
3. Pattern-based + LLM-powered analysis will achieve >85% accuracy in vulnerability detection
4. The system can test 100+ attack variations in <10 minutes (vs. 8+ hours manually)

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">
</div>

## Technologies  

Edit this as per your project

| **Frontend** | **Backend** | **Database** | **Authentication** | **Payment** |
|-------------|------------|-------------|--------------------|-------------|
| ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3) ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react) | ![Node.js](https://img.shields.io/badge/Node.js-43853D?style=for-the-badge&logo=node.js) | ![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb) | ![Firebase Authentication](https://img.shields.io/badge/Firebase%20Auth-FFCA28?style=for-the-badge&logo=firebase) | ![PayPal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal) |

| **APIs & Services** | **Tools** |  
|--------------------|-----------|  
| ![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase) | ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git) |  
| ![Firebase Auth](https://img.shields.io/badge/Firebase%20Auth-FFCA28?style=for-the-badge&logo=firebase) | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github) |  
|  | ![Jira](https://img.shields.io/badge/Jira-0052CC?style=for-the-badge&logo=jira) |  
|  | ![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code) |  

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">
</div>

## Roles and Responsibilities  
(Edit this as per your project)

#### **Member 1 (Team Lead)**  
- **Role:** Project Management, Frontend Development  
- **Responsibilities:**  
  - Oversee project progress and delivery  
  - Develop frontend and manage UI components  

### **Team Members**  

#### **Member 2**  
- **Role:** Frontend Developer, UI Design  
- **Responsibilities:**  
  - Design and implement user-friendly UI  
  - Collaborate on frontend features and responsiveness  

#### **Member 3**  
- **Role:** Backend Developer, Data Fetching  
- **Responsibilities:**  
  - Develop backend APIs for data retrieval  
  - Optimize backend performance and security  

#### **Member 4**  
- **Role:** Backend Developer, Data Designing  
- **Responsibilities:**  
  - Design and manage database architecture  
  - Ensure data integrity and smooth database operations  


<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">
</div>

## Project Team Members  

<table>
  <tr align="center">
    <td>
      <img src="<image link>?size=250" width="180" height="180" /><br>
      <b>Member 1 (Team Lead)</b><br>
      Frontend Developer Project Management
    </td>
    <td>
     <img src="<image link>?size=250" width="180" height="180" /><br>
      <b>Member 2</b><br>
      Frontend Developer UI Design
    </td>
    <td>
     <img src="<image link>?size=250" width="180" height="180" /><br>
      <b>Member 3</b><br>
      Backend Developer Data Fetching
    </td>
    <td>
      <img src="<image link>?size=250" width="180" height="180" /><br>
      <b>Member 4</b><br>
      Backend Developer Data Designing
    </td>
  </tr>
</table>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">
</div>

## Installation & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd llm_security_auditor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys: GROQ_API_KEY, NVIDIA_API_KEY

# Start MongoDB (Docker)
docker-compose up -d mongodb

# Run the application
streamlit run frontend/app.py
