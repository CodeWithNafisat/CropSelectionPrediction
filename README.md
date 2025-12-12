#🌾 Intelligent Crop Recommendation System> **Bridging the gap between agricultural science and artificial intelligence.**

##📖 Table of Contents* [Problem Statement](https://www.google.com/search?q=%23-problem-statement)
* [What the App Does](https://www.google.com/search?q=%23-what-the-app-does)
* [Our Solution](https://www.google.com/search?q=%23-solution)
* [Challenges We Overcame](https://www.google.com/search?q=%23-challenges)
* [Technologies Used](https://www.google.com/search?q=%23-technologies--requirements)
* [How to Run Locally](https://www.google.com/search?q=%23-how-to-use)
* [About Me](https://www.google.com/search?q=%23-about-me)

---

##🚩 Problem StatementAgriculture is increasingly becoming data-driven, yet many farmers and agricultural planners lack accessible tools to interpret complex soil and climate data. Choosing the wrong crop for specific environmental conditions can lead to:

* **Reduced yields** and financial loss.
* **Soil degradation** due to nutrient imbalance.
* **Inefficient resource usage** (water and fertilizers).

Traditional methods often rely on intuition or generic guidelines that don't account for the subtle interactions between soil nutrients (N, P, K) and climatic factors.

---

##📱 What the App DoesThis application is an **interactive decision-support tool** that recommends the most suitable crop to grow based on specific environmental parameters.

It goes beyond simple prediction by offering **Explainable AI (XAI)**. It doesn't just tell the user *what* to grow; it explains *why* that recommendation was made, breaking down the biological impact of each soil nutrient and weather condition on the decision.

---

##💡 SolutionWe built a deep learning classification system wrapped in a user-friendly interface.

1. **Deep Learning Core:** Utilizes a custom PyTorch Feed-Forward Neural Network to analyze patterns across 7 different agricultural features.
2. **Interactive Dashboard:** A Streamlit-based UI allows users to easily input soil data via sliders and receive instant feedback.
3. **Transparency:** Integrated **SHAP (SHapley Additive exPlanations)** values to visualize feature importance. The app generates dynamic text explaining how factors like Nitrogen levels or Rainfall specifically influenced the model's choice (e.g., *"High rainfall positively influenced the recommendation for Rice"*).

---

##🧗 ChallengesBuilding this system came with specific hurdles:

* **Model Explainability:** Deep learning models are often "black boxes." Implementing SHAP to translate complex tensor calculations into human-readable biological insights was a key technical challenge.
* **User Experience (UX):** Designing a "Smart Reset" feature to ensure old predictions don't persist when inputs change required implementing custom session state callbacks in Streamlit.
* **Data Balancing:** ensuring the model didn't favor common crops over niche ones required careful data preprocessing and scaling.

---

##🛠 Technologies & RequirementsThis project is built using the following stack:

| Category | Technology |
| --- | --- |
| **Language** | Python 3.x |
| **Frontend** | Streamlit |
| **Deep Learning** | PyTorch (`torch`, `torch.nn`) |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Visualization** | Matplotlib, Seaborn |

**Dependencies:**
To run this project, you will need the specific libraries listed in `requirements.txt`.

---

##🚀 How to UseFollow these steps to get the app running on your local machine.

###1. Clone the RepositoryOpen your terminal and run:

```bash
git clone https://github.com/CodeWithNafisat/Crop-Recommendation-System.git
cd Crop-Recommendation-System

```

###2. Create a Virtual Environment (Optional but Recommended)```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

###3. Install Dependencies```bash
pip install -r requirements.txt

```

*(Note: Ensure you have `best_model.pth`, `scaler.pkl`, `le.pkl`, and `bg.csv` in the root directory).*

###4. Run the Application```bash
streamlit run app.py

```

###5. Access the AppThe app will automatically open in your default web browser at:
`http://localhost:8501`

---

##👩‍💻 About Me**Nafisat Abdulraheem**

I'm passionate about data science and machine learning, and I focus on building AI tools that are actually useful and understandable for real people solving real problems.

* **GitHub:** [CodeWithNafisat](https://github.com/CodeWithNafisat)
* **LinkedIn:** [Nafisat Abdulraheem](https://www.linkedin.com/in/nafisat-abdulraheem-7a193b337)

---

##📄 LicenseThis project uses the **MIT License**. Check the LICENSE file for details.
