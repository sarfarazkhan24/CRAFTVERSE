# Abstainity (Built on ClinSure)
Abstainity is a PyTorch-native reliability layer that gives your image classifiers the ability to say "I don't know" when faced with ambiguous, out-of-distribution, or low-confidence inputs.

## Quickstart

### 1. Run the Backend API
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

### 2. Run the Frontend UI
In a separate terminal:
```bash
npm install
npm run dev
```

See `docs/how-it-works.md` for a plain-language explanation of the 3 Abstainity capability tiers (Entropy fallback, MC-Dropout, and Mahalanobis OOD).

---

# Original ClinSure Documentation
ClinSure is an uncertainty-aware chest X-ray classification and safety triage frontend designed to prevent dangerous overconfident predictions. It combines **temperature calibration**, **MC-Dropout for epistemic uncertainty**, and **Mahalanobis-based Out-of-Distribution (OOD) detection** to classify radiographs into three distinct actionable clinical states:

* **`ACCEPT`**: High calibrated confidence, low epistemic uncertainty, in-distribution. Eligible for automated report draft.
* **`UNCERTAIN`**: High model parameter doubt or borderline confidence. Flagged for secondary radiologist review.
* **`ABSTAIN`**: Out-of-Distribution (OOD) by Mahalanobis feature space distance (scanner shift, foreign hardware/pacemakers, corrupted acquisition). Prediction is withheld and escalated for manual clinical workflow.

> [!IMPORTANT]
> **Safety Notice**: ClinSure is a clinical decision-support and research prototype, **NOT** an autonomous diagnostic system. The UI communicates that all AI outputs are assistive and require qualified clinical review.

---

## 1. Quick Start

### Installation
```bash
cd C:\Users\91988\.gemini\antigravity\scratch\clinsure
npm install
```

### Development Server
```bash
npm run dev
```
The application will start at `http://localhost:3000`.

### Production Build
```bash
npm run build
npm run preview
```

---

## 2. Project Structure

```
clinsure/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── public/
│   ├── favicon.svg
│   └── samples/
└── src/
    ├── App.tsx                  # Main router configuration
    ├── main.tsx                 # Application entry point
    ├── index.css                # Global medical workstation styles & tailwind
    ├── types/
    │   └── index.ts             # TypeScript domain models (AnalysisResult, OOD, Uncertainty, etc.)
    ├── data/
    │   └── mockData.ts          # Realistic clinical mock datasets & 3 pre-configured demo cases
    ├── services/
    │   └── api.ts               # API service layer (FastAPI bridge + offline simulation engine)
    ├── hooks/
    │   ├── useAnalysis.ts       # Hook for managing multi-stage analysis pipeline
    │   ├── useAnalysisHistory.ts# Hook for case registry data
    │   └── useDashboardData.ts  # Hook for aggregate statistics
    ├── components/
    │   ├── layout/
    │   │   ├── Navbar.tsx       # Top navigation & system status
    │   │   ├── Sidebar.tsx      # Persistent sidebar with route links & demo switcher
    │   │   ├── Footer.tsx       # Persistent safety disclaimer
    │   │   └── MainLayout.tsx   # Responsive workstation layout shell
    │   ├── common/
    │   │   ├── DecisionBadge.tsx# Color + Icon + Label badges for ACCEPT / UNCERTAIN / ABSTAIN
    │   │   ├── MetricCard.tsx   # Clinical KPI metric cards
    │   │   └── DemoBanner.tsx   # Demo mode switcher & clinical research disclaimer
    │   ├── viewer/
    │   │   └── XRayViewer.tsx   # Professional PACS viewer (Zoom, Pan, Invert, Grad-CAM, Attention)
    │   ├── analysis/
    │   │   ├── UploadDropzone.tsx       # Drag & drop upload + sample radiograph selector
    │   │   ├── WorkflowProgress.tsx     # 7-stage animated analysis progress
    │   │   ├── ProbabilityBars.tsx      # Calibrated predictions + 95% Bayesian credible intervals
    │   │   ├── UncertaintyPanel.tsx     # Epistemic mutual info, entropy & MC variance gauges
    │   │   ├── OODPanel.tsx             # Mahalanobis distance meter vs calibrated threshold
    │   │   └── DecisionFlowExplainer.tsx# Visual decision graph & "Why did ClinSure decide this?"
    │   └── charts/
    │       ├── PredictionDistributionDonut.tsx  # Triage breakdown donut
    │       ├── ConfidenceVsUncertaintyScatter.tsx # Confidence vs epistemic uncertainty scatter
    │       ├── ReliabilityDiagram.tsx          # Calibration curve vs perfect diagonal (ECE)
    │       ├── ConfidenceDistributionChart.tsx  # Confidence histogram
    │       ├── UncertaintyDistributionChart.tsx # Epistemic uncertainty histogram
    │       ├── RobustnessComparisonChart.tsx   # Cross-hospital benchmark (Hospitals A, B, C, D)
    │       └── AbstentionBehaviorChart.tsx     # Triage response under distribution shift
    └── pages/
        ├── LandingPage.tsx      # Overview, problem statement, 3-state system, AI architecture
        ├── DashboardPage.tsx    # Clinical triage overview, KPI cards, recent analyses table
        ├── AnalyzePage.tsx      # Upload, sample loader, and 7-stage evaluation workflow
        ├── ResultsPage.tsx      # Deep dive: X-ray viewer, uncertainty gauges, OOD meter, decision breakdown
        ├── HistoryPage.tsx      # Searchable & filterable study registry with pagination
        ├── AnalyticsPage.tsx    # Model accuracy, calibration reliability diagrams, ECE reduction
        ├── RobustnessPage.tsx   # Cross-hospital shift evaluation (acquisition, artifact, pediatric shift)
        └── AboutPage.tsx        # Methodology, AI components, tech stack, and ethical disclaimer
```

---

## 3. Connecting the FastAPI Backend

The frontend is specifically architected for plug-and-play connection to a FastAPI backend.

### Configuration
Create a `.env` file in the project root:
```env
VITE_API_URL=http://localhost:8000
```

When `VITE_API_URL` is set, `src/services/api.ts` automatically redirects calls from the offline demo simulation to your live FastAPI endpoints.

### API Endpoints Expected by Frontend:

| Method | Endpoint | Description | Request Payload | Response |
|---|---|---|---|---|
| `POST` | `/api/analyze` | Analyze chest X-ray | `multipart/form-data` with `file` | `AnalysisResult` JSON |
| `GET` | `/api/results/{id}` | Get study by Case ID | URL parameter | `AnalysisResult` JSON |
| `GET` | `/api/history` | Get recent case registry | None | `AnalysisHistoryItem[]` JSON |
| `GET` | `/api/dashboard/stats`| Get aggregate triage stats | None | `DashboardStats` JSON |
| `GET` | `/api/model/metrics` | Get calibration & ECE data | None | `ModelMetrics` JSON |
| `GET` | `/api/robustness/metrics`| Cross-hospital evaluation | None | `RobustnessMetrics` JSON |

### Example FastAPI Implementation Stub:
```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="ClinSure Backend API")

@app.post("/api/analyze")
async def analyze_radiograph(file: UploadFile = File(...)):
    # 1. Preprocess & extract deep features z
    # 2. Run MC-Dropout inference (25 stochastic passes)
    # 3. Apply Temperature Scaling calibration (T=1.18)
    # 4. Compute Mahalanobis distance M(z) to class centroids
    # 5. Execute 3-way triage policy:
    #    if M(z) > 1.50 -> ABSTAIN
    #    elif epistemic_uncertainty > 0.35 or confidence < 0.85 -> UNCERTAIN
    #    else -> ACCEPT
    return {
        "id": "XR-2026-999",
        "decision": "ACCEPT",
        # ... conforms to AnalysisResult interface in src/types/index.ts
    }
```

---

## 4. Demo Mode & Testing

ClinSure includes a built-in **Demo Mode** with three pre-configured clinical test cases:

* **Case 001 (`XR-2026-001`)**:
  * Prediction: Pneumonia (94.2%)
  * Epistemic Uncertainty: 0.18 (Low)
  * Mahalanobis OOD Score: 0.82 (In-Distribution, limit 1.50)
  * Decision: **`ACCEPT`**
* **Case 002 (`XR-2026-002`)**:
  * Prediction: No Finding (71.4%)
  * Epistemic Uncertainty: 0.48 (Moderate / Disagreement across passes)
  * Mahalanobis OOD Score: 1.22 (In-Distribution)
  * Decision: **`UNCERTAIN`** (Secondary review required)
* **Case 003 (`XR-2026-003`)**:
  * Raw Softmax: Pneumonia (88.1% - False confidence blocked)
  * Epistemic Uncertainty: 0.89 (High)
  * Mahalanobis OOD Score: 2.14 (Out-of-Distribution ICU Shift / Pacemaker artifact)
  * Decision: **`ABSTAIN`** (Prediction withheld; escalated to radiologist)

You can launch and test these cases anytime from the persistent header banner, sidebar, or sample picker on the Analyze page.
