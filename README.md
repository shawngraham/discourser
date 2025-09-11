# discourser

A Streamlit application for analyzing the influence of a core corpus of materials against another corpus.

- Python 3.10

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

You can create a conda environment with:
```bash
conda create -n discourser python=3.10
conda activate discourser
```

## Running the Application

```bash
streamlit run app.py
```

## Usage

1. **Project Setup**: Create or load a project
2. **Core Corpus**: Upload CSV metadata + text files for influential texts
3. **Target Corpus**: Upload corpus to analyze for influence
4. **Analysis**: 
5. **Results**:

Nb the 'save project' is just saving your core and target corpuses; you still need to initialize the embeddings and so on.

## Data Format

### CSV Metadata
Required columns: filename, title, author, date, source
Optional columns: document_type

### Text Files
- UTF-8 encoded .txt files
- Paragraphs separated by double blank lines
- Filename must match CSV metadata


```mermaid
graph TD
    A[Project Setup] --> B[Core Corpus Upload]
    A --> C[Target Corpus Upload]
    
    B --> D[Core Corpus Processing]
    C --> E[Target Corpus Processing]
    
    D --> F[Core Embeddings Generation]
    E --> G[Target Embeddings Generation]
    
    F --> H[Term Extraction from Core]
    H --> I[Suggested Terms for Vectors]
    
    F --> J[Core-Target Similarity Matrix]
    G --> J
    
    J --> K[Most Influential Core Texts]
    
    I --> L[Custom Vector Creation]
    F --> L
    
    L --> M[Vector Projection Analysis]
    F --> M
    G --> M
    
    M --> N[2D/3D Vector Spaces]
    
    F --> O[Topic Modeling on Core]
    O --> P[Topic-Target Analysis]
    G --> P
    
    J --> Q[Results & Visualization]
    K --> Q
    N --> Q
    P --> Q
    
    subgraph "Core Corpus Flow"
        B
        D
        F
        H
        I
        O
    end
    
    subgraph "Target Corpus Flow"
        C
        E
        G
    end
    
    subgraph "Comparative Analysis"
        J
        K
        M
        N
        P
    end
    
    subgraph "Vector Analysis"
        L
        M
        N
    end
    
    style B fill:#e1f5fe
    style C fill:#f3e5f5
    style F fill:#e8f5e8
    style G fill:#fff3e0
    style J fill:#ffebee
    style L fill:#f1f8e9
    style O fill:#e0f2f1
```
