## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- pnpm

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:
```
ANTHROPIC_API_KEY=your-key-here
```

```bash
python app.py
```

### Frontend
```bash
cd materialaize
pnpm install
pnpm dev
```

Backend runs on `localhost:8000`, frontend on `localhost:3000`.
