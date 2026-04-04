# NEPSE AI Analyzer Platform

A professional, AI-powered fintech platform for analyzing the Nepal Stock Exchange (NEPSE). Features live market data, advanced technical signals, persistent portfolio tracking, and structural trend analysis with Gemini 1.5 Flash.

## Architecture

- **Backend**: Node.js + Express
  - Live market data via `@rumess/nepse-api`.
  - Technical analysis engine (RSI, SMA, MACD, Bollinger Bands) using `technicalindicators`.
  - Gemini 1.5 Flash integration for AI-powered trade recommendations.
  - Local JSON persistence for portfolio tracking.
- **Frontend**: React + Vite + Tailwind CSS v4
  - Modern, responsive dark-themed dashboard.
  - Advanced data visualization with `recharts`.
  - Client-side routing with `react-router-dom`.

## Getting Started

### 1. Prerequisites
- Node.js (v18+)
- Gemini API Key ([Get one here](https://aistudio.google.com/))

### 2. Backend Setup
```bash
cd backend
npm install
```
Create a `.env` file in the `backend` directory:
```env
GEMINI_API_KEY=your_key_here
PORT=5000
```
Run the server:
```bash
npm start
```

### 3. Frontend Setup
```bash
cd react-frontend
npm install
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

## Deployment to Vercel

### Frontend (Vite)
1. Push your code to GitHub.
2. Connect your repository to [Vercel](https://vercel.com).
3. Vercel will automatically detect the Vite project. 
4. Ensure the **Root Directory** is set to `react-frontend`.
5. Set the **Build Command** to `npm run build` and **Output Directory** to `dist`.

### Backend
For production deployment, consider:
- Deploying the `backend` to a service like **Render**, **Railway**, or **Heroku**.
- Updating the `API_BASE_URL` in `react-frontend/src/services/api.js` to your deployed backend URL.
- > [!NOTE]
  > The current portfolio system uses local JSON. For multi-user production, replace `portfolioService.js` logic with a database like MongoDB.
