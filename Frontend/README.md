# SS Argitech Frontend - README

## Project Overview

The frontend is a modern Next.js 14 application with React 18, Tailwind CSS for styling, and Axios for API communication. It provides a user-friendly interface for farmers to access laws, manage profiles, and track activities.

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Configuration

Create `.env.local` file:

```
NEXT_PUBLIC_API_URL=http://localhost:5000/api
NEXT_PUBLIC_AI_API_URL=http://localhost:8000/api/ai
```

### Running the App

**Development:**

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

**Production:**

```bash
npm run build
npm start
```

## API Integration

### API Client Setup

The `lib/api.ts` file provides:

- Axios instance with base URL configuration
- Request interceptor for token injection
- Response interceptor for 401 handling
- Organized API methods by feature

## Environment Variables

| Variable              | Required | Default                   |
| --------------------- | -------- | ------------------------- |
| `NEXT_PUBLIC_API_URL` | Yes      | http://localhost:5000/api |
| `NEXT_PUBLIC_AI_API_URL` | Yes   | http://localhost:8000/api/ai |

## Tailwind CSS

### Configuration Highlights

```typescript
{
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: { extend: {} },
  plugins: [],
}
```

## Deployment

### Vercel (Recommended for Next.js)

```bash
npm install -g vercel
vercel
```

### Docker

```bash
docker build -t ss-argitech-frontend .
docker run -p 3000:3000 ss-argitech-frontend
```

### Docker Compose

```bash
docker-compose up frontend
```

### Manual Deployment

1. Build: `npm run build`
2. Deploy `out/` or `.next/` directory
3. Set environment variables
4. Run: `npm start`

## Development Commands

```bash
npm run dev          # Start dev server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
```

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Troubleshooting

**API connection failed**

- Verify backend is running on port 5000
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Check CORS settings in backend

**AI image analysis failed**

- Verify the Django AI backend is running on `http://localhost:8000`
- Check `NEXT_PUBLIC_AI_API_URL` in `.env.local`
- Start the AI backend with the project Python environment, not a global Python install

**Build errors**

- Delete `.next` folder: `rm -rf .next`
- Reinstall dependencies: `npm install`
- Clear cache: `npm cache clean --force`

**Port already in use**

- Change port: `npm run dev -- -p 3001`
- Or kill process: `lsof -i :3000`

## Contributing

- Use functional components
- Add 'use client' directive when using hooks
- Follow Tailwind naming conventions
- Add TypeScript types for all variables
- Keep components focused and reusable

## License

MIT
