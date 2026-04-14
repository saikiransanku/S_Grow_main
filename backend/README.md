# SS Argitech Backend - README

## Project Overview

The backend is a Node.js Express server with TypeScript, PostgreSQL, and Prisma ORM. It provides RESTful APIs for authentication, user management, farmer laws reference, and usage tracking.

## Getting Started

### Prerequisites

- Node.js 18+
- PostgreSQL 12+
- npm or yarn

### Installation

```bash
cd backend
npm install
cp .env.example .env
```

### Configuration

Update `.env` file with your database connection:

```
DATABASE_URL="postgresql://user:password@localhost:5432/ss_argitech"
JWT_SECRET="your-secret-key"
PORT=5000
NODE_ENV="development"
```

### Database Setup

```bash
# Initialize database and run migrations
npm run db:push

# Or use Prisma migrations
npm run db:migrate
```

### Running the Server

**Development:**

```bash
npm run dev
```

**Production:**

```bash
npm run build
npm start
```

## Project Structure

```
backend/
├── src/
│   ├── index.ts           # Server entry point
│   ├── routes/
│   │   ├── auth.ts        # Authentication endpoints
│   │   ├── users.ts       # User management
│   │   ├── laws.ts        # Farmer laws
│   │   └── history.ts     # Usage history
│   ├── middleware/
│   │   └── auth.ts        # JWT authentication middleware
│   └── lib/
│       └── database.ts    # Database connection
├── prisma/
│   └── schema.prisma      # Database schema
└── dist/                  # Compiled JavaScript
```

## API Endpoints

### Authentication Routes

**Register User**

```
POST /api/auth/register
Content-Type: application/json

{
  "email": "farmer@example.com",
  "password": "securepassword",
  "firstName": "John",
  "lastName": "Doe"
}
```

**Login User**

```
POST /api/auth/login
Content-Type: application/json

{
  "email": "farmer@example.com",
  "password": "securepassword"
}
```

Response:

```json
{
  "message": "Login successful",
  "token": "jwt_token_here",
  "user": { ... }
}
```

### User Routes

**Get All Users**

```
GET /api/users
```

**Get User by ID**

```
GET /api/users/:id
Authorization: Bearer token
```

**Update User**

```
PUT /api/users/:id
Authorization: Bearer token
Content-Type: application/json

{
  "firstName": "John",
  "phone": "9876543210",
  "city": "Mumbai"
}
```

### Farmer Laws Routes

**Get All Laws**

```
GET /api/laws
```

**Get Laws by Category**

```
GET /api/laws/category/:category
```

**Get Specific Law**

```
GET /api/laws/:id
```

**Create Law (Admin)**

```
POST /api/laws
Authorization: Bearer token
Content-Type: application/json

{
  "title": "Farm Laws 2021",
  "description": "New agricultural laws",
  "category": "Agriculture",
  "content": "Full content here...",
  "source": "Government",
  "link": "https://example.com"
}
```

### History Routes

**Get User History**

```
GET /api/history/user/:userId
Authorization: Bearer token
```

**Log Action**

```
POST /api/history
Authorization: Bearer token
Content-Type: application/json

{
  "userId": "user_id",
  "action": "viewed_law",
  "data": "law_id"
}
```

## Database Models

### User

```prisma
model User {
  id        String     @id @default(cuid())
  email     String     @unique
  password  String
  firstName String
  lastName  String
  phone     String?
  address   String?
  city      String?
  state     String?
  pincode   String?
  profile   Profile?
  usageHistory UsageHistory[]
  createdAt DateTime  @default(now())
  updatedAt DateTime  @updatedAt
}
```

### Profile

```prisma
model Profile {
  id        String    @id @default(cuid())
  userId    String    @unique
  user      User      @relation(...)
  farmSize  Float?
  cropType  String?
  bio       String?
  avatar    String?
  createdAt DateTime  @default(now())
  updatedAt DateTime  @updatedAt
}
```

### FarmerLaw

```prisma
model FarmerLaw {
  id          String    @id @default(cuid())
  title       String
  description String    @db.Text
  category    String
  content     String    @db.Text
  source      String?
  link        String?
  createdAt   DateTime  @default(now())
  updatedAt   DateTime  @updatedAt
}
```

### UsageHistory

```prisma
model UsageHistory {
  id        String    @id @default(cuid())
  userId    String
  user      User      @relation(...)
  action    String
  data      String?
  createdAt DateTime  @default(now())
}
```

## Environment Variables

| Variable       | Description                          |
| -------------- | ------------------------------------ |
| `DATABASE_URL` | PostgreSQL connection string         |
| `JWT_SECRET`   | Secret key for JWT tokens            |
| `PORT`         | Server port (default: 5000)          |
| `NODE_ENV`     | Environment (development/production) |

## Development Tips

1. **Auto-reload**: The development server uses nodemon and tsc watch
2. **Type Safety**: Always use TypeScript for new files
3. **Validation**: Use express-validator for input validation
4. **Error Handling**: Wrap async operations in try-catch
5. **Prisma Studio**: Run `npm run db:studio` to visualize database

## Deployment

### Using Docker

```bash
docker build -t ss-argitech-backend .
docker run -p 5000:5000 -e DATABASE_URL="..." ss-argitech-backend
```

### Using Docker Compose

```bash
docker-compose up backend
```

### On Cloud Platforms

**Vercel/Railway/Render:**

1. Connect your GitHub repository
2. Set environment variables
3. Deploy with `npm start`

## Troubleshooting

**Database connection failed**

- Verify PostgreSQL is running
- Check `DATABASE_URL` in `.env`
- Ensure database exists: `createdb ss_argitech`

**Port already in use**

- Change `PORT` in `.env`
- Or kill existing process: `lsof -i :5000`

**Migrations not applied**

- Run: `npm run db:migrate`
- Or reset: `npm run db:push`

## Contributing

- Follow TypeScript strict mode rules
- Add input validation to all endpoints
- Include error handling
- Document new routes in this README

## License

MIT
