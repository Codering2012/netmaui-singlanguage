# Lesson Endpoint Usage Guide

## Endpoint

`GET /api/learn/lessons/{lessonId}`

Returns lesson metadata plus a nested `data` object for UI/demo payload.

---

## Authentication

This endpoint is protected with `[Authorize]`.

Include a valid JWT token:

```http
Authorization: Bearer <your-jwt-token>
```

---

## Request Example

```http
GET /api/learn/lessons/1 HTTP/1.1
Host: localhost:5157
Authorization: Bearer eyJhbGciOi...
```

---

## Successful Response (`200 OK`)

```json
{
  "id": 1,
  "title": "Introduction to Sign Language",
  "description": "Learn the basics of sign language.",
  "thumbnail": "https://example.com/thumbnail.jpg",
  "durationSeconds": 600,
  "difficulty": "Beginner",
  "completionPercentage": 50,
  "instructorName": "John Doe",
  "categoryId": 2,
  "data": {
    "durationSeconds": 600,
    "difficulty": "Beginner",
    "completionPercentage": 50,
    "instructorName": "John Doe",
    "categoryId": 2,
    "uiLayout": {
      "fileName": "LessonView.xaml",
      "xamlContent": "<ContentPage ...>",
      "codeBehindContent": "namespace App; ..."
    }
  }
}
```

---

## Camera Practice Lessons

The following lesson IDs return camera-practice UI layout payloads:

- `7` → `RealtimeHandSignalPracticeSet1View.xaml`
- `8` → `RealtimeHandSignalPracticeSet2View.xaml`
- `9` → `RealtimeHandSignalPracticeSet3View.xaml`

Use these with `POST /api/gesture/predict` for real-time sign testing flow.

---

## Error Responses

### `401 Unauthorized`
Missing/invalid JWT token.

### `404 Not Found`
Lesson ID does not exist.

```json
{ "message": "Lesson not found" }
```

### `500 Internal Server Error`
Server-side issue.

```json
{ "message": "Error fetching lesson" }
```

---

## Quick Test with `curl`

```bash
curl -X GET "https://localhost:7001/api/learn/lessons/7" \
  -H "Authorization: Bearer <your-jwt-token>"
```

---

## Typical Client Flow

1. User logs in (`/api/auth/login`) and gets JWT.
2. Client calls `GET /api/learn/lessons/{lessonId}`.
3. Client renders lesson details (`title`, `description`, `thumbnail`).
4. Client optionally uses `data.uiLayout` as dynamic UI config for demo rendering.
5. For lesson IDs `7-9`, client can open camera UI and call `/api/gesture/predict`.
