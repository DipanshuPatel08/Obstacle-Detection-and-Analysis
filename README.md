# 🤖 Obstacle Detection and Scene Analysis using Gemini API

A Flask web application that allows users to upload images for **robotics-oriented obstacle detection** and receive intelligent scene analysis using **Google's Gemini 1.5 Flash** generative AI.

---

## 🌟 Features

- 📤 Upload `.jpg`, `.jpeg`, or `.png` image files
- 🧠 Scene analysis for robotics, HARC, and navigation planning
- 🔍 Obstacle detection using OpenCV (edge & contour-based)
- 📈 Provides structured data (bounding boxes, object area, etc.)
- 📝 AI-generated scene description using Gemini 1.5 Flash
- 🔒 Flask session and file validation support
- 📦 API-friendly JSON response structure

---

## 🛠 Tech Stack

| Component        | Tech                             |
|------------------|----------------------------------|
| Web Framework    | Flask                            |
| Image Processing | OpenCV, Pillow (PIL)             |
| AI Integration   | Google Generative AI (Gemini)    |
| Env Management   | python-dotenv                    |
| Frontend         | HTML + JS + Bootstrap (templated)|