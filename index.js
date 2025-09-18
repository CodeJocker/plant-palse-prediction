import express from 'express';
import multer from 'multer';
import fs from 'fs';
import 'dotenv/config';
import cors from "cors"
import os from "os";
import path from "path";    

import { GoogleGenAI, HarmCategory, HarmBlockThreshold } from '@google/genai';

const app = express();

const upload = multer({
  dest: path.join(os.tmpdir(), "uploads/"), // e.g., '/tmp/uploads/' on Vercel
});
app.use(
  cors({
    origin: "*", // or ["http://localhost:3000"] for your React app only
    methods: ["GET", "POST", "PUT", "DELETE"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// Use environment variable for API key
const apiKey = process.env.GOOGLE_API_KEY;
if (!apiKey) {
    console.error('GOOGLE_API_KEY is not set in the environment variables.');
    process.exit(1); // Exit if no API key
}

const genAI = new GoogleGenAI({
    apiKey: apiKey
});

console.log('Initializing Gemini client with API key from environment.');

app.get('/', (req, res) => {
    console.log('Health check hit');
    res.send('🌱 Plant Disease Detection Server is running ✅');
});

app.post('/predict', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).send({
            success: false,
            message: 'No file uploaded.',
        });
    }

    const filePath = req.file.path;
    console.log('Received file:', req.file.originalname);

    try {
        const imageBuffer = fs.readFileSync(filePath);
        console.log('Image buffer read successfully.');

        const imagePart = {
            inlineData: {
                data: Buffer.from(imageBuffer).toString('base64'),
                mimeType: req.file.mimetype,
            },
        };

        const textPrompt = `
You are a plant doctor specialized in helping farmers.
Analyze the following image of a plant leaf and provide practical advice.

Please return the response in three clear sections:

1. Disease Name: Name the disease(s) affecting this leaf (if any). Keep it simple and understandable.
2. Causes: Explain in simple terms why this disease may happen.
3. Prevention & Remedies: Provide clear, practical steps a farmer can follow to prevent and treat the disease.

Format your response exactly like this:

Disease Name:
Causes:
Prevention & Remedies:
`;

        const contents = [{
            role: 'user',
            parts: [
                imagePart,
                { text: textPrompt }
            ]
        }];

        console.log('Sending multimodal prompt to Gemini...');

        const result = await genAI.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: contents,
            safetySettings: [
                {
                    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
            ],
        });

        if (result.promptFeedback?.blockReason) {
            console.error('Request payload:', JSON.stringify({
                model: 'gemini-2.5-flash',
                contents: contents,
            }, null, 2));
            throw new Error(`Content generation blocked: ${result.promptFeedback.blockReason}`);
        }

        if (!result.candidates || result.candidates.length === 0) {
            throw new Error('No candidates returned from API.');
        }

        const responseText = result.candidates[0].content.parts[0].text;
        console.log('Received response from Gemini:', responseText);

        res.send({
            success: true,
            result: responseText,
        });

    } catch (error) {
        console.error('Error during prediction:', error);
        res.status(500).send({
            success: false,
            message: 'Error processing the prediction',
            error: error.message,
        });
    } finally {
        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
            console.log('Uploaded file deleted');
        }
    }
});

// Add a test endpoint to verify Gemini API connection
app.get('/test-gemini', async (req, res) => {
    try {
        const testPrompt = 'Hello, Gemini!';
        const contents = [{
            role: 'user',
            parts: [{ text: testPrompt }]
        }];

        const result = await genAI.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: contents,
            safetySettings: [
                {
                    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold: HarmBlockThreshold.BLOCK_NONE,
                },
            ],
        });

        if (result.promptFeedback?.blockReason) {
            console.error('Test request payload:', JSON.stringify({
                model: 'gemini-2.5-flash',
                contents: contents,
            }, null, 2));
            throw new Error(`Test content generation blocked: ${result.promptFeedback.blockReason}`);
        }

        if (!result.candidates || result.candidates.length === 0) {
            throw new Error('No candidates returned from API in test.');
        }

        const responseText = result.candidates[0].content.parts[0].text;
        console.log('Test response from Gemini:', responseText);
        res.send({
            success: true,
            message: 'Gemini API is working correctly.',
            result: responseText,
        });
    } catch (error) {
        console.error('Error during Gemini API test interaction:', error);
        res.status(500).send({
            success: false,
            message: 'Error testing the Gemini API',
            error: error.message,
        });
    }
});

const PORT = process.env.PORT || 5005;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});